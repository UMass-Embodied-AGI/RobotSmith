import ast
import os
import json
import logging
import random
import time
import re
import trimesh

import backoff
import os
import base64
from PIL import Image
from io import BytesIO
from typing import Union
import traceback
import pickle
import subprocess

import open3d as o3d
import numpy as np
import math

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--task_name', type=str)
argparser.add_argument('--task_prompt_json_dir', type=str)
args = argparser.parse_args()


log_dir = os.path.join(project_path, args.task_name, 'trial')
os.makedirs(log_dir, exist_ok=True)
n_tries = len([fil for fil in os.listdir(log_dir) if not '.' in fil])
log_dir = os.path.join(log_dir, f"{n_tries:03d}")
os.makedirs(log_dir, exist_ok=True)

def encode_image(img: Union[str, Image.Image]) -> str:
    if isinstance(img, str): # if it's image path, open and then encode/decode
        with open(img, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    elif isinstance(img, Image.Image): # if it's image already, buffer and then encode/decode
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        raise Exception("img can only be either str or Image.Image")

class Generator:
    def __init__(self, lm_source, lm_id, max_tokens=4096, temperature=0.7, top_p=1, logger=None):
        self.lm_source = lm_source
        self.lm_id = lm_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.logger = logger
        self.caller_analysis = {}
        if self.logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG)
            self.logger.addHandler(logging.StreamHandler())
        self.max_retries = 3
        self.cost = 0 # cost in us dollars
        self.cache_path = f"cache_{lm_id}.pkl"
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                self.cache = pickle.load(f)
        else:
            self.cache = {}
        if self.lm_id == "gpt-4o":
            self.input_token_price = 2.5 * 10 ** -6
            self.output_token_price = 10 * 10 ** -6
        elif self.lm_id == "o3-mini":
            self.input_token_price = 1.1 * 10 ** -6
            self.output_token_price = 4.4 * 10 ** -6
        elif self.lm_id == "gpt-35-turbo":
            self.input_token_price = 1 * 10 ** -6
            self.output_token_price = 2 * 10 ** -6
        else:
            self.input_token_price = -1 * 10 ** -6
            self.output_token_price = -2 * 10 ** -6
        if self.lm_source == "openai":
            from openai import OpenAI
            self.client = OpenAI(
                api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
                max_retries=self.max_retries,
            ) if 'OPENAI_API_KEY' in os.environ else None
        elif self.lm_source == "azure":
            from openai import AzureOpenAI
            try:
                api_keys = json.load(open(os.path.join(project_path, "api_keys.json"), "r"))
                if "embedding" in self.lm_id:
                    api_keys = api_keys["embedding"]
                else:
                    api_keys = api_keys["all"]
                api_keys = random.sample(api_keys, 1)[0]
                self.logger.info(f"Using Azure API key: {api_keys['AZURE_ENDPOINT']}")
                self.client = AzureOpenAI(
                    azure_endpoint=api_keys['AZURE_ENDPOINT'],
                    api_key=api_keys['OPENAI_API_KEY'],
                    api_version="2024-12-01-preview",
                )
            except Exception as e:
                self.logger.error(f"Error loading .api_keys.json: {e} with traceback: {traceback.format_exc()}")
                self.client = None
        elif self.lm_source == "huggingface":
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            # self.client = AutoModelForCausalLM.from_pretrained(self.lm_id)
            # self.tokenizer = AutoTokenizer.from_pretrained(self.lm_id)
            self.client = pipeline(
                "text-generation",
                model=self.lm_id,
                device_map="auto",
            )
            # lm_id: "meta-llama/Meta-Llama-3.1-8B-Instruct"
        elif self.lm_source == "llava":
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
            from llava.constants import (
                IMAGE_TOKEN_INDEX,
                DEFAULT_IMAGE_TOKEN,
                DEFAULT_IM_START_TOKEN,
                DEFAULT_IM_END_TOKEN,
                IMAGE_PLACEHOLDER,
            )
            from llava.conversation import conv_templates, SeparatorStyle
            import torch
            from llava.mm_utils import (
                process_images,
                tokenizer_image_token,
                get_model_name_from_path,
                KeywordsStoppingCriteria,
            )
            self.model_name = get_model_name_from_path(self.lm_id)
            if 'lora' in self.model_name and '7b' in self.model_name:
                self.lm_base = "liuhaotian/llava-v1.5-7b"
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=self.lm_id, model_base=self.lm_base, model_name=self.model_name, )  # load_4bit=True)
        elif self.lm_source == "vla": # will merge to huggingface later
            from transformers import AutoModelForVision2Seq, AutoProcessor
            from peft import PeftModel
            import torch
            self.processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
            self.base_model = AutoModelForVision2Seq.from_pretrained(
                "openvla/openvla-7b",
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            self.lora_model = PeftModel.from_pretrained(
                self.base_model,
                "/home/zheyuanzhang/Documents/GitHub/VLA/adapter_tmp/openvla-7b+ella_dataset+b16+lr-0.0005+lora-r32+dropout-0.0", # will add the lora adapter path into env config later
                torch_dtype=torch.bfloat16,
                device_map="auto",
            ).to("cuda:0")
        elif self.lm_source == "google":
            import google.generativeai as genai
            from google.generativeai.types import HarmCategory, HarmBlockThreshold
            from google.api_core.exceptions import ResourceExhausted
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            self.client = genai.GenerativeModel(self.lm_id)
        elif self.lm_source == "local":
            from openai import OpenAI
            from tools.model_manager import global_model_manager
            self.client = global_model_manager.get_model("completion")
            self.embed_client = global_model_manager.get_model("embedding")
            self.input_token_price = 0
            self.output_token_price = 0
        else:
            raise NotImplementedError(f"{self.lm_source} is not supported!")

    def generate(self, prompt, max_tokens=None, temperature=None, top_p=None, img=None, json_mode=False, chat_history=None, caller="none"):
        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p
            
        if self.lm_source == 'openai' or self.lm_source == 'azure':
            return self.openai_generate(prompt, max_tokens, temperature, top_p, img, json_mode, chat_history, caller)
        elif self.lm_source == 'gemini':
            return self.gemini_generate(prompt)
        elif self.lm_source == 'huggingface':
            return self.huggingface_generate(prompt, max_tokens, temperature, top_p)
        elif self.lm_source == 'vla':
            return self.vla_generate(prompt, img, max_tokens)
        elif self.lm_source == 'local':
            message = [] if chat_history is None else chat_history
            message.append({ "role": "user", "content": prompt })
            return self.client.complete(message, max_tokens, temperature, top_p)
        else:
            raise ValueError(f"Invalid lm_source: {self.lm_source}")

    def openai_generate(self, prompt, max_tokens, temperature, top_p, img: Union[str, Image.Image, None, list], json_mode=False, chat_history=None, caller="none"):
        @backoff.on_exception(
            backoff.expo,  # Exponential backoff
            Exception,  # Base exception to catch and retry on
            max_tries=self.max_retries,  # Maximum number of retries
            jitter=backoff.full_jitter,  # Add full jitter to the backoff
            logger=self.logger  # Logger for retry events, which is in the level of INFO
        )
        def _generate():
            content = [{
                        "type": "text",
                        "text": prompt
                    }, ]
            if img is not None:
                if type(img) != list:
                    imgs = [img]
                else:
                    imgs = img
                for each_img in imgs:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode_image(each_img)}"},
                        # "detail": "low"
                    })
            if chat_history is not None:
                messages = chat_history
            else:
                messages = []
            messages.append(
                {
                    "role": "user",
                    "content": content
                })
            start = time.perf_counter()
            if self.lm_id[0] == 'o':
                response = self.client.chat.completions.create(
                    # reasoning_effort='high',
                    model=self.lm_id,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    timeout=40,
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.lm_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    response_format={
                        "type": "json_object" if json_mode else "text"
                    },
                    timeout=40,
                )
            self.logger.debug(f"api request time: {time.perf_counter() - start}")
            with open(f"chat_raw.jsonl", 'a') as f:
                chat_entry = {
                    "prompt": prompt,
                    "response": response.model_dump_json(indent=4)
                }
                # Write as a single JSON object per line
                f.write(json.dumps(chat_entry))
                f.write('\n')
            usage = dict(response.usage)
            self.cost += usage['completion_tokens'] * self.output_token_price + usage['prompt_tokens'] * self.input_token_price
            if caller in self.caller_analysis:
                self.caller_analysis[caller].append(usage['total_tokens'])
            else:
                self.caller_analysis[caller] = [usage['total_tokens']]
            response = response.choices[0].message.content
            # self.logger.debug(f'======= prompt ======= \n{prompt}', )
            # self.logger.debug(f'======= response ======= \n{response}')
            # self.logger.debug(f'======= usage ======= \n{usage}')
            if self.cost > 7:
                self.logger.critical(f'COST ABOVE 7 dollars! There must be sth wrong. Stop the exp immediately!')
                raise Exception(f'COST ABOVE 7 dollars! There must be sth wrong. Stop the exp immediately!')
            self.logger.info(f'======= total cost ======= {self.cost}')
            return response
        try:
            return _generate()
        except Exception as e:
            self.logger.error(f"Error with openai_generate: {e}, the prompt was:\n {prompt}")
            return None

    def gemini_generate(self, prompt):
        try:
            response = self.client.generate_content(prompt, safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            })
            usage = response.usage_metadata.total_token_count
            self.cost += usage * self.input_token_price
            response = response.text
            self.logger.debug(f'======= prompt ======= \n{prompt}', )
            self.logger.debug(f'======= response ======= \n{response}')
            self.logger.debug(f'======= usage ======= \n{usage}')
            self.logger.debug(f'======= total cost ======= {self.cost}')
        except Exception as e:
            self.logger.error(f"Error generating content: {e}")
            raise e
        return response

    def huggingface_generate(self, prompt, max_tokens, temperature, top_p):
        messages = []
        messages.append(
            {
                "role": "system",
                "content": "You are a helpful assistant."
            })
        messages.append(
            {
                "role": "user",
                "content": prompt
            })
        response = self.client(
            prompt,
            do_sample = False if temperature == 0 else True,
            temperature=temperature if temperature != 0 else 1,
            top_p=top_p,
            max_new_tokens=max_tokens,)
        response = response[0]['generated_text']
        self.logger.debug(f'======= prompt ======= \n{prompt}', )
        self.logger.debug(f'======= response ======= \n{response}')
        return response
        # inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True, padding="max_length", max_length=self.max_tokens, truncation=True)
        # outputs = self.client.generate(**inputs, max_length=self.max_tokens, num_return_sequences=1, temperature=self.temperature, top_p=self.top_p)
        # response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # self.logger.debug(f'======= prompt ======= \n{prompt}', )
        # self.logger.debug(f'======= response ======= \n{response}')

    def vla_generate(self, prompt, img, max_tokens):
        import torch
        inputs = self.processor(prompt, Image.fromarray(img)).to("cuda:0", dtype=torch.bfloat16)
        with torch.no_grad():
            outputs = self.lora_model.generate(**inputs, max_length=max_tokens)
        return self.processor.decode(outputs[0], skip_special_tokens=True)

designer = Generator(
    lm_source='azure',
    lm_id='o3-mini',
    max_tokens=16000,
    temperature=0.7,
    top_p=1.0,
    logger=None
)
critic = Generator(
    lm_source='azure',
    lm_id='gpt-4o',
    max_tokens=16000,
    temperature=0.7,
    top_p=1.0,
    logger=None
)

def parse_json(prompt, response, last_call=False):
    json_str = None
    if "```json" in response:
        # Step 1: Extract the JSON part
        start = response.find("```json") + len("```json")
        end = response.find("```", start)
        json_str = response[start:end].strip()
    else:
        if not last_call:
            chat_history = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            data = designer.generate(
                f"The output format is wrong. Output the formatted json string enclosed in ```json``` only! Do not include any other character in the output!", chat_history=chat_history)
            return parse_json(None, data, last_call=True)
        else:
            return None
    try:
        response = json.loads(json_str)
    except json.JSONDecodeError as e:
        if not last_call:
            chat_history = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            data = designer.generate(
                f"The output format is wrong. Output the formatted json string enclosed in ```json``` only! Do not include any other character in the output!", chat_history=chat_history)
            return parse_json(None, data, last_call=True)
    return response


def write_design_code(filename, tool_json):
    outp = ''
    print('tool_json', tool_json)   
    with open(os.path.join(project_path, 'utils', 'api_tool_design.py'), 'r') as fi:
        for line in fi.readlines():
            outp += line
        outp += '\n\n\n\n\n'

    outp += tool_json['assemble_func']
    outp += '\n\n\n'

    outp += 'parts = '
    parts = json.dumps(tool_json['parts'], indent=4)
    parts = parts.replace('true', 'True')
    parts = parts.replace('false', 'False')
    outp += parts
    outp += '\n'   
    outp += 'filenames = assemble(parts)\n'
    outp += 'print(filenames)\n'
    
    print(outp)

    with open(filename, 'w') as fo:
        fo.write(outp)

def look_at(cam_pos, target=np.array([0, 0, 0]), up=np.array([0, 0, 1])):
    forward = (target - cam_pos)
    forward /= np.linalg.norm(forward)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    R = np.eye(4)
    R[:3, :3] = np.stack([right, up, -forward], axis=1)
    R[:3, 3] = -cam_pos
    return np.linalg.inv(R)

def render_and_save(mesh_path, output_folder, num_views=10):
    os.makedirs(output_folder, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    mesh.translate(-mesh.get_center())        
    vertices = np.asarray(mesh.vertices)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=640, height=480)
    vis.add_geometry(mesh)
    opt = vis.get_render_option()
    opt.light_on = True
    opt.background_color = np.array([1, 1, 1]) 

    for i in range(num_views):
        # Random camera position on a sphere
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        radius = max(abs(vertices.min()), vertices.max()) * 3
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        cam_pos = np.array([x, y, z])

        # Compute extrinsic
        extrinsic = look_at(cam_pos)

        # Set the view
        ctr = vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        param.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(param)

        vis.poll_events()
        vis.update_renderer()

        # Save image
        img = vis.capture_screen_float_buffer(False)
        img = (255 * np.asarray(img)).astype(np.uint8)
        o3d.io.write_image(f"{output_folder}/{i:03d}.png", o3d.geometry.Image(img))

    vis.destroy_window()

def render_and_save_with_objects(mesh_path, json_filename, output_folder, num_views=10):
    os.makedirs(output_folder, exist_ok=True)

    tool_json = json.load(open(json_filename, 'r'))
    tool_placement = tool_json["placement_func"]
    _, p1, p2, p3 = re.findall(r"(pos)\s*=\s*\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)", tool_placement)[0]
    _, e1, e2, e3 = re.findall(r"(euler)\s*=\s*\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)", tool_placement)[0]
    _, s1, s2, s3 = re.findall(r"(scale)\s*=\s*\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)", tool_placement)[0]
    tool_pos = np.array([float(p1), float(p2), float(p3)])
    tool_euler = np.array([float(e1), float(e2), float(e3)])
    tool_scale = np.array([float(s1), float(s2), float(s3)])
    mesh = trimesh.load(mesh_path)
    mesh.apply_scale(tool_scale)
    rotation = trimesh.transformers.euler_matrix(
        math.radians(float(e1)), math.radians(float(e2)), math.radians(float(e3)), axes='sxyz'
    )
    mesh.apply_transform(rotation)
    mesh.apply_translation(tool_pos)
    new_mesh_path = mesh_path[:-4], '_loaded.obj'
    mesh.export(new_mesh_path)

    mesh = o3d.io.read_triangle_mesh(new_mesh_path)
    mesh.compute_vertex_normals()
    mesh.translate(-mesh.get_center())        
    vertices = np.asarray(mesh.vertices)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=640, height=480)
    vis.add_geometry(mesh)
    opt = vis.get_render_option()
    opt.light_on = True
    opt.background_color = np.array([1, 1, 1]) 

    for i in range(num_views):
        # Random camera position on a sphere
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        radius = max(abs(vertices.min()), vertices.max()) * 3
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        cam_pos = np.array([x, y, z])

        # Compute extrinsic
        extrinsic = look_at(cam_pos)

        # Set the view
        ctr = vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        param.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(param)

        vis.poll_events()
        vis.update_renderer()

        # Save image
        img = vis.capture_screen_float_buffer(False)
        img = (255 * np.asarray(img)).astype(np.uint8)
        o3d.io.write_image(f"{output_folder}/{i:03d}.png", o3d.geometry.Image(img))

    vis.destroy_window()

def render_and_save_with_genesis(mesh_path, output_folder, num_views=10):
    os.makedirs(output_folder, exist_ok=True)
    scene_envname = 'Reaching'
    prog = f"import genesis as gs\nclass Env({scene_envname}):\n        "

    with open('tmp.py', 'w') as fo:
        fo.write(prog)
    



def run_tool_design(task_name, task_prompt_json_dir):
    
    designer_prompt =  open(os.path.join(project_path, 'utils', 'template_tool_design.txt'), 'r').read()
    designer_prompt_json = json.load(open(task_prompt_json_dir, 'r'))
    designer_prompt = designer_prompt.replace("$3D_OBJECT_DESCRIPTION$", designer_prompt_json['3D_OBJECT_DESCRIPTION'])
    designer_prompt = designer_prompt.replace("$GOAL_DESCRIPTION$", designer_prompt_json['GOAL_DESCRIPTION'])
    designer_prompt = designer_prompt.replace("$3D_CONFIGURATION$", designer_prompt_json['3D_CONFIGURATION'])
    designer_prompt = designer_prompt.replace("$TIPS_FOR_DESIGNER$", designer_prompt_json['TIPS_FOR_DESIGNER'])
    
    designer_response = designer.generate(prompt=designer_prompt, img=None, json_mode=False)

    critic_cnt = 0
    design_chat_history = [{
        'role': 'user',
        'content': designer_prompt
    }]
    while critic_cnt <= 5:
        critic_cnt += 1
        try:
            designer_response = parse_json(designer_prompt, designer_response)
            json_filename = os.path.join(log_dir, f"design{critic_cnt}.json")
            json.dump(designer_response, open(json_filename, 'w'), indent=4)
            code_filename = os.path.join(log_dir, f"design{critic_cnt}.py")
            write_design_code(code_filename, designer_response)
            
            design_chat_history.append({
                'role': 'assistant',
                'content': json.dumps(designer_response)
            })

            result = subprocess.run(
                ["python3", code_filename],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                raise Exception(f"Error in subprocess: {result.stderr}")
            print(result.stdout)
            output_files = ast.literal_eval(result.stdout)
            assert isinstance(output_files, list), "Output files should be a list"
            os.system(f"mkdir {log_dir}/{critic_cnt}")
            
            imgs = []
            for output_file in output_files:
                os.system(f"cp {output_file} {log_dir}/{critic_cnt}/")
                render_and_save_with_objects(f"{log_dir}/{critic_cnt}/{output_file}", json_filename, f"{log_dir}/{critic_cnt}/rendered_views", num_views=5, )
            
                for i in range(5):
                    img_path = os.path.join(f"{log_dir}/{critic_cnt}/rendered_views", f"{i:03d}.png")
                    imgs.append(img_path)

            critic_prompt =  open(os.path.join(project_path, 'utils', 'template_tool_critic.txt'), 'r').read()
            critic_prompt = critic_prompt.replace("$3D_OBJECT_DESCRIPTION$", designer_prompt_json['3D_OBJECT_DESCRIPTION'])
            critic_prompt = critic_prompt.replace("$GOAL_DESCRIPTION$", designer_prompt_json['GOAL_DESCRIPTION'])
            critic_prompt = critic_prompt.replace("$3D_CONFIGURATION$", designer_prompt_json['3D_CONFIGURATION'])
    
            critic_response = critic.generate(prompt=critic_prompt, img=imgs, json_mode=False)
            designer_prompt = critic_response
            print('critic+response', designer_prompt)
            if 'DONE' in critic_response:
                break
            designer_response = designer.generate(prompt=designer_prompt, img=None, json_mode=False, chat_history=design_chat_history)

            design_chat_history.append({
                'role': 'user',
                'content': designer_prompt
            })

        except Exception as e:
            print(f"Error in critic {critic_cnt}: {e} with traceback: {traceback.format_exc()}")
            break

    planing_chat_history = [design_chat_history[0], design_chat_history[-1]]
    planing_prompt = open(os.path.join(project_path, 'utils', 'template_manipulate.txt'), 'r').read()
    planing_prompt = planing_prompt.replace("$3D_OBJECT_DESCRIPTION$", designer_prompt_json['3D_OBJECT_DESCRIPTION'])
    planing_prompt = planing_prompt.replace("$GOAL_DESCRIPTION$", designer_prompt_json['GOAL_DESCRIPTION'])
    planing_prompt = planing_prompt.replace("$3D_CONFIGURATION$", designer_prompt_json['3D_CONFIGURATION'])

    planing_response = designer.generate(prompt=planing_prompt, img=None, json_mode=False, chat_history=planing_chat_history)
    with open(os.path.join(log_dir, 'plan.txt'), 'w') as fo:
        fo.write(planing_response)

if __name__ == "__main__":

    run_tool_design(task_name=args.task_name, task_prompt_json_dir=args.task_prompt_json_dir)