import os
import json 

project_path = '/home/xhrlyb/Projects/FuncAny'

tid = 3

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

designer_response = json.load(open(f'proposer{tid}.json', 'r'))

code_filename = f"design{tid}.py"
write_design_code(code_filename, designer_response)

# os.system(f'python3 {code_filename}')

# if result.returncode != 0:
#     raise Exception(f"Error in subprocess: {result.stderr}")
# print(result.stdout)
# output_files = ast.literal_eval(result.stdout)
# assert isinstance(output_files, list), "Output files should be a list"
# os.system(f"mkdir {log_dir}/{critic_cnt}")

# imgs = []
# for output_file in output_files:
#     os.system(f"cp {output_file} {log_dir}/{critic_cnt}/")
#     render_and_save_with_objects(f"{log_dir}/{critic_cnt}/{output_file}", json_filename, f"{log_dir}/{critic_cnt}/rendered_views", num_views=5, )

#     for i in range(5):
#         img_path = os.path.join(f"{log_dir}/{critic_cnt}/rendered_views", f"{i:03d}.png")
#         imgs.append(img_path)