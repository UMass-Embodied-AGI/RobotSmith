import os
import re
import ast
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

def write_code(file_id, tool_placement, code_file, tool_file, plan_file, template_file):
    with open(template_file, 'r') as fi:
        template_lines = fi.readlines()
    outp = ''

    # before tool loading
    line_id = 0
    for line in template_lines:
        line_id += 1        
        outp += line
        if ') # dough' in line:
            break
    
    # skip tool loading
    while True:
        line_id += 1
        if ') # tool' in template_lines[line_id-1]:
            break

    # tool loading 
    scale = tool_placement['scale']
    pos = tool_placement['pos']
    euler = tool_placement['euler']
    outp += f"""
    tool = scene.add_entity(
        morph = gs.morphs.Mesh(
            file=\'{tool_file}\',
            scale=({scale[0]}, {scale[1]}, {scale[2]}),
            pos=({pos[0]}, {pos[1]}, {pos[2]}),
            euler=({euler[0]}, {euler[1]}, {euler[2]}),
            fixed=False,
            collision=True,
            decompose_nonconvex=False,
            convexify=False,                                                                                                                                                                                                           
            decimate=True,                                                                                                                                                                                                             
            decimate_face_num=3000, 
        ),
        surface=gs.surfaces.Default(
            color    = (0.1, 0.1, 0.1),
        ),  
    ) # tool
"""

    template_lines = template_lines[line_id:]
    line_id = 0
    # before evaluate
    for line in template_lines:
        line_id += 1
        if 'def evaluate' in line:
            break
        outp += line

    # skip evaluate
    while True:
        line_id += 1
        if '# ) evaluate' in template_lines[line_id-1]:
            break
    
    # evaluate
    with open(plan_file, 'r') as f:
        raw_content = f.read()
    raw_content = raw_content.strip()
    
    if raw_content.startswith('```python') and raw_content.endswith('```'):
        raw_content = re.sub(r'^```python\s*', '', raw_content)
        raw_content = re.sub(r'```$', '', raw_content)
        raw_content = raw_content.strip()
        
    if '```python' in raw_content:
        begin = raw_content.find('```python')
        raw_content = raw_content[begin+9:]
    if '```' in raw_content:
        end = raw_content.find('```')
        raw_content = raw_content[:end]
        
    if raw_content.startswith('{') or raw_content.startswith('['):
        try:
            json_obj = json.loads(raw_content)
            if isinstance(json_obj, dict):
                code = list(json_obj.values())[0]
            elif isinstance(json_obj, str):
                code = json_obj
            else:
                raise ValueError("Unrecognized JSON structure for plan.txt")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in plan.txt")
    else:
        code = raw_content
    print(code)
    tree = ast.parse(code)
    func_node = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func_node = node
            break
    if func_node is None:
        raise ValueError("No execute_trajectory() function found.")
    found_grasp, found_grasp_ever = False, False
    adjust_calls = []
    for stmt in func_node.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            func_name = getattr(call.func, 'id', None)
            if func_name is None and isinstance(call.func, ast.Attribute):
                func_name = call.func.attr
            if func_name == 'grasp':
                print(ast.unparse(stmt))
                obj_arg = None
                if call.args:
                    obj_arg = call.args[0]
                else:
                    obj_kw = next((kw for kw in call.keywords if kw.arg == 'obj'), None)
                    if obj_kw:
                        obj_arg = obj_kw.value
                print(obj_arg   )
                if obj_arg is None:
                    raise ValueError("grasp() must have an object argument, either positional or obj= keyword!")
                if isinstance(obj_arg, ast.Attribute):
                    if 'tool' not in obj_arg.attr:
                        raise ValueError("grasp() must be called with self.tool or 'tool'! attr")
                elif isinstance(obj_arg, ast.Constant):
                    if 'tool' not in obj_arg.value:
                        raise ValueError("grasp() must be called with self.tool or 'tool'! const")
                elif isinstance(obj_arg, ast.Name):
                    if 'tool' not in obj_arg.id:
                        raise ValueError("grasp() must be called with self.tool or 'tool'! name")
                else:
                    raise ValueError("grasp() must be called with self.tool or 'tool'!  else")
                found_grasp = True
                found_grasp_ever = True
            elif func_name == 'release':
                found_grasp = False
            elif func_name == 'adjust_gripper_pose':
                if found_grasp:
                    adjust_calls.append(call)
    if not found_grasp_ever:
        raise ValueError("No grasp(self.tool) found!")
    poses = []
    quats = []
    assignments = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            value = node.value
            assignments[var_name] = value
    for call in adjust_calls:
        args = {kw.arg: kw.value for kw in call.keywords} if call.keywords else {}
        if args:
            pos_arg = args.get('pos', call.args[0] if call.args else None)
            quat_arg = args.get('quat', call.args[1] if len(call.args) > 1 else None)
        else:
            pos_arg = call.args[0]
            quat_arg = call.args[1]

        if isinstance(pos_arg, ast.Name) and pos_arg.id in assignments:
            pos_arg = assignments[pos_arg.id]
        if isinstance(quat_arg, ast.Name) and quat_arg.id in assignments:
            quat_arg = assignments[quat_arg.id]

        if not (isinstance(pos_arg, ast.Tuple) and isinstance(quat_arg, ast.Tuple)):
            raise ValueError("adjust_gripper_pose arguments must be tuples, variables assigned to tuples, or keyword arguments.")

        pos = tuple(ast.literal_eval(elt) for elt in pos_arg.elts)
        quat = tuple(ast.literal_eval(elt) for elt in quat_arg.elts)

        poses.append(pos)
        quats.append(quat)

    print(code)

    print("adjust_calls:", ast.unparse(adjust_calls))

    print("poses:", poses)

    print("quats:", quats)

    unique_quats = []
    unique_eulers = []
    quat_indices = []
    for quat in quats:
        if quat not in unique_quats:
            if abs(quat[0]) > 0.9:
                euler = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler('xyz', degrees=True)
            else:
                euler = R.from_quat([quat[0], quat[1], quat[2], quat[3]]).as_euler('xyz', degrees=True)
            unique_quats.append(quat)
            unique_eulers.append(tuple(euler))
        quat_indices.append(unique_quats.index(quat))
    
    print("unique_quats:", unique_quats)

    print("quat_indices:", quat_indices)

    x_init = []
    for uq in unique_eulers:
        x_init.extend(list(uq))
    for pos in poses:
        x_init.extend(list(pos))

    print("x_init:", x_init)

    lines = []
    lines.append(f"    def evaluate(x, cur_iter, vis, img_steps, img_save_dir):")
    lines.append(f"        x = np.array(x).reshape(({len(unique_quats) + len(poses)}, 3))")
    lines.append("        scene.reset()")
    lines.append("")
    lines.append("        tool_quats = x[:{}]".format(len(unique_quats)))
    lines.append("        tool_poses = x[{}:]".format(len(unique_quats)))
    lines.append("")
    for i, idx in enumerate(quat_indices):
        lines.append("        control_tool(tool, tool_poses[{}], tool_quats[{}])".format(i, idx))
    lines.append("")
    lines.append("        dough_particles = dough.get_particles()")
    lines.append("        penalty = dough_particles[:, 2].max()")
    lines.append("")
    lines.append("        return penalty, 0")
    for line in lines:
        outp += line + "\n"


    n_envs = 15
    project_path = '/scratch3/workspace/haotianyuan_umass_edu-shared/FuncAny'

    # n_envs = 3
    # project_path = '/home/xhrlyb/Projects/FuncAny'

    # after evaluate
    template_lines = template_lines[line_id:]
    line_id = 0
    for line in template_lines:
        line_id += 1
        if 'dim, ' in line:
            outp += f'    dim, rng, sigma, iters = {len(unique_quats) + len(poses)}, 0.1, 0.02, 50\n'
        elif 'init_params = [' in line:
            outp += f'    init_params = np.array({x_init}).flatten()\n'
        elif 'init_params = np' in line:
            pass
        elif 'project_path = ' in line:
            outp += f'    project_path = \'{project_path}\'\n'
        elif 'n_envs = ' in line:
            outp += f'    n_envs = {n_envs}\n'
        elif 'opt_log' in line:
            line = line.replace('opt_log', f'opt_log{file_id}')
            outp += line
            outp += f'    open(os.path.join(img_save_dir, \"{file_id}.txt\"), \"w\").write(\"{file_id}\")\n'
        elif 'best_traj.npy' in line:
            line = line.replace('best_traj.npy', f'best_traj{file_id}.npy')
            outp += line
        elif 'best_score.txt' in line:
            line = line.replace('best_score.txt', f'best_score{file_id}.txt')
            outp += line
        elif 'cmaes_ckpt.pkl' in line:
            line = line.replace('cmaes_ckpt.pkl', f'cmaes_ckpt{file_id}.pkl')
            outp += line
        elif 'plot.png' in line:
            line = line.replace('plot.png', f'plot{file_id}.png')
            outp += line
        else:
            outp += line
        

    with open(code_file, 'w') as fo:
        fo.write(outp)



# write_code(tool_placement, code_file, tool_file, plan_file, template_file='cmaes_tool_multiprocess.py'

prefix = "/home/xhrlyb/Projects/FuncAny/blenderkit/"
unity_prefix = '/scratch3/workspace/haotianyuan_umass_edu-shared/FuncAny/blenderkit/'

code_file = prefix + "task03_cmaes.py"
tool_file = unity_prefix + "task03.obj"
load_tool_file = "task03.py"
plan_file = prefix + "task03_init_traj.txt"
template_file = "/home/xhrlyb/Projects/FuncAny/task03_flaten/cmaes_tool_multiprocess.py"
file_id = '_blenderkit03'

def get_tool_placement(file):
    with open(file) as fi:
        lines = fi.readlines()
    lid = 0
    while ') # cam' not in lines[lid]: lid += 1
    tool_code = ''
    for line in lines[lid + 1:]:
        tool_code += line
        if ') # tool' in line:
            break
    print(tool_code)
    _, p1, p2, p3 = re.findall(r"(pos)\s*=\s*\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)", tool_code)[0]
    _, e1, e2, e3 = re.findall(r"(euler)\s*=\s*\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)", tool_code)[0]
    tool_placement = {
        'scale': (1, 1, 1),
        'pos': (p1, p2, p3),
        'euler': (e1, e2, e3)
    }
    return tool_placement
    

tool_placement = get_tool_placement(prefix + 'task03.py')

write_code(file_id, tool_placement, code_file, tool_file, plan_file, template_file)
   































# with open(cmaes_file, "r") as f:
#         cmaes_code = f.read()
#     eval_single_match = re.search(r"def evaluate_single\(self, poss\):(.+?)(?=^\s*def |\Z)", cmaes_code, flags=re.S)
#     if not eval_single_match:
#         raise ValueError("Cannot find evaluate_single method.")
#     body = eval_single_match.group(1)
#     if "my_grasp(" in body:
#         if not re.search(r"my_grasp\([^,]+, *[^,]+, *self\.tool", body):
#             raise ValueError("Error: my_grasp is not called with self.tool!")
#     else:
#         raise ValueError("No my_grasp found in evaluate_single.")
#     my_grasp_pos = body.index("my_grasp(")
#     release_pos = body.index("release(")
#     adjust_block = body[my_grasp_pos:release_pos]
#     adjust_calls = re.findall(r"adjust_gripper_pose_without_plan\([^\)]*pos=([^,]*),[^\)]*\)", adjust_block)
#     n_controls = len(adjust_calls)
#     new_function = [
#         "    def evaluate(self, x):",
#         "        x = x.reshape(-1, 6)",
#         "        scene.reset()",
#     ]
#     for idx in range(n_controls):
#         new_function.append(f"        self.control_tool(tool, pos=x[{idx}, :3], rot=x[{idx}, 3:])")
#     new_function.append("        dough_particles = dough.get_particles()")
#     new_function.append("        penalty = dough_particles[:, 2].max()")
#     new_function.append("        return penalty, 0")
#     for line in new_function:
#         outp += line + "\n"

#     # get dim & init_params
#     init_params_match = re.search(r"init_params\\s*=\\s*np\\.array\\((\\[.+?\\])\\)\\.flatten\\(\\)", code, flags=re.S)
#     if not init_params_match:
#         raise ValueError("Cannot find init_params in the code.")
#     init_params_str = init_params_match.group(1)
#     init_params_array = eval(init_params_str) 
#     new_init_params = init_params_array[3:] 
#     new_init_params_full = []
#     for pos in new_init_params:
#         new_init_params_full.extend(list(pos) + [0,0,0])
#     new_dim = len(new_init_params_full)

#     # after evaluate
#     template_lines = template_lines[line_id:]
#     line_id = 0
#     for line in template_lines:
#         line_id += 1
#         if 'dim, ' in line:
#             outp += f'    dim, rng, sigma, n_envs, iters = {n_controls*3}, 0.1, 0.02, 15, 50\n'
#         outp += line