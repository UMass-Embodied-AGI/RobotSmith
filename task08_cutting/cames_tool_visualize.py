import multiprocessing

import numpy as np
import genesis as gs
import time
import cma
import os
from itertools import accumulate
import imageio
import matplotlib.pyplot as plt

import argparse

argparse = argparse.ArgumentParser()
argparse.add_argument('-i', '--trial_id', type=int)
argparse.add_argument('-t', '--traj_path', type=str)
args = argparse.parse_args()

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__, '..')))

best_traj = np.load(project_path + '/task08_cutting/try/' + args.traj_path +'/'+ f'best_traj{args.trial_id}.npy')
img_dir = os.path.join(project_path, 'task08_cutting/try', args.traj_path, f'best_traj{args.trial_id}')
print("img_dir: ", img_dir)
os.makedirs(img_dir, exist_ok=True)


gs.init(seed=0, precision='32', logging_level='error')
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=2e-3, substeps=10, gravity=(0,0,-9.8),
    ),
    mpm_options=gs.options.MPMOptions(
        lower_bound=(-0.1,-0.1,-0.1), upper_bound=(0.6,0.6,0.6),
        grid_density=128,
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_fov=30, res=(960, 640),
    ),
    vis_options=gs.options.VisOptions(
        env_separate_rigid = True
    ),
    show_viewer = False,
)
plane = scene.add_entity(
    morph=gs.morphs.Plane(),
)

dough = scene.add_entity(
    material=gs.materials.MPM.ElastoPlastic(E=100000, nu=0.3, rho=600.0, yield_lower = 0.0025, yield_higher = 0.0045, use_von_mises=True, von_mises_yield_stress=2000),
    morph=gs.morphs.Box(
        upper=(0.42,  0.015, 0.03), 
        lower=(0.30, -0.015, 0.00)
    ),
    surface=gs.surfaces.Rough(
        color=(231.0 / 255.0, 211.0 / 255.0, 198.0 / 255.0),
        vis_mode='recon',
        recon_backend='splashsurf-1.5-smooth-25'
    ),
)
tool = scene.add_entity(
    morph = gs.morphs.Mesh(
        file="{}/task08_cutting_tools/0.obj".format(project_path),
        scale=(1, 1, 1),
        pos=(0.36, 0.0, 0.015),
        euler=(0, 0, 0),
        fixed=False,
        collision=True,
        decompose_nonconvex=False,
        convexify=False,
        decimate=False,
    ),
    surface=gs.surfaces.Default(
        color    = (0.1, 0.1, 0.1),
    ),  
)        
cam = scene.add_camera(
    pos=(0.4,-0.9,0.9), lookat=(0.4,0.0,0.0), fov=50, res=(1440,1440), GUI=False,
)

scene.build()

for entity in scene.sim.rigid_solver.entities[2:]:
    for link in entity.links:
        link._inertial_mass = 0.015

tool.set_dofs_kp(np.array([250, 250, 250, 100, 100, 100]) * 2)
tool.set_dofs_kv(np.array([70, 70, 70, 30, 30, 30]) * 2)

img_cnt = 0

def save_img():
    global img_cnt
    img_file = os.path.join(img_dir, f'{img_cnt:04d}.png')
    img_cnt += 1
    img = cam.render()[0][0]
    print(img.shape)
    imageio.imwrite(img_file, img) 
    print('saved image: ', img_file) 
    

def control_tool(tool, pos, rot=None):
    if rot is None:
        rot = tool.get_dofs_position()[3:]
    q_tool = np.concatenate((pos, rot))
    tool.control_dofs_position(q_tool)
    for _ in range(200):
        scene.step()
        if _ % 20 == 0:
            save_img()

def evaluate(x, img_dir):
    x = np.array(x).reshape((5, 3))
    scene.reset()

    tool_quats = x[:1]
    tool_poses = x[1:]

    control_tool(tool, tool_poses[0], tool_quats[0])
    control_tool(tool, tool_poses[1], tool_quats[0])
    control_tool(tool, tool_poses[2], tool_quats[0])
    control_tool(tool, tool_poses[3], tool_quats[0])
    
    dough_particles = dough.get_particles()
    penalty = dough_particles[:, 2].max()

    return penalty

score = evaluate(best_traj, img_dir)
print("score: ", score)