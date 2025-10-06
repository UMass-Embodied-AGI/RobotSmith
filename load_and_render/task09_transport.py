import multiprocessing

import numpy as np
import genesis as gs
import time
import cma
import os
from itertools import accumulate
import imageio
import matplotlib.pyplot as plt
from genesis.constants import backend as gs_backend

project_path = os.path.dirname(os.path.join(os.path.abspath(__file__), '..'))

gs.init(seed=0, precision='32', logging_level='error', backend=gs_backend.gpu)
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=2e-3, substeps=10, gravity=(0,0,-9.8),
    ),
    sph_options=gs.options.SPHOptions(
        particle_size=0.008, lower_bound=(-2,-2,0), upper_bound=(2,2,2)
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_fov=30, res=(960, 640),
    ),
    vis_options=gs.options.VisOptions(
        env_separate_rigid = True
    ),
    show_viewer = False,
) # scene
plane = scene.add_entity(
    morph=gs.morphs.Plane(),
) # plane
xarm = scene.add_entity(
    gs.morphs.URDF(
        file='{}/assets/xarm7_with_gripper_reduced_dof.urdf'.format(project_path), 
        fixed=True, 
        collision=True, 
        links_to_keep=["link_tcp"]),
) # xarm
mat_rigid = gs.materials.Rigid(
    coup_friction=0.1,
    coup_softness=0.001,
    coup_restitution=0.001,
    sdf_cell_size=0.0001,
    sdf_min_res=64,
    sdf_max_res=256
) # mat_rigid
cup = scene.add_entity(
    material=mat_rigid,
    morph = gs.morphs.Mesh(
        file="{}/task09_transport/simplify_red_cup.obj".format(project_path),
        scale=1.0,
        pos=(0.3, -0.2, 0.064),
        euler=(90, 0, 0),
        fixed=False,
        collision=True,
        decompose_nonconvex=False,
        convexify=False,
        decimate=False,
    ),
    surface=gs.surfaces.Default(
        color    = (0.0, 0.0, 0.8),
    ),  
) # cup
tank = scene.add_entity(
    material=mat_rigid,
    morph = gs.morphs.Mesh(
        file='{}/task09_transport/tank.obj'.format(project_path),
        scale=0.3,
        pos=(0.4, 0.2, 0.07),
        euler=(90, 0, 0),
        fixed=True,
        collision=True,
        decompose_nonconvex=False,
        convexify=False,
        decimate=False,
    ),
    surface=gs.surfaces.Default(
        color    = (0.8, 0.8, 0.8, 0.5),
    ),  
) # cup
eps=0.03
water = scene.add_entity(
    material=gs.materials.SPH.Liquid(mu=0.01, sampler="regular"),
    morph = gs.morphs.Box(
        lower=(0.26+eps, 0.09+eps, 0.01+eps),
        upper=(0.54-eps, 0.31-eps, 0.13-eps),
    ),
    surface=gs.surfaces.Default(
        color=(0.5, 0.7, 0.9, 0.7),
    ),  
) # water
cam = scene.add_camera(
    pos=(0.4,-0.9,0.9), lookat=(0.4,0.0,0.0), fov=50, res=(1440,1440), GUI=False,
) # cam

scene.build()

def save_img(cam, nam):
    img = cam.render()[0][0]
    imageio.imwrite(nam, img)
    return img

# for _ in range(300):
#     scene.step()