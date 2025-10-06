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
) # dough
cam = scene.add_camera(
    pos=(0.4,-0.9,0.9), lookat=(0.4,0.0,0.0), fov=50, res=(1440,1440), GUI=False,
) # cam

scene.build()

def save_img(cam, nam):
    img = cam.render()[0][0]
    imageio.imwrite(nam, img)
    return img
