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
bowl = scene.add_entity(
    morph=gs.morphs.Mesh(
        file="{}/task07_lifting/simplify_simplify_bowl.obj".format(project_path),
        scale=0.09,
        pos=(0.35, 0.3, 0.05),
        euler=(90, 0, 90),
        collision=True,
        decompose_nonconvex=False,
        convexify=False,
        decimate=False,
    ),
    surface=gs.surfaces.Default(
        color    = (0.8, 0.8, 0.8),
    ),  
) # bowl
cam = scene.add_camera(
    pos=(0.4,-0.9,0.9), lookat=(0.4,0.0,0.0), fov=50, res=(1440,1440), GUI=False,
) # cam

scene.build()

def save_img(cam, nam):
    img = cam.render()[0][0]
    imageio.imwrite(nam, img)
    return img
