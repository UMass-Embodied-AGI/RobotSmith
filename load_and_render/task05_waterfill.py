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
        particle_size=0.004, lower_bound=(-2,-2,0), upper_bound=(2,2,2)
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
bottle = scene.add_entity(
    material=mat_rigid,
    morph=gs.morphs.Mesh(
        file="{}/task05_waterfill/bottle.obj".format(project_path),
        scale=0.0015,
        pos=(0.0143, 0.5, 0.156),
        euler=(90, 0, 90),
        fixed=True,
        collision=True,
        decompose_nonconvex=False,
        convexify=False,
        decimate=False,
    ),
    surface=gs.surfaces.Glass(
        color    = (1.0, 1.0, 1.0, 0.9),
    ),  
) # bottle
cup = scene.add_entity(
    material=mat_rigid,
    morph=gs.morphs.Mesh(
        file="{}/task05_waterfill/simplify_red_cup.obj".format(project_path),
        scale=1.0,
        pos=(0, 0.3, 0.064),
        euler=(90, 0, 0),
        fixed=False,
        collision=True,
        decompose_nonconvex=False,
        convexify=False,
        decimate=False,
    ),
    surface=gs.surfaces.Default(
        color    = (0.8, 0.0, 0.0),
    ),  
) # cup
water = scene.add_entity(
    morph=gs.morphs.Nowhere(n_particles=9600),
    material=gs.materials.SPH.Liquid(mu=0.01, sampler="regular"),
    surface=gs.surfaces.Glass(
        color=(0.5, 0.7, 0.9, 1.0),
        vis_mode='particle',
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

water_pos = np.load('{}/task05_waterfill/water.npy'.format(project_path))
water.set_pos(0, water_pos)
water._solver._kernel_set_particles_active(
    water._sim.cur_substep_local,
    0, 9600, 
    gs.ACTIVE,
) # load_water
