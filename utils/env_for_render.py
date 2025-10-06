import genesis as gs
import imageio
import torch
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from genesis.engine.entities import MPMEntity
import os 
import cma
import json
import matplotlib.pyplot as plt

def euler_to_quat(euler):
    cy = np.cos(euler[2] * 0.5)
    sy = np.sin(euler[2] * 0.5)
    cp = np.cos(euler[1] * 0.5)
    sp = np.sin(euler[1] * 0.5)
    cr = np.cos(euler[0] * 0.5)
    sr = np.sin(euler[0] * 0.5)

    q = np.zeros(4)
    q[0] = cy * cp * cr + sy * sp * sr
    q[1] = cy * cp * sr - sy * sp * cr
    q[2] = sy * cp * sr + cy * sp * cr
    q[3] = sy * cp * cr - cy * sp * sr
    return q

def quat_to_euler(quat):
    x, y, z, w = quat
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    Y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = np.arctan2(t3, t4)

    return np.array([X, Y, Z])

class RenderEnv():
    project_path = os.path.dirname(os.path.join(os.path.abspath(__file__), '..'))
    def __init__(self, task):
        self.task = task
        gs.init()
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=2e-4, substeps=1, gravity=(0,0,-9.8),
            ),
            mpm_options=gs.options.MPMOptions(
                lower_bound=(-0.5,-0.5,-0.1), upper_bound=(1.5,1.5,1),
                grid_density=256, enable_CPIC=True, 
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
            renderer=gs.renderers.RayTracer(
                env_radius=200.0,
                env_surface=gs.surfaces.Emission(
                    emissive_texture=gs.textures.ImageTexture(
                        image_path=f"{self.project_path}/assets/hdr.hdr",
                        image_color=(0.5, 0.5, 0.5),
                    )
                ),
                lights=[
                    {'pos': (0, -70, 40), 'color': (255.0, 255.0, 255.0), 'radius': 7, 'intensity': 0.3 * 1.4},
                ]
            ),
            show_viewer = False,
        )
        mat_rigid = gs.materials.Rigid(coup_friction=5.0)#
        self.desk_height = 0.8
        table = self.scene.add_entity(
            material=mat_rigid,
            morph=gs.morphs.Mesh(
                file=f"{self.project_path}/assets/wooden_table_02_4k.glb",
                scale=(1.2, 1, 1.5),
                pos=(0,0.58,0),
                euler=(90, 0, 90),
                fixed=True,
                collision=True,
                convexify=False,
                decompose_nonconvex=False,
                coacd_options=gs.options.CoacdOptions(threshold=0.01, preprocess_resolution=60, max_convex_hull=20),
                decimate_face_num=2000,
            ),
            surface=gs.surfaces.Plastic(
                # vis_mode='collision',
            ),
        )
        wall = self.scene.add_entity(
            # material=mat_rigid,
            morph=gs.morphs.Box(
                lower=(-0.9, -1.0, 0),
                upper=(-0.8, 2.5, 2.0),
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                color=(1.0, 0.99, 0.93),
            ),
        )
        # plane = self.scene.add_entity(
        #     morph=gs.morphs.Plane(),
        # )
        floor = self.scene.add_entity(
            # material=mat_rigid,
            morph=gs.morphs.Mesh(
                file=f"{self.project_path}/assets/carpet.obj",
                scale=0.03,
                pos=(0,0,0),
                euler=(0, 0, 0),
                fixed=True,
                collision=False,
            )
        )
        self.xarm = self.scene.add_entity(
            gs.morphs.URDF(
                file='{}/assets/xarm7_with_gripper_reduced_dof.urdf'.format(self.project_path), 
                pos=(0, 0, self.desk_height),
                euler=(0, 0, 90),
                fixed=True, 
                collision=False, 
                links_to_keep=["link_tcp"]),
        )
        self.cam_gallery = None
        self.cam_trajectory = None
        self.n_traj_imgs = 0
        self.create_log_dir()
        self.add_entities()
        self.init_mass()

    def create_log_dir(self):
        log_dir = os.path.join(self.project_path, self.task, 'try')
        os.makedirs(log_dir, exist_ok=True)
        self.img_save_dir = log_dir
        
    def init_mass(self, mass=0.015):
        for entity in self.scene.sim.rigid_solver.entities[2:]:
            for link in entity.links:
                link._inertial_mass = mass
    
    def add_camera_for_gallery(self):
        raise NotImplementedError()
        
    def add_camera_for_trajectory(self):
        raise NotImplementedError()

    def add_entities(self):
        raise NotImplementedError()

    def save_gallery_img(self):
        img,_,_,_ = self.cam_gallery.render()
        file_name = os.path.join(self.img_save_dir, f"gallery.png")
        imageio.imwrite(file_name, img)

    def save_trajectory_img(self):
        img,_,_,_ = self.cam_trajectory.render()
        file_name = os.path.join(self.img_save_dir, f"traj_{self.n_traj_imgs:04d}.png")
        imageio.imwrite(file_name, img)
        self.n_traj_imgs += 1

    def save_gif(self):
        save_dir = self.img_save_dir
        images = []
        file_list = [f for f in os.listdir(save_dir) if f.endswith('.png')]
        file_list.sort()
        for f in file_list:
            images.append(imageio.imread(os.path.join(save_dir, f)))
        imageio.mimsave(os.path.join(save_dir, 'movie.gif'), images)