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


class CMAESOptimizer():
    project_path = os.path.dirname(os.path.join(os.path.abspath(__file__), '..'))
    def __init__(self, task, scene=None, GUI=False, log_dir=None):
        self.task = task
        self.GUI = GUI
        gs.init()
        if scene is None:
            self.scene = gs.Scene(
                sim_options=gs.options.SimOptions(
                    dt=0.01, substeps=10, gravity=(0,0,-9.8),
                ),
                mpm_options=gs.options.MPMOptions(
                    lower_bound=(-0.1,-0.1,-0.1), upper_bound=(0.6,0.6,0.6),
                    grid_density=128, #enable_CPIC=True, 
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
                show_viewer = self.GUI,
            )
        else:
            self.scene = scene
        self.plane = self.scene.add_entity(
            morph=gs.morphs.Plane(),
        )
        self.xarm = self.scene.add_entity(
            gs.morphs.URDF(
                file='{}/assets/xarm7_with_gripper_reduced_dof.urdf'.format(self.project_path), 
                fixed=True, 
                collision=True, 
                links_to_keep=["link_tcp"]),
        )
        self.cam = self.scene.add_camera(
            pos=(1.5,0.0,1.5), lookat=(0.0,0.0,0.5), fov=50, res=(1440,1440), GUI=False,
        )
        self.cam_up = self.scene.add_camera(
            pos=(0.5,0.0,1.5), lookat=(0.5,0.0,0.5), fov=50, res=(1440,1440), GUI=False,
        )
        self.cam_right = self.scene.add_camera(
            pos=(0.5,0.5,0.02), lookat=(0.5,0.0,0.02), fov=50, res=(1440,1440), GUI=False,
        )
        self.cam_front = self.scene.add_camera(
            pos=(1.0,0.0,0.02), lookat=(0.5,0.0,0.02), fov=50, res=(1440,1440), GUI=False,
        )

        self.scene_built_for_training = False
        self.scene_built_for_evaluation = False
        self.img_save_dir = None
        self.img_steps = 0

        self.cmaes_optimizer_created = False
        self.iter = 0

        if log_dir is None:
            log_dir = os.path.join(self.project_path, self.task)
        else:
            log_dir = os.path.join(self.project_path, log_dir)

        self.create_log_dir(log_dir)

        self.add_entities_for_task()

        self.add_tools_for_task()

        self.init_mass()

    def set_cmaes_params(self, dim_params, init_params, range, sigma=0.02, n_envs=15, iters=50):
        if self.cmaes_optimizer_created:
            raise ValueError("CMA-ES optimizer has already been created. You cannot set parameters again.")
        self.n_envs = n_envs
        self.dim_params = dim_params
        lower_bounds = init_params - range
        upper_bounds = init_params + range
        self.es = cma.CMAEvolutionStrategy(
            init_params,
            sigma,
            {
                "maxiter": iters,
                "popsize": n_envs,
                "bounds": [lower_bounds, upper_bounds],
            },
        )
        self.cmaes_optimizer_created = True
        # self.build_scene_for_training()
        
    def create_log_dir(self, log_dir):
        log_dir = os.path.join(log_dir, 'try')
        os.makedirs(log_dir, exist_ok=True)
        n_tries = len([fil for fil in os.listdir(log_dir) if not '.' in fil])
        self.img_save_dir = os.path.join(log_dir, f"{n_tries:03d}")
        os.makedirs(self.img_save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.img_save_dir, "opt_log"), exist_ok=True)
        
    def init_mass(self, mass=0.015):
        for entity in self.scene.sim.rigid_solver.entities[2:]:
            for link in entity.links:
                link._inertial_mass = mass
    
    def add_entities_for_task(self):
        raise NotImplementedError()
        
    def add_tools_for_task(self):
        raise NotImplementedError()

    def metric(self):
        raise NotImplementedError()

    def evaluate(self, trajs):
        raise NotImplementedError()

    def build_scene_for_training(self):
        if not self.cmaes_optimizer_created:
            raise ValueError("CMA-ES optimizer must be set before building the scene for training.")
        if self.scene_built_for_evaluation:
            raise ValueError(f"Scene is already built for evaluation, n_envs=1.")
        if not self.scene_built_for_training:
            self.scene.build(self.n_envs)
            self.scene_built_for_training = True

    def build_scene_for_evaluation(self):
        if self.scene_built_for_training:
            raise ValueError(f"Scene is already built for training, n_envs={self.n_envs}.")
        if not self.scene_built_for_evaluation:
            self.scene.build()
            self.n_envs = 1
            self.scene_built_for_evaluation = True

    def reset_xarm(self):
        self.xarm.set_dofs_kp(
            np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 300, 300]),
        )
        self.xarm.set_dofs_kv(
            np.array([450, 450, 350, 350, 200, 200, 200, 30, 30]),
        )
        self.xarm.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -300, -300]),
            np.array([87, 87, 87, 87, 12, 12, 12, 300, 300]),
        )
        q_init = np.array([0, -0.44, 0, 0.96, 0, 1.4, 0, 0, 0])
        self.xarm.set_dofs_position(q_init)

    def save_img(self, log='', iter=None):
        als,_,_,_ = self.cam.render()
        ups,_,_,_ = self.cam_up.render()
        frs,_,_,_ = self.cam_front.render()
        ris,_,_,_ = self.cam_right.render()
        file_dir = self.img_save_dir if iter is None else os.path.join(self.img_save_dir, f"iter{iter:03d}")
        os.makedirs(file_dir, exist_ok=True)
        if als.shape[0] != self.n_envs:
            als, ups, frs, ris = np.expand_dims(als, axis=0), np.expand_dims(ups, axis=0), np.expand_dims(frs, axis=0), np.expand_dims(ris, axis=0)
        n = als.shape[0]
        for i in range(n):
            al, up, fr, ri = als[i], ups[i], frs[i], ris[i]
            comb = np.concatenate([np.concatenate((al, up), axis=1), np.concatenate((fr, ri), axis=1)], axis=0)
            os.makedirs(os.path.join(file_dir, f"{i:03d}"), exist_ok=True)
            file_name = os.path.join(file_dir, f"{i:03d}", f"{self.img_steps:004d}.png" if log == '' else f"{self.img_steps:004d}_{log}.png")
            print(comb.shape, file_name)
            imageio.imwrite(file_name, comb)
        self.img_steps += 1

    def save_gif(self, save_dir):
        images = []
        file_list = [f for f in os.listdir(save_dir) if f.endswith('.png')]
        file_list.sort()
        for f in file_list:
            images.append(imageio.imread(os.path.join(save_dir, f)))
        imageio.mimsave(os.path.join(save_dir, 'movie.gif'), images)
    
    def reset(self):
        self.scene.reset()
        self.reset_xarm()
        self.init_mass()

    def optimizer_step(self):
        self.iter += 1
        solutions = np.array(self.es.ask())
        np.save(os.path.join(self.img_save_dir, "opt_log", f"solutions_iter_{self.iter}.npy"), solutions)
        rewards = self.evaluate(solutions.reshape((self.n_envs, -1, self.dim_params)))
        self.es.tell(solutions, rewards)
        fbest = np.array(self.es.result.fbest) # score
        xbest = np.array(self.es.result.xbest) # traj
        log_file = os.path.join(self.img_save_dir, "opt_log", f"opt_iter{self.iter:03d}.npz")
        np.savez(log_file, solutions, rewards, fbest, xbest)
        return fbest, xbest

    def optimize(self):
        iters, rewds = [], []
        while len(iters) < 50 or not self.es.stop():
            fbest, xbest = self.optimizer_step()
            iters.append(self.iter)
            rewds.append(fbest)

            np.save(os.path.join(self.img_save_dir, "best_traj.npy"), xbest)

            plt.plot(iters, rewds)
            plt.xlabel('Iteration')
            plt.ylabel('Reward')
            plt.title('CMA-ES Opt')
            plt.savefig(os.path.join(self.img_save_dir, "cmaes.png"))
