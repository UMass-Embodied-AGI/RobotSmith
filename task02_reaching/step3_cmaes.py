import numpy as np
import genesis as gs

from step0_scene import ReachingEnv

class ReachingEnvTool(ReachingEnv):
    def __init__(self, task='task02_reaching'):
        super().__init__(task)

    def add_tools_for_task(self):
        self.tool = self.scene.add_entity(
            material=gs.materials.Rigid(friction = 5.0),
            morph = gs.morphs.Mesh(
                file="history/reach_extension_tool.obj",
                scale=1.0,
                pos=(0.2, 0.2, 0.2),
                euler=(90, 0, 90),
                fixed=False,
                collision=True,
                decompose_nonconvex=False,
                convexify=False,
                decimate=False,
            ),
            surface=gs.surfaces.Default(
                color    = (0.8, 0.0, 0.0),
            ),  
        )

    def evaluate(self, trajs, close_gripper=[]):
        single_traj = (trajs.shape[0] == 1)
        n_step = trajs.shape[1]
        self.reset()
        for i in range(n_step):
            ik_pos = trajs[:, i, :3]
            ik_quat = self.xarm.get_link('link_tcp').get_quat().cpu()
            q, err = self.xarm.inverse_kinematics(
                link=self.xarm.get_link('link_tcp'),
                pos=ik_pos, 
                quat=ik_quat, 
                return_error=True
            )
            q[:,-2:] = 0.0 if i in close_gripper else q[:,-2:]  # Close the gripper if in close_gripper list
            self.xarm.control_dofs_position(q)
            for _ in range(60):
                self.scene.step()
                if _ % 20 == 0:
                    self.save_img(iter=self.iter)
        for _ in range(600):
            self.scene.step()
            if _ % 100 == 0:
                self.save_img(iter=self.iter)
        L2 = self.metric()
        return L2
    
env = ReachingEnvTool()
dim, rng, sigma, n_envs, iters = 30, 0.03, 0.02, 15, 50
initial_params = [(0.2,0.2,0.2),(0.2,0.2,0.3),
                    (0.4,0.2,0.3),(0.6,0.2,0.3),
                    (0.6,0.3,0.3),(0.6,0.4,0.3),
                    (0.5,0.4,0.15),
                    (0.4,0.3,0.15),(0.3,0.2,0.15),
                    (0.1,0.0,0.15)
                    ]
initial_params = np.array(initial_params).flatten()
env.set_cmaes_params(dim, initial_params, range=rng, sigma=sigma, n_envs=n_envs, iters=iters)
env.save_img(iter=0)  # Save an initial image for reference
env.optimize()
#env.build_scene_for_evaluation()
#env.save_img(iter=0)  # Save an initial image for reference