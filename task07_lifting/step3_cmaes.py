import numpy as np
import genesis as gs
from scipy.spatial.transform import Rotation as R

from step0_scene import LiftingEnv

class LiftingEnvTool(LiftingEnv):
    def __init__(self, task='task07_lifting'):
        super().__init__(task)

    def evaluate(self, trajs, close_gripper=[]):
        #close_gripper = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        if trajs.ndim == 2:
            trajs = trajs.reshape(1, trajs.shape[0], trajs.shape[1])
        
        if trajs.shape[2] != 3:
            trajs = trajs.reshape(trajs.shape[0], -1, 3)
        single_traj = (trajs.shape[0] == 1)
        n_step = trajs.shape[1]
        self.reset()
        for i in range(n_step):
            ik_pos = trajs[:, i, :3]
            ik_quat = [ 0,0.7071,-0.7071,0]#self.xarm.get_link('link_tcp').get_quat().cpu()
            ik_quat = np.array(ik_quat)
            ik_quat = np.tile(ik_quat, (ik_pos.shape[0], 1))
            if single_traj:
                ik_pos = ik_pos[0]
                ik_quat = ik_quat[0]
            #breakpoint()
            q, err = self.xarm.inverse_kinematics(
                link=self.xarm.get_link('link_tcp'),
                pos=ik_pos, 
                quat=ik_quat, 
                return_error=True,
                respect_joint_limit=False,
            )
            if single_traj:
                q[-2:] = 0.044 if i != 0 else 0  # Close the gripper if in close_gripper list
            else:
                q[:,-2:] = 0.044 if i != 0 else 0  # Close the gripper if in close_gripper list
            self.xarm.control_dofs_position(q)
            for _ in range(60):
                self.scene.step()
                if _ % 20 == 0: 
                    self.save_img(iter=self.iter)
        
        for i in range(300):
            self.scene.step()
            if i % 100 == 0:
                self.save_img(f"final",iter=self.iter)

        L2 = - self.metric()
        print("L2 distance: ", L2)
        return L2

if __name__ == "__main__":
    option = "training" # "training"

    if option == "evaluation":
        env = LiftingEnvTool()
        env.build_scene_for_evaluation()
        env.reset()
        env.save_img(iter = 0)
        #breakpoint()
        trajs = [(0.22,0.2,0.005),(0.2,0.2,0.005), (0.2,0.2,0.1), (0.2,0.2,0.2), (0.3,0.2,0.2), (0.4,0.2,0.2), (0.475,0.2, 0.2), (0.465,0.2,0.1), (0.455,0.2,0.01),(0.455,0.03,0.01),(0.455,0.03,0.1), (0.455,0.03,0.2)]
        trajs = np.array(trajs).flatten()
        trajs = trajs.reshape(1, -1, 3)
        
        env.evaluate(trajs)
    
    elif option == "training":
        dim, rng, sigma, n_envs, iters = 36, 0.06, 0.02, 20, 150 # range:0.06 sigma: 0.03
        initial_params = [(0.22,0.2,0.005),(0.2,0.2,0.005), (0.2,0.2,0.1), (0.2,0.2,0.2), (0.3,0.2,0.2), (0.4,0.2,0.2), (0.475,0.2, 0.2), (0.465,0.2,0.1), (0.455,0.2,0.01),(0.455,0.03,0.01),(0.455,0.03,0.1), (0.455,0.03,0.2)]
        initial_params = np.array(initial_params).flatten()

        rng = [rng] * dim
        rng[:12] = [1e-5] * 12

        lower_bounds = initial_params - rng
        upper_bounds = initial_params + rng
        env = LiftingEnvTool()
        env.set_cmaes_params(dim, initial_params, range=rng, sigma=sigma, n_envs=n_envs, iters=iters, _lower_bounds=lower_bounds, _upper_bounds=upper_bounds)
        env.save_img(iter=0)  # Save an initial image for reference
        env.optimize()
    else:
        raise ValueError("Invalid option. Choose 'training' or 'evaluation'.")

# Lifting, holder optimization
# Joint optimization
# New Task design

# Sim video, metric, CMA-ES curve picture

# Paper writing: Method, Experiment