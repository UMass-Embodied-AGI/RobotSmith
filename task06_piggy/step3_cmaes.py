import numpy as np
import genesis as gs
from scipy.spatial.transform import Rotation as R

from step0_scene import PiggyEnv

class PiggyEnvTool(PiggyEnv):
    def __init__(self, task='task06_piggy'):
        super().__init__(task)
    

    def evaluate(self, trajs, close_gripper=[]):
        if trajs.ndim == 2:
            trajs = trajs.reshape(1, trajs.shape[0], trajs.shape[1])
        
        if trajs.shape[2] != 3:
            trajs = trajs.reshape(trajs.shape[0], -1, 3)
        
        def quaternion_multiply(q1, q2):
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            return np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 + y1*w2 + z1*x2 - x1*z2,
                w1*z2 + z1*w2 + x1*y2 - y1*x2
            ])
        single_traj = (trajs.shape[0] == 1)
        n_envs, n_step = trajs.shape[0], trajs.shape[1]
        self.reset()
        close_size = 0.044
        for i in range(n_step - 9):
            ik_pos = trajs[:, i, :3]
            #ik_quat = self.xarm.get_link('link_tcp').get_quat().cpu()
            ik_quat = [ 0,0.7071,-0.7071,0]
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
                q[-2:] = 0.044 if i >= 2 else 0  # Close the gripper if in close_gripper list
            else:
                q[:,-2:] = 0.044 if i >= 2 else 0  # Close the gripper if in close_gripper list
            self.xarm.control_dofs_position(q)
            for _ in range(60):
                self.scene.step()
                if _ % 20 == 0: 
                    self.save_img(f"step{i}",iter=self.iter)
        

        def rotation(lift_pos, euler):
            b_theta_x, b_theta_y, b_theta_z = euler[:, 0], euler[:, 1], euler[:, 2]
            ik_quat = self.xarm.get_link('link_tcp').get_quat().cpu().numpy()
            if single_traj:
                ik_quat = ik_quat.reshape(1, -1)
            for i in range(n_envs):
                quat = (np.cos(b_theta_x[i]/2), np.sin(b_theta_x[i]/2),0, 0)
                quat_y = (np.cos(b_theta_y[i]/2), 0, np.sin(b_theta_y[i]/2),0)
                quat_z = (np.cos(b_theta_z[i]/2), 0, 0, np.sin(b_theta_z[i]/2))
                ik_quat[i] = quaternion_multiply(ik_quat[i], quat)
                ik_quat[i] = quaternion_multiply(ik_quat[i], quat_y)
                ik_quat[i] = quaternion_multiply(ik_quat[i], quat_z)

            if single_traj:
                ik_quat = ik_quat[0]
                lift_pos = lift_pos[0]
            q, err = self.xarm.inverse_kinematics(
                link=self.xarm.get_link('link_tcp'),
                pos=lift_pos,
                quat=ik_quat,
                return_error=True,
                respect_joint_limit=False,
            )
            if single_traj:
                q[-2:] = close_size
            else:
                q[:,-2:] = close_size
            self.xarm.control_dofs_position(q)
            for _ in range(150):
                self.scene.step()
                if _ % 30 == 0:
                    self.save_img(log = f"step_{n_step - 1}_rotation_{_}", iter=self.iter)
        
        rotation(trajs[:, -9, :3], trajs[:, -8, :3])
        rotation(trajs[:, -7, :3], trajs[:, -6, :3])

        for i in range(n_step - 5, n_step - 3):
            ik_pos = trajs[:, i, :3]
            ik_quat = self.xarm.get_link('link_tcp').get_quat().cpu()
            #ik_quat = #[ 0,0.7071,-0.7071,0]
            #ik_quat = np.array(ik_quat)
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
                q[-2:] = 0.044 if i >= 2 else 0  # Close the gripper if in close_gripper list
            else:
                q[:,-2:] = 0.044 if i >= 2 else 0  # Close the gripper if in close_gripper list
            self.xarm.control_dofs_position(q)
            for _ in range(60):
                self.scene.step()
                if _ % 2 == 0: 
                    self.save_img(f"step{i}",iter=self.iter)

        rotation(trajs[:, -3, :3], trajs[:, -2, :3])

        for i in range(n_step - 1, n_step):
            ik_pos = trajs[:, i, :3]
            ik_quat = self.xarm.get_link('link_tcp').get_quat().cpu()
            #ik_quat = #[ 0,0.7071,-0.7071,0]
            #ik_quat = np.array(ik_quat)
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
                q[-2:] = 0.044 if i >= 2 else 0  # Close the gripper if in close_gripper list
            else:
                q[:,-2:] = 0.044 if i >= 2 else 0  # Close the gripper if in close_gripper list
            self.xarm.control_dofs_position(q)
            for _ in range(60):
                self.scene.step()
                if _ % 20 == 0: 
                    self.save_img(f"step{i}",iter=self.iter)


        for i in range(300):
            
            self.scene.step()
            if i % 100 == 0:
                self.save_img(f"step_{n_step + 1}_{i}",iter=self.iter)
        
        L2_loss = self.metric()
        return L2_loss

env = PiggyEnvTool()
dim, rng, sigma, n_envs, iters = 24, 0.07, 0.1, 20, 150 # range:0.06 sigma: 0.03
initial_params = [(0.40,0.15,0.1),(0.40,0.15,0.003),(0.40,0.15,0.003),(0.40, 0.15, 0.15),(0.3, 0.2, 0.3),(0.2,0.2,0.37), (0.18,0.205,0.37),(0, 1.57, 0), (0.18,0.205,0.37),(-1.57, 0, 0), (0.18,0.205,0.27),(0.18,0.205,0.2), (0.18,0.205,0.2),(1.57, 0, 0),(0.18,0.205,0.37)]
initial_params = np.array(initial_params).flatten()
option = "evaluation" # "evaluation"
if option == "training":
    rng = [rng] * dim
    rng[-3:] = [np.pi, np.pi, np.pi]
    rng[:12] = [1e-5] * 12
    lower_bounds = initial_params - rng
    upper_bounds = initial_params + rng
    env.set_cmaes_params(dim, initial_params, range=rng, sigma=sigma, n_envs=n_envs, iters=iters, _lower_bounds=lower_bounds, _upper_bounds=upper_bounds)
    env.save_img(iter=0)  # Save an initial image for reference
    env.optimize()
elif option == "evaluation":
    env.build_scene_for_evaluation()
    trajs = np.array([initial_params])
    trajs = trajs.reshape(-1, 3)
    L2_loss = env.evaluate(trajs=trajs, close_gripper=[1, 2, 3, 4, 5, 6, 7])
    print(f"Evaluation L2 loss: {L2_loss}")
    env.save_img(iter=0)  # Save an initial image for reference
    env.save_video()
else:
    raise ValueError("Invalid option. Choose 'training' or 'evaluation'.")
