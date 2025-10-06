import numpy as np
import genesis as gs
from scipy.spatial.transform import Rotation as R

from step0_scene import HolderEnv

class HolderEnvTool(HolderEnv):
    def __init__(self, task='task04_holder'):
        super().__init__(task)

    def add_tools_for_task(self):
        
        pass
    
    def reset_position(self):

        q_franka = np.array([ 4.2366, -0.8135, -4.1206,  1.4427, -2.3837, -1.0637,  1.4335,  0.0122,
         0.0124])
        v_franka = np.array([-4.6063e-04, -6.9945e-05,  5.6052e-04, -1.2440e-04,  2.1296e-04,
            1.2321e-04, -1.2930e-04,  1.4475e-04, -3.2250e-05])
        
        self.xarm.set_dofs_position(q_franka)
        self.xarm.set_dofs_velocity(v_franka)
        self.phone.set_dofs_position([ 0.3990,  0.2010,  0.0800,  1.5795, -0.0028, -0.0356])
        self.scene.step()
        q_franka[-2:] = 0.044
        self.xarm.control_dofs_position(q_franka)

        for i in range(50):
            self.scene.step()

    def evaluate(self, trajs, close_gripper=[]):
        if trajs.ndim == 2:
            trajs = trajs.reshape(1, trajs.shape[0], trajs.shape[1])
        
        if trajs.shape[2] != 3:
            trajs = trajs.reshape(trajs.shape[0], -1, 3)
        
        single_traj = (trajs.shape[0] == 1)
        n_step = trajs.shape[1]
        self.reset()
        self.reset_position()

        close_size = 0.044
        ee_name = "link_tcp"
        ee = self.xarm.get_link("link_tcp")
        ee_pos = ee.get_pos().cpu().numpy()
        ee_quat = ee.get_quat().cpu().numpy()
        
        n_envs, n_wpts = trajs.shape[0], trajs.shape[1]
        for i in range(n_wpts - 2):
            ik_pos = trajs[:, i, :3]
            lift_pos = ik_pos + ee_pos
            quat = self.xarm.get_link(ee_name).get_quat().cpu()
            if single_traj:
                lift_pos = lift_pos[0]
            q, err = self.xarm.inverse_kinematics(
                link=self.xarm.get_link(ee_name),
                pos=lift_pos,
                quat=quat,
                return_error=True,
                respect_joint_limit=False,
            )
            if single_traj:
                q[-2:] = close_size
            else:
                q[:,-2:] = close_size
            self.xarm.control_dofs_position(q)
            for _ in range(60):
                self.scene.step()
                if _ % 10 == 0:
                    self.save_img(log=f"step_{i}_{_}",iter=self.iter)
        
        def quaternion_multiply(q1, q2):
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            return np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 + y1*w2 + z1*x2 - x1*z2,
                w1*z2 + z1*w2 + x1*y2 - y1*x2
            ])
        
        ik_pos = trajs[:, -2, :3]
        lift_pos = ik_pos + ee_pos
        euler = trajs[:, -1, :3]
        euler = np.deg2rad(euler)
        b_theta_x, b_theta_y, b_theta_z = euler[:, 0], euler[:, 1], euler[:, 2]
        ik_quat = ee.get_quat().cpu().numpy()
        if single_traj:
            ik_quat = ik_quat.reshape(1, -1)
        #breakpoint()
        for i in range(n_envs):
            ik_quat[i] = quaternion_multiply(ik_quat[i], R.from_euler('xyz', [b_theta_x[i], b_theta_y[i], b_theta_z[i]]).as_quat())
        
        if single_traj:
            ik_quat = ik_quat[0]
            lift_pos = lift_pos[0]
        q, err = self.xarm.inverse_kinematics(
            link=self.xarm.get_link(ee_name),
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
        for _ in range(60):
            self.scene.step()
            if _ % 30 == 0:
                self.save_img(log = f"step_{n_wpts - 1}_rotation_{_}", iter=self.iter)
        
        cur_q = self.xarm.get_dofs_position()
        if single_traj:
            cur_q[-2:] = 0.00
        else:
            cur_q[:,-2:] = 0.00
        self.xarm.control_dofs_position(cur_q)
        for _ in range(50):
            self.scene.step()
            if _ % 25 == 0:
                self.save_img(f"step_{n_wpts}_release_{_}", iter=self.iter)
        
        for i in range(300):
            
            self.scene.step()
            if i % 100 == 0:
                self.save_img(f"step_{n_wpts + 1}_{i}",iter=self.iter)
        
        L2_loss = self.metric()
        return L2_loss


env = HolderEnvTool()
dim, rng, sigma, n_envs, iters = 18, 0.2, 0.5, 20, 150 # range:0.06 sigma: 0.03
initial_params = [(0,0,0.1),(0,0,0.2),(-0.1, 0.2, 0.2),(-0.15, 0.25, 0.2),(-0.15, 0.25, 0.2),(1.5 * np.pi, 0, 0)]
initial_params = np.array(initial_params).flatten()
option = "training" # "evaluation"
if option == "training":
    rng = [rng] * dim
    rng[-3:] = [np.pi, np.pi, np.pi]
    lower_bounds = initial_params - rng
    upper_bounds = initial_params + rng
    env.set_cmaes_params(dim, initial_params, range=rng, sigma=sigma, n_envs=n_envs, iters=iters, _lower_bounds=lower_bounds, _upper_bounds=upper_bounds)
    env.save_img(iter=0)  # Save an initial image for reference
    env.optimize()
elif option == "evaluation":
    env.build_scene_for_evaluation()
    trajs = np.array([initial_params])
    trajs = trajs.reshape(-1, 3)
    L2_loss = env.evaluate(trajs=trajs, close_gripper=[0, 1, 2, 3, 4, 5, 6, 7])
    print(f"Evaluation L2 loss: {L2_loss}")
    env.save_img(iter=0)  # Save an initial image for reference
else:
    raise ValueError("Invalid option. Choose 'training' or 'evaluation'.")