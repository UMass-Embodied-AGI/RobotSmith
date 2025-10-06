import numpy as np
import genesis as gs
from utils.env_for_render import RenderEnv, euler_to_quat, quat_to_euler

class CutREnv(RenderEnv):
    def __init__(self, task='task08_cutting'):
        super().__init__(task)
        self.dest_pos = np.array([0.3, 0.3, 0.05])

    def add_entities(self):
        self.tool = self.scene.add_entity(
            morph = gs.morphs.Mesh(
                file="cutter.obj",
                scale=0.5,
                pos=(0.28, 0.415, 0.15+self.desk_height),
                euler=(0, 0, 90),
                fixed=False,
                collision=True,
                decompose_nonconvex=False,
                convexify=False,
                decimate=False,
            ),
            surface=gs.surfaces.Default(
                color    = (0.1, 0.1, 0.1),
                # vis_mode = 'collision',
            ),  
        )

        self.dough = self.scene.add_entity(
            # material=gs.materials.MPM.ElastoPlastic(E=5000, nu=0.48, rho=600.0, yield_lower = 0.01, yield_higher = 0.002, use_von_mises=True, von_mises_yield_stress=300),
            material=gs.materials.MPM.ElastoPlastic(E=100000, nu=0.3, rho=600.0, yield_lower = 0.0025, yield_higher = 0.0045, use_von_mises=True, von_mises_yield_stress=2000),
            morph=gs.morphs.Box(
                upper=(0.015, 0.42, 0.03+self.desk_height), 
                lower=(-0.015, 0.3, 0.00+self.desk_height)
            ),
            surface=gs.surfaces.Rough(
                color=(231.0 / 255.0, 211.0 / 255.0, 198.0 / 255.0),
                vis_mode='recon',
                recon_backend='splashsurf-1.5-smooth-25'
            ),
        )

    def add_camera_for_gallery(self):
        self.cam_gallery = self.scene.add_camera(
            pos=(0.4,0.4,1.2), 
            lookat=(-0.1,0.4,0.8), 
            fov=30, 
            res=(1440,1440), 
            GUI=False,
        )
        
    def add_camera_for_trajectory(self):
        self.cam_trajectory = self.scene.add_camera(
            pos=(2.1,0.55,2.5), 
            lookat=(-0.3,0.55,0.8), 
            fov=30, 
            res=(1440,1440), 
            GUI=False,
        )
        

env = CutREnv()
env.add_camera_for_gallery()
env.add_camera_for_trajectory()
env.scene.build()


env.xarm.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
)
env.xarm.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
)
env.xarm.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
)

q_tool = np.array([0, 0.36, 0.97, 0.0, 180.0, 0.0])
q_tool[3:] = np.deg2rad(q_tool[3:])
env.tool.set_dofs_position(q_tool)

q_hand = np.array([0.03, 0.36, 0.92, 0.0, 90.0, 0.0])
q_hand[3:] = np.deg2rad(q_hand[3:])
q_xarm, err = env.xarm.inverse_kinematics(
    link=env.xarm.get_link("link_tcp"),
    pos=q_hand[:3],
    quat=euler_to_quat(q_hand[3:]),
    return_error=True,
    respect_joint_limit=False,
)
q_xarm[-2:] = 0.015
env.xarm.set_dofs_position(q_xarm)
env.xarm.control_dofs_position(q_xarm)

env.tool.set_dofs_kp(np.array([250, 250, 250, 100, 100, 100]) * 0.8 * 10)
env.tool.set_dofs_kv(np.array([50, 50, 50, 20, 20, 20]) * 1.4 * 10)

env.scene.visualizer.update()
env.save_gallery_img()
env.save_trajectory_img()

import genesis.utils.geometry_utils as gu

tool_pos = env.tool.get_pos().cpu().numpy()
tool_quat = env.tool.get_quat().cpu().numpy()
tool_T = gu.trans_quat_to_T(tool_pos, tool_quat)

ee_pos = env.xarm.get_link("link_tcp").get_pos().cpu().numpy()
ee_quat = env.xarm.get_link("link_tcp").get_quat().cpu().numpy()
ee_T = gu.trans_quat_to_T(ee_pos, ee_quat)
ee_rel_T = np.linalg.inv(tool_T) @ ee_T

env.tool.set_dofs_velocity((0, 0, -1.0, 0, 0, 0))
for _ in range(100):
    env.scene.step()
    if _ % 50 == 0:
        env.save_trajectory_img()
        env.save_gallery_img()
        print(env.tool.get_AABB(), env.tool.get_dofs_position())

q_tool = np.array([0, 0.56, 0.94, 0.0, 180.0, 0.0])
q_tool[3:] = np.deg2rad(q_tool[3:])
env.tool.control_dofs_position(q_tool)
for _ in range(31):
    q = env.tool.get_dofs_position()
    q[2] = 0.92
    q[0] = 0.03
    env.scene.step()
    if _ % 10 == 0:
        q, err = env.xarm.inverse_kinematics(
            link=env.xarm.get_link("link_tcp"),
            pos=q[:3],
            quat=euler_to_quat(q_hand[3:]),
            return_error=True,
            respect_joint_limit=False,
        )
        q[-2:] = 0.015
        env.xarm.set_dofs_position(q)
        env.scene.visualizer.update()
        env.save_trajectory_img()
        env.save_gallery_img()
