import numpy as np
import genesis as gs
from utils.env_for_render import RenderEnv, euler_to_quat, quat_to_euler

class CalabashREnv(RenderEnv):
    def __init__(self, task='task01_calabash'):
        super().__init__(task)
        self.dest_pos = np.array([0.3, 0.3, 0.05])

    def add_entities(self):
        self.mold_top = self.scene.add_entity(
            morph = gs.morphs.Mesh(
                file="mold_top.stl",
                scale=0.05,
                pos=(0, 0.55, 0.001+self.desk_height),
                euler=(-90, 0, 90),
                fixed=False,
                collision=True,
                decompose_nonconvex=False,
                convexify=False,
                decimate=False,
            ),
            surface=gs.surfaces.Default(
                color    = (0.1, 0.1, 0.1),
            ),  
        )
        self.mold_bottom = self.scene.add_entity(
            morph = gs.morphs.Mesh(
                file="mold_bottom.stl",
                scale=0.05,
                pos=(0, 0.45, 0.041+self.desk_height) ,
                euler=(-90, 0, 90),
                fixed=False,
                collision=True,
                decompose_nonconvex=False,
                convexify=False,
                decimate=False,
            ),
            surface=gs.surfaces.Default(
                color    = (0.1, 0.1, 0.1),
            ),
        )
        
        self.dough = self.scene.add_entity(
            material=gs.materials.MPM.ElastoPlastic(E=100000, nu=0.3, rho=600.0, yield_lower = 0.0025, yield_higher = 0.0045, use_von_mises=True, von_mises_yield_stress=2000),
            morph=gs.morphs.Box(
                upper=(0.035, 0.48, 0.11+self.desk_height), 
                lower=(-0.035, 0.42, 0.05+self.desk_height)
            ),
            surface=gs.surfaces.Rough(
                color=(231.0 / 255.0, 211.0 / 255.0, 198.0 / 255.0),
                vis_mode='recon',
                recon_backend='splashsurf-1.5-smooth-25'
            ),
        )

    def add_camera_for_gallery(self):
        self.cam_gallery = self.scene.add_camera(
            pos=(0.2,0.45,1.5), 
            lookat=(0,0.45,0.8), 
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
        

env = CalabashREnv()
env.add_camera_for_gallery()
env.add_camera_for_trajectory()
env.scene.build()

bbox = env.mold_bottom.get_AABB()

def in_bbox(pos):
    eps = 0.01
    if pos[0] <= bbox[0][0] + eps or pos[0] >= bbox[1][0] - eps:
        return False
    if pos[1] <= bbox[0][1] + eps or pos[1] >= bbox[1][1] - eps:
        return False
    return True

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

q_top = np.array([0, 0.45, 0.912, 0.0, -90.0, 90.0])
q_top[3:] = np.deg2rad(q_top[3:])
env.mold_top.set_dofs_position(q_top)

q_hand = np.array([0.0, 0.2, 1.0, 0.0, 90.0, 0.0])
q_hand[3:] = np.deg2rad(q_hand[3:])
q_xarm, err = env.xarm.inverse_kinematics(
    link=env.xarm.get_link("link_tcp"),
    pos=q_hand[:3],
    quat=euler_to_quat(q_hand[3:]),
    return_error=True,
    respect_joint_limit=False,
)
env.xarm.set_dofs_position(q_xarm)
env.xarm.control_dofs_position(q_xarm)

env.mold_top.set_dofs_kp(np.array([250, 250, 250, 100, 100, 100]) * 0.8 * 10)
env.mold_top.set_dofs_kv(np.array([50, 50, 50, 20, 20, 20]) * 1.4 * 10)
env.mold_top.control_dofs_position([0.3,0.3,-0.0766,0,0,1.5708])

env.mold_top.control_dofs_velocity(np.array([0,0,-0.8,0,0,0]))

for _ in range(100):
    env.scene.step()
    if _ % 10 == 0:
        env.save_trajectory_img()

env.mold_top.set_dofs_position((0, 0.35, 0.902, 0.0, 0.0, 90.0))
env.scene.step()
env.save_trajectory_img()

q_top = np.array([0, 0.35, 0.902, 0.0, -90.0, -10.0])
q_top[3:] = np.deg2rad(q_top[3:])
env.mold_top.set_dofs_position(q_top)

q_hand = np.array([0, 0.35, 0.902, -90, 90, 0])
q_hand[3:] = np.deg2rad(q_hand[3:])
q_xarm, err = env.xarm.inverse_kinematics(
    link=env.xarm.get_link("link_tcp"),
    pos=q_hand[:3],
    quat=euler_to_quat(q_hand[3:]),
    return_error=True,
    respect_joint_limit=False,
)
env.xarm.set_dofs_position(q_xarm)

env.scene.visualizer.update()
env.save_gallery_img()


