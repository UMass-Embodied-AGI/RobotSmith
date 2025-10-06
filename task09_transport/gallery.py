import numpy as np
import genesis as gs
from utils.env_for_render import RenderEnv, euler_to_quat, quat_to_euler

class TransportREnv(RenderEnv):
    def __init__(self, task='task09_transport'):
        super().__init__(task)
        self.dest_pos = np.array([0.3, 0.3, 0.05])

    def add_entities(self):
        mat_rigid = gs.materials.Rigid(
            coup_friction=0.1,
            coup_softness=0.001,
            coup_restitution=0.001,
            sdf_cell_size=0.0001,
            sdf_min_res=64,
            sdf_max_res=256
        )
        self.cup = self.scene.add_entity(
            material=mat_rigid,
            morph = gs.morphs.Mesh(
                file="simplify_red_cup.obj",
                scale=1.0,
                pos=(0, 0.3, 0.064+self.desk_height),
                euler=(90, 0, 0),
                fixed=True,
                collision=True,
                decompose_nonconvex=False,
                convexify=False,
                decimate=False,
            ),
            surface=gs.surfaces.Glass(
                color    = (1.0, 1.0, 1.0, 0.9),
            ),  
        )
        self.tank = self.scene.add_entity(
            material=mat_rigid,
            morph = gs.morphs.Mesh(
                file='simplify_tank.obj',
                scale=0.3,
                pos=(0.1, 0.6, 0.07+self.desk_height),
                euler=(90, 0, 90),
                fixed=True,
                collision=True,
                decompose_nonconvex=False,
                convexify=False,
                decimate=False,
            ),
            surface=gs.surfaces.Glass(
                color    = (0.8, 0.8, 0.8, 0.5),
            ),  
        )        
        eps=0.03
        self.water = self.scene.add_entity(
            material=gs.materials.SPH.Liquid(mu=0.01, sampler="regular"),
            morph = gs.morphs.Box(
                pos=(0.1, 0.6, 0.07+self.desk_height),
                size=(0.15, 0.25, 0.2),
            ),
            surface=gs.surfaces.Glass(
                color=(0.5, 0.7, 0.9, 0.5),
                vis_mode='recon',
                # recon_backend='splashsurf-1.5', #-smooth-25',
            ),
        )
        self.bowl_water = self.scene.add_entity(
            material=gs.materials.SPH.Liquid(mu=0.01, sampler="regular"),
            morph = gs.morphs.Cylinder(
                pos=(0.01, 0.20, 1.15),
                radius=0.03,
                height=0.15,
            ),
            surface=gs.surfaces.Glass(
                color=(0.5, 0.7, 0.9, 1.0),
                vis_mode='recon',
            ),  
        )

        self.tool = self.scene.add_entity(
            material=mat_rigid,
            morph = gs.morphs.Mesh(
                file="solidify_tool_2.obj",
                scale=1.3,
                pos=(0.3, 0.3, 0.2+self.desk_height),
                euler=(90, 0, 90),
                fixed=False,
                collision=True,
                decompose_nonconvex=False,
                convexify=False,
                decimate=True,
                decimate_face_num=3000,
            ),
            surface=gs.surfaces.Default(
                color    = (0.1, 0.1, 0.1),
            ),
        )


    def add_camera_for_gallery(self):
        self.cam_gallery = self.scene.add_camera(
            pos=(1.5,0.5,1.5), 
            lookat=(-0.3,0.5,1.0), 
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
        

env = TransportREnv()
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

env.tool.set_dofs_kp(np.array([250, 250, 250, 100, 100, 100]) * 0.8 * 100)
env.tool.set_dofs_kv(np.array([50, 50, 50, 20, 20, 20]) * 1.4 * 100)
env.tool.set_dofs_force_range(-np.array([2500, 2500, 2500, 2500, 2500, 2500]), 
                             np.array([2500, 2500, 2500, 2500, 2500, 2500]))

q_hand = np.array([0.2, 0.35, 1.3, 180, 0, 90])
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
water0 = np.load('water0.npy')

q_tool = np.array([0.01, 0.20, 1.05, 0, 90, 90])
q_tool[3:] = np.deg2rad(q_tool[3:])
env.tool.set_dofs_position(q_tool)
env.tool.control_dofs_position(q_tool)

env.scene.step()


for _ in range(1500):
    env.scene.step()

q_tilt = np.array([0.01, 0.20, 1.05, 90, 0, -60])
q_tool[3:] = np.deg2rad(q_tool[3:])
env.tool.control_dofs_position(q_tool)

for _ in range(2000):
    env.scene.step()

arr = np.array([9, 1.7, 41.5, 104.7, 16.5,93.9,-35.8, 0.015, 0.015])
arr[:-2] = np.deg2rad(arr[:-2])

ik_pos = [-0.1, 0.38, 1.15]
ik_euler = np.deg2rad([0, 150, 90])
ik_quat = euler_to_quat(ik_euler)

q_xarm, err = env.xarm.inverse_kinematics(
    link=env.xarm.get_link("link_tcp"),
    pos=ik_pos,
    quat=ik_quat,
    return_error=True,
    respect_joint_limit=False,
)
q_xarm[-2:] = 0.041
env.xarm.set_dofs_position(q_xarm)
env.xarm.control_dofs_position(q_xarm)
env.scene.step()
env.water.set_pos(0, water0)
env.save_trajectory_img()