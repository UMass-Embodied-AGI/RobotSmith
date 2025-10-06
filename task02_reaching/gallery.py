import numpy as np
import genesis as gs
from utils.env_for_render import RenderEnv, euler_to_quat, quat_to_euler

class ReachingREnv(RenderEnv):
    def __init__(self, task='task02_reaching'):
        super().__init__(task)
        self.dest_pos = np.array([0.3, 0.3, 0.05])

    def add_entities(self):
        self.dest_item = self.scene.add_entity(
            morph=gs.morphs.Box(
                size=(0.1, 0.1, 0.0005),
                pos=(0.3, 0.3, 0.0005 + self.desk_height),
            ),
            surface=gs.surfaces.Rough(
                color=(1.0, 0.0, 0.0)
            ),
        )
        self.item = self.scene.add_entity(
            morph=gs.morphs.Box(
                size=(0.1, 0.1, 0.1),
                pos=(0.4, 0.7, 0.05 + self.desk_height),
            ),
            surface=gs.surfaces.Rough(
                color=(231.0 / 255.0, 211.0 / 255.0, 198.0 / 255.0),
            ),
        ) 
        self.tool = self.scene.add_entity(
            material=gs.materials.Rigid(friction = 5.0),
            morph = gs.morphs.Mesh(
                file="tool_2.obj",
                scale=1.0,
                pos=(0.2, 0.2, 0.2 + self.desk_height),
                euler=(90, 0, 90),
                fixed=False,
                collision=True,
                decompose_nonconvex=False,
                convexify=False,
                decimate=False,
            ),
            surface=gs.surfaces.Default(
                # color    = (0.8, 0.0, 0.0),
                color    = (0.1, 0.1, 0.1),
            ),  
        )

    def add_camera_for_gallery(self):
        self.cam_gallery = self.scene.add_camera(
            pos=(2.1,0.55,2.5), 
            lookat=(-0.3,0.55,0.8), 
            fov=30, 
            res=(1440,1440), 
            GUI=False,
        )
        
    def add_camera_for_trajectory(self):
        self.cam_tradd_camera_for_trajectory = self.scene.add_camera(
            pos=(3.5,2.5,2.5), 
            lookat=(0.0,0.0,0.5), 
            fov=50, 
            res=(1440,1440), 
            GUI=False,
        )
        

env = ReachingREnv()
env.add_camera_for_gallery()
# env.add_camera_for_trajectory()
env.scene.build()

env.item.set_pos((0.4, 1.0, 0.05 + env.desk_height))

q_hand = np.array([0.4, 0.6, 1.0, -1, 90+90, 80])
q_hand[3:] = np.deg2rad(q_hand[3:])
q_xarm, err = env.xarm.inverse_kinematics(
    link=env.xarm.get_link("link_tcp"),
    pos=q_hand[:3],
    quat=euler_to_quat(q_hand[3:]),
    return_error=True,
    respect_joint_limit=False,
)
env.xarm.set_dofs_position(q_xarm)

q_tool = np.array([0.4, 0.6, 1.0, -1, 90, 80])
q_tool[3:] = np.deg2rad(q_tool[3:])
env.tool.set_dofs_position(q_tool)

env.scene.step()

env.save_gallery_img()
