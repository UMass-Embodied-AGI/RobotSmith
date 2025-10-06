import numpy as np
import genesis as gs
from utils.env_for_render import RenderEnv, euler_to_quat, quat_to_euler

class LiftREnv(RenderEnv):
    def __init__(self, task='task07_lifting'):
        super().__init__(task)
        self.dest_pos = np.array([0.3, 0.3, 0.05])

    def add_entities(self):
        self.bowl = self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file="simplify_simplify_bowl.obj",
                scale=0.09,
                pos=(0.35, 0.3, 0.05+self.desk_height),
                euler=(90, 0, 90),
                collision=True,
                decompose_nonconvex=False,
                convexify=False,
                decimate=False,
            ),
            surface=gs.surfaces.Default(
                color    = (0.8, 0.8, 0.8),
                # vis_mode = 'collision',
            ),  
        )
        self.tool = self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file="bowl_lifting_tool_2.obj",
                scale=1.7,
                pos=(0.2, 0.3, 0.026+self.desk_height),
                euler=(90, 0, 0),
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


    def add_camera_for_gallery(self):
        # self.cam_gallery = self.scene.add_camera(
        #     pos=(0.2,0.45,1.5), 
        #     lookat=(0,0.45,0.8), 
        #     fov=30, 
        #     res=(1440,1440), 
        #     GUI=False,
        # )
        self.cam_gallery = self.scene.add_camera(
            pos=(1.5,0.3,1.5), 
            lookat=(-0.3,0.3,1.0), 
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
        

env = LiftREnv()
env.add_camera_for_gallery()
env.add_camera_for_trajectory()
env.scene.build()

q_hand = np.array([0.1, 0.2, 1.06, 180, 0, 90])
q_hand[3:] = np.deg2rad(q_hand[3:])
q_xarm, err = env.xarm.inverse_kinematics(
    link=env.xarm.get_link("link_tcp"),
    pos=q_hand[:3],
    quat=euler_to_quat(q_hand[3:]),
    return_error=True,
    respect_joint_limit=False,
)
q_xarm[-2:] = 0.02
env.xarm.set_dofs_position(q_xarm)
env.xarm.control_dofs_position(q_xarm)

q_tool = np.array([0.1, 0.28, 1.066, 90, 0, 0])
q_tool[3:] = np.deg2rad(q_tool[3:])
env.tool.set_dofs_position(q_tool)

print(env.tool.get_AABB())


q_bowl = np.array([0.0, 0.45, 1.08, 90, 0, 0])
q_bowl[3:] = np.deg2rad(q_bowl[3:])
env.bowl.set_dofs_position(q_bowl)
env.bowl.control_dofs_position(q_bowl)

print(env.bowl.get_AABB())

env.scene.visualizer.update()
env.save_gallery_img()
env.save_trajectory_img()