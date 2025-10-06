import numpy as np
import genesis as gs
from utils.env_for_render import RenderEnv, euler_to_quat, quat_to_euler

class HoldREnv(RenderEnv):
    def __init__(self, task='task04_holder'):
        super().__init__(task)
        self.dest_pos = np.array([0.3, 0.3, 0.05])

    def add_entities(self):
        self.tool = self.scene.add_entity(
            morph = gs.morphs.Mesh(
                file="holder_2000.obj",
                scale=0.06,
                pos=(0.2, 0.4, 0.06+self.desk_height),
                euler=(90, 0, 90),
                fixed=True,
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
        self.phone = self.scene.add_entity(
            morph = gs.morphs.Mesh(
                file="iphone15.glb",
                scale=0.1,
                pos=(0.2, 0.44, 0.04+self.desk_height),
                euler=(90, 0, 180),
                fixed=False,
                collision=True,
                decompose_nonconvex=False,
                convexify=False,
                decimate=False,
            ),
            # surface=gs.surfaces.Default(
            #     color    = (0.0, 0.0, 0.8),
            #     #vis_mode = 'collision',
            # ),  
        )


    def add_camera_for_gallery(self):
        self.cam_gallery = self.scene.add_camera(
            pos=(0.65,0.2,1.2), 
            lookat=(0,0.45,0.8), 
            fov=40, 
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
        

env = HoldREnv()
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

print('tool', env.tool.get_pos(), env.tool.get_AABB()[0], env.tool.get_AABB()[1])
print('phone', env.phone.get_pos(), env.phone.get_AABB()[0], env.phone.get_AABB()[1])

# q_tool = np.array([0, 0.36, 0.97, 0.0, 180.0, 0.0])
# q_tool[3:] = np.deg2rad(q_tool[3:])
# env.tool.set_dofs_position(q_tool)

q_phone = np.array([0.2, 0.4409, 0.8176, -1.1768, 0.0056, -3.0352])
env.phone.set_dofs_position(q_phone)

q_hand = np.array([0.2, 0.4, 1.1, 0.0, 70.0, 90])
q_hand[3:] = np.deg2rad(q_hand[3:])
q_xarm, err = env.xarm.inverse_kinematics(
    link=env.xarm.get_link("link_tcp"),
    pos=q_hand[:3],
    quat=euler_to_quat(q_hand[3:]),
    return_error=True,
    respect_joint_limit=False,
)
# q_xarm[-2:] = 0.015
env.xarm.set_dofs_position(q_xarm)
env.xarm.control_dofs_position(q_xarm)

# for _ in range(100):
#     env.scene.step()
#     if _ % 10 == 0:
#         env.save_gallery_img()
#         env.save_trajectory_img()

# print(env.phone.get_dofs_position())


env.scene.visualizer.update()
env.save_gallery_img()
env.save_trajectory_img()




