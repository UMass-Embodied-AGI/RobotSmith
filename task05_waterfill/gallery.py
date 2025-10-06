import numpy as np
import genesis as gs
from utils.env_for_render import RenderEnv, euler_to_quat, quat_to_euler

class FillREnv(RenderEnv):
    def __init__(self, task='task05_waterfill'):
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
        self.bottle = self.scene.add_entity(
            material=mat_rigid,
            morph=gs.morphs.Mesh(
                file="bottle.obj",
                scale=0.0015,
                pos=(0.0143, 0.5, 0.156+self.desk_height),
                euler=(90, 0, 90),
                fixed=True,
                collision=True,
                decompose_nonconvex=False,
                convexify=False,
                decimate=False,
            ),
            surface=gs.surfaces.Glass(
                color    = (1.0, 1.0, 1.0, 0.9),
                # vis_mode = 'collision',
            ),  
        )
        self.cup = self.scene.add_entity(
            material=mat_rigid,
            morph = gs.morphs.Mesh(
                file="simplify_red_cup.obj",
                scale=1.0,
                pos=(0, 0.3, 0.064+self.desk_height),
                euler=(90, 0, 0),
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
        self.tool = self.scene.add_entity(
            material=mat_rigid,
            morph = gs.morphs.Mesh(
                file="simplify_funnel.obj",
                scale=0.05,
                pos=(0.3, 0.3, 0.052+self.desk_height),
                euler=(90, 0, 0),
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
        # self.water_emit = self.scene.add_emitter(
        #     material=gs.materials.SPH.Liquid(mu=0.01, sampler="regular"),
        #     max_particles=200000,
        #     surface=gs.surfaces.Default(
        #         color=(0.5, 0.7, 0.9, 1.0),
        #         # vis_mode="recon",
        #     ),
        # )
        self.water = self.scene.add_entity(
            morph=gs.morphs.Nowhere(n_particles=9600),
            material=gs.materials.SPH.Liquid(mu=0.01, sampler="regular"),
            surface=gs.surfaces.Glass(
                color=(0.5, 0.7, 0.9, 1.0),
                vis_mode='recon',
                # recon_backend='splashsurf-1.5', #-smooth-25',
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
            pos=(1.5,0.5,1.5), 
            lookat=(-0.3,0.5,1.0), 
            fov=30, 
            res=(1440,1440), 
            GUI=False,
        )

    def add_camera_for_trajectory(self):
        self.cam_trajectory = self.scene.add_camera(
            pos=(1.5,0.5,1.5), 
            lookat=(-0.3,0.5,1.0), 
            fov=30, 
            res=(1440,1440), 
            GUI=False,
        )
        #     pos=(2.1,0.55,2.5), 
        #     lookat=(-0.3,0.55,0.8), 
        #     fov=30, 
        #     res=(1440,1440), 
        #     GUI=False,
        # )
        

env = FillREnv()
env.add_camera_for_gallery()
env.add_camera_for_trajectory()
env.scene.build()

# cup tensor([0.0000, 0.3000, 0.8640, 1.5708, 0.0000, -0.0000], device='cuda:0')
# tool tensor([0.3000, 0.3000, 0.8520, 1.5708, 0.0000, -0.0000], device='cuda:0')
# bottle tensor([], device='cuda:0')


# env.save_gallery_img()

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

# cup tensor([ 1.2603e-07,  4.0000e-01,  1.2999e+00,  8.3239e-03,  1.1399e-08,
#         -6.9439e-09], device='cuda:0')
q_hand = np.array([0.0, 0.4, 1.3, 180, 0, 90])
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

q_tool = np.array([0.0001, 0.5076, 1.1279, 90, 0, 0])
q_tool[3:] = np.deg2rad(q_tool[3:])
env.tool.set_dofs_position(q_tool)

env.cup.set_dofs_kp(np.array([250, 250, 250, 100, 100, 100]) * 0.8 * 100)
env.cup.set_dofs_kv(np.array([50, 50, 50, 20, 20, 20]) * 1.4 * 100)
env.cup.set_dofs_force_range(-np.array([2500, 2500, 2500, 2500, 2500, 2500]), 
                             np.array([2500, 2500, 2500, 2500, 2500, 2500]))

q_cup = np.array([0.0, 0.4, 1.3, 90, 0, 0])
q_cup[3:] = np.deg2rad(q_cup[3:])
env.cup.set_dofs_position(q_cup)
env.cup.control_dofs_position(q_cup)

water_pos = np.load('water.npy')
env.water.set_pos(0, water_pos)
env.water._solver._kernel_set_particles_active(
    env.water._sim.cur_substep_local,
    0, 9600, 
    gs.ACTIVE,
)


print('cup', env.cup.get_AABB()[0].tolist(), env.cup.get_AABB()[1].tolist())
print('tool', env.tool.get_AABB()[0].tolist(), env.tool.get_AABB()[1].tolist())
print('bottle', env.bottle.get_AABB()[0].tolist(), env.bottle.get_AABB()[1].tolist())

print('')
print('cup', env.cup.get_dofs_position())
print('tool', env.tool.get_dofs_position())
print('bottle', env.bottle.get_dofs_position())

env.scene.visualizer.update()
env.save_gallery_img()
# exit()

env.scene.step()

q_cup_pour = np.array([0.0, 0.4, 1.3, 30, 0, 0])
env.cup.control_dofs_position(q_cup_pour)

for _ in range(130):
    env.xarm.set_dofs_position(q_xarm)
    env.xarm.control_dofs_position(q_xarm)
    env.scene.step()
    if _ % 10 == 0:
        env.save_trajectory_img()

q_cup_new = env.cup.get_dofs_position()
print('cup', env.cup.get_dofs_position())
env.cup.control_dofs_position(q_cup_new)

# new_water_pos = env.water.get_pos()
# np.save('new_water', new_water_pos)

for _ in range(500):
    env.cup.control_dofs_position(q_cup_new)
    env.xarm.set_dofs_position(q_xarm)
    env.xarm.control_dofs_position(q_xarm)
    env.scene.step()
    if _ % 10 == 0:
        env.save_trajectory_img()




# q_tool = np.array([0.0, 0.5071, 1.1616, 90, 0, 0])
# q_tool[3:] = np.deg2rad(q_tool[3:])
# env.tool.set_dofs_position(q_tool)


# env.cup.set_dofs_kp(np.array([250, 250, 250, 100, 100, 100]) * 0.8 * 10)
# env.cup.set_dofs_kv(np.array([50, 50, 50, 20, 20, 20]) * 1.4 * 10)

# q_cup = np.array([0.3,0.3,-0.0766,0,0,1.5708])
# q_cup[3:] = np.deg2rad(q_cup[3:])
# env.cup.set_dofs_position(q_cup)
# env.cup.control_dofs_position(q_cup)

# print('cup', env.cup.get_AABB()[0].tolist(), env.cup.get_AABB()[1].tolist())
# print('tool', env.tool.get_AABB()[0].tolist(), env.tool.get_AABB()[1].tolist())
# print('bottle', env.bottle.get_AABB()[0].tolist(), env.bottle.get_AABB()[1].tolist())

# print('')
# print('cup', env.cup.get_dofs_position())
# print('tool', env.tool.get_dofs_position())
# print('bottle', env.bottle.get_dofs_position())

# env.scene.visualizer.update()
# env.save_gallery_img()
# # exit()

# for _ in range(70):
#     env.scene.step()
#     if _ % 10 == 0:
#         env.save_trajectory_img()

# print(env.tool.get_dofs_position())
# print(env.tool.get_AABB())





# for _ in range(300):
#     env.water_emit.emit(
#         pos=np.array([0.0, 0.4, 1.5,]),
#         direction=np.array([0, 0, -1.0]),
#         speed=1.0,
#         droplet_shape="rectangle",
#         droplet_size=[0.03, 0.03],
#     )
#     env.scene.step()
#     if _ % 10 == 0:
#         env.save_trajectory_img()

# for _ in range(100):
#     env.scene.step()
#     if _ % 10 == 0:
#         env.save_trajectory_img()

# state = env.water_emit._entity.get_state()
# points = state._pos.detach().cpu().numpy()[:env.water_emit._next_particle]
# np.save('water.npy', points)
# print(points.shape)