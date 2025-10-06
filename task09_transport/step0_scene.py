import numpy as np
import genesis as gs
from utils.env_with_CMAES import CMAESOptimizer

class TransportingEnv(CMAESOptimizer):
    def __init__(self, task='task09_transport'):
        super().__init__(task)

    def add_entities_for_task(self):
        mat_rigid = gs.materials.Rigid(
            coup_friction=0.1,
            coup_softness=0.0001,
            coup_restitution=0.001,
            sdf_cell_size=0.0001,
            sdf_min_res=64,
            sdf_max_res=64, #256
        )
        self.cup = self.scene.add_entity(
            material=mat_rigid,
            morph = gs.morphs.Mesh(
                file="cup2.obj",
                scale=0.05,
                pos=(0.3, -0.2, 0.001),
                euler=(90, 0, 0),
                fixed=False,
                collision=True,
                decompose_nonconvex=False,
                convexify=False,
                decimate=False,
            ),
            surface=gs.surfaces.Default(
                color    = (0.0, 0.0, 0.8),
            ),  
        )
        self.tank = self.scene.add_entity(
            material=mat_rigid,
            morph = gs.morphs.Mesh(
                file='tank.obj',
                scale=0.3,
                pos=(0.4, 0.2, 0.07),
                euler=(90, 0, 0),
                fixed=True,
                collision=True,
                decompose_nonconvex=False,
                convexify=False,
                decimate=False,
            ),
            surface=gs.surfaces.Default(
                color    = (0.8, 0.8, 0.8, 0.5),
            ),  
        )
        eps=0.03
        self.water = self.scene.add_entity(
            material=gs.materials.SPH.Liquid(mu=0.01, sampler="regular"),
            morph = gs.morphs.Box(
                lower=(0.26+eps, 0.09+eps, 0.01+eps),
                upper=(0.54-eps, 0.31-eps, 0.13-eps),
            ),
            surface=gs.surfaces.Default(
                color=(0.5, 0.7, 0.9, 0.7),
            ),  
        )
    
    def add_tools_for_task(self):
        return None

    def metric(self):
        item_pos = self.item.get_pos().cpu().numpy()
        L2_distance = np.linalg.norm(item_pos - self.dest_pos)
        return L2_distance



env = TransportingEnv()
env.build_scene_for_evaluation()
for _ in range(200):
    env.scene.step()
    if _ % 20 == 0:
        env.save_img(iter=0)
# print(env.cup.get_AABB())
# print(env.tank.get_AABB())