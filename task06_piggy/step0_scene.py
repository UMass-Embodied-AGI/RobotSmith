import numpy as np
import genesis as gs
from utils.env_with_CMAES import CMAESOptimizer

class PiggyEnv(CMAESOptimizer):
    def __init__(self, task='task06_piggy', log_dir=None):
        super().__init__(task, log_dir=log_dir)

    def add_entities_for_task(self):
        self.mat_rigid = gs.materials.Rigid(
            coup_friction=0.1,
            coup_softness=0.0001,
            coup_restitution=0.0001,
            sdf_cell_size=0.0001,
            sdf_min_res=64,
            sdf_max_res=64, #256
        )
        self.container = self.scene.add_entity(
            material=self.mat_rigid,
            morph = gs.morphs.Mesh(
                file="piggy_fab_simf_b.obj",
                scale=0.057,
                pos=(0.2, 0.2, 0.01),
                euler=(90, 0, 90),
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

        for link in self.container.links:
            link._inertial_mass = 0.01
    
    def add_tools_for_task(self):
        
        self.tool = self.scene.add_entity(
            material=self.mat_rigid,
            morph = gs.morphs.Mesh(
                file="tool.obj",
                scale=0.03,
                pos=(0.4, 0.2, 0.03),
                euler=(90, 0, 90),
                fixed=False,
                collision=True,
                decompose_nonconvex=False,
                convexify=False,
                decimate=False,
            ),
            surface=gs.surfaces.Default(
                color    = (1.0, 0.0, 0.0),
            ),  
        )

        for link in self.tool.links:
            link._inertial_mass = 0.01

    def metric(self):
        
        piggy_pos = self.container.get_pos().cpu().numpy()

        #Return z coordinate of piggy bank
        return piggy_pos[2]


if __name__ == "__main__":

    env = PiggyEnv()
    env.build_scene_for_evaluation()
    env.save_img(iter=0)

    for _ in range(10):
        env.scene.step()
        env.save_img(iter=0)
