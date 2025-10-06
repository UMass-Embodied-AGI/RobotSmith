import numpy as np
import genesis as gs
from utils.env_with_CMAES import CMAESOptimizer

class LiftingEnv(CMAESOptimizer):
    def __init__(self, task='task07_lifting', log_dir=None):
        super().__init__(task, log_dir=log_dir)
        self.dest_pos = np.array([0.3, 0.3, 0.05])

    def add_entities_for_task(self):
        self.mat_rigid = gs.materials.Rigid(
            coup_friction=0.1,
            coup_softness=0.001,
            coup_restitution=0.001,
            sdf_cell_size=0.0001,
            sdf_min_res=32,
            sdf_max_res=64
        )

        self.bowl = self.scene.add_entity(
            material=self.mat_rigid,
            morph=gs.morphs.Mesh(
                file="simplify_simplify_bowl.obj",
                scale=0.09,
                pos=(0.35, 0.3, 0.05),
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

        for link in self.bowl.links:
            link._inertial_mass = 0.001
        
    def add_tools_for_task(self):
        self.bowl_lifter = self.scene.add_entity(
            material=self.mat_rigid,
            morph=gs.morphs.Mesh(
                file="bowl_lifting_tool_2.obj",
                scale=1.7,
                pos=(0.2, 0.3, 0.026),
                euler=(90, 0, 0),
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

        for link in self.bowl_lifter.links:
            link._inertial_mass = 0.01


    def metric(self):
        
        #raise NotImplementedError("Metric function not implemented for LiftingEnv.")

        bowl_pos = self.bowl.get_pos().cpu().numpy()
        
        if bowl_pos.ndim == 1:
            return bowl_pos[2]
        else:
            return bowl_pos[:, 2]

from utils.api_manipulate import *

if __name__ == "__main__":
    env = LiftingEnv()
    env.build_scene_for_evaluation()
    env.reset()
    env.save_img(iter = 0)
