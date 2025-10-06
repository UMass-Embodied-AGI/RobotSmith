import numpy as np
import genesis as gs
from utils.env_with_CMAES import CMAESOptimizer
from utils.metric import clip_score

class FlattenEnv(CMAESOptimizer):
    def __init__(self, task='task03_flatten', log_dir=None):
        super().__init__(task, log_dir=log_dir)
        self.dest_height = 0.02

    def add_entities_for_task(self):
        self.dough = self.scene.add_entity(
            material=gs.materials.MPM.ElastoPlastic(E=100000, nu=0.3, rho=600.0, yield_lower = 0.0025, yield_higher = 0.0045, use_von_mises=True, von_mises_yield_stress=2000),
            morph=gs.morphs.Box(
                upper=(0.38, 0.035, 0.075), 
                lower=(0.32, -0.035, 0.005)
            ),
            surface=gs.surfaces.Rough(
                color=(231.0 / 255.0, 211.0 / 255.0, 198.0 / 255.0),
                vis_mode='recon',
                recon_backend='splashsurf-1.5-smooth-25'
            ),
        )

    def metric(self):
        dough_particles = self.dough.get_particles()
        max_height = np.max(dough_particles[:, 2])
        return max_height
