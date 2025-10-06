import numpy as np
import genesis as gs
from utils.env_with_CMAES import CMAESOptimizer
from utils.metric import clip_score

class FillEnv(CMAESOptimizer):
    def __init__(self, task='task05_waterfill', log_dir=None):
        super().__init__(task, log_dir=log_dir)
        self.dest_height = 0.02


    def add_entities_for_task(self):
        self.mat_rigid = gs.materials.Rigid(
            coup_friction=0.1,
            coup_softness=0.001,
            coup_restitution=0.001,
            sdf_cell_size=0.0001,
            sdf_min_res=64,
            sdf_max_res=256
        )
        self.bottle = self.scene.add_entity(
            material=self.mat_rigid,
            morph=gs.morphs.Mesh(
                file="bottle.obj",
                scale=0.0015,
                pos=(0.0143, 0.5, 0.156),
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
            material=self.mat_rigid,
            morph = gs.morphs.Mesh(
                file="simplify_red_cup.obj",
                scale=1.0,
                pos=(0, 0.3, 0.064),
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
        self.water = self.scene.add_entity(
            morph=gs.morphs.Nowhere(n_particles=9600),
            material=gs.materials.SPH.Liquid(mu=0.01, sampler="regular"),
            surface=gs.surfaces.Glass(
                color=(0.5, 0.7, 0.9, 1.0),
                vis_mode='recon',
                # recon_backend='splashsurf-1.5', #-smooth-25',
            ),
        )

    def metric(self):
        dough_particles = self.dough.get_particles()
        max_height = np.max(dough_particles[:, 2])
        return max_height
