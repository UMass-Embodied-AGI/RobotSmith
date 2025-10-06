import numpy as np
import genesis as gs
from utils.env_with_CMAES import CMAESOptimizer

class HolderEnv(CMAESOptimizer):
    def __init__(self, task='task04_holder', log_dir=None):
        super().__init__(task, log_dir=log_dir)
        self.dest_pos = np.array([0.3, 0.3, 0.05])

    def add_entities_for_task(self):
        self.holder = self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file="holder_2000.obj",
                scale=0.054,
                pos=(0.2, 0.4, 0.08),
                euler=(90, 0, 90),
                fixed=True,
                collision=True,
                decompose_nonconvex=False,
                convexify=False,
                decimate=False,
            ),
            surface=gs.surfaces.Default(
                color    = (0.8, 0.0, 0.0),
                # vis_mode = 'collision',
            ),  
        )

        self.phone = self.scene.add_entity(
            morph = gs.morphs.Mesh(
                file="iphone_2000.obj",
                scale=0.027,
                pos=(0.4, 0.2, 0.08),
                euler=(90, 0, 0),
                fixed=False,
                collision=True,
                decompose_nonconvex=False,
                convexify=False,
                decimate=False,
            ),
            surface=gs.surfaces.Default(
                color    = (0.0, 0.0, 0.8),
                #vis_mode = 'collision',
            ),  
        )
        self.block = self.scene.add_entity(
            morph=gs.morphs.Box(
                size=(0.045, 0.25, 0.05),
                pos=(0.4, 0.2, 0.05),
                fixed=True,
                collision=True,
                #decompose_nonconvex=False,
                #convexify=False,
                #decimate=False,
            ),
            surface=gs.surfaces.Default(
                color    = (0.0, 0.8, 0.0),
                #vis_mode = 'collision',
            ),  
        )

        for link in self.phone.links:
            link._inertial_mass = 0.01

        self.target_qpos = [0.2005, 0.4279, 0.0617, 0.9820, 0.1872, 0.0069, 0.0241]


    def metric(self):
        
        #raise NotImplementedError("Metric function not implemented for HolderEnv")

        item_pos = self.phone.get_pos().cpu().numpy()
        item_quat = self.phone.get_quat().cpu().numpy()
        target_qpos = np.array([0.2005, 0.4279, 0.0617, 0.9820, 0.1872, 0.0069, 0.0241])
        if self.n_envs == 1:
            item_qpos = np.concatenate((item_pos, item_quat), axis=-1)
            dist = item_qpos - target_qpos
            L2_loss = np.sum(dist ** 2)**0.5
        else:
            target_qpos = np.tile(target_qpos, (self.n_envs, 1))
            item_qpos = np.concatenate((item_pos, item_quat), axis=-1)
            dist = item_qpos - target_qpos
            L2_loss = np.sum(dist ** 2, axis=-1)**0.5
        return L2_loss
    

# env = ReachingEnv()
