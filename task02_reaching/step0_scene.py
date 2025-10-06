import numpy as np
import genesis as gs
from utils.env_with_CMAES import CMAESOptimizer

class ReachingEnv(CMAESOptimizer):
    def __init__(self, task='task02_reaching', log_dir=None):
        super().__init__(task, log_dir=log_dir)
        self.dest_pos = np.array([0.3, 0.3, 0.05])

    def add_entities_for_task(self):
        self.dest_item = self.scene.add_entity(
            morph=gs.morphs.Box(
                size=(0.1, 0.1, 0.0005),
                pos=(0.3, 0.3, 0.0005),
            ),
            surface=gs.surfaces.Rough(
                color=(1.0, 0.0, 0.0)
            ),
        )
        self.item = self.scene.add_entity(
            morph=gs.morphs.Box(
                size=(0.1, 0.1, 0.1),
                pos=(0.4, 0.7, 0.05),
            ),
            surface=gs.surfaces.Rough(
                color=(231.0 / 255.0, 211.0 / 255.0, 198.0 / 255.0),
            ),
        ) 

    def metric(self):
        item_pos = self.item.get_pos().cpu().numpy()
        distance = item_pos - self.dest_pos
        # If batch size is 1, return the distance
        if len(distance.shape) == 1:
            L2_distance = np.linalg.norm(distance)
        # If batch size is greater than 1, return the L2 distance for each item
        else:
            L2_distance = np.linalg.norm(distance, axis=1)
        return L2_distance

# env = ReachingEnv()
