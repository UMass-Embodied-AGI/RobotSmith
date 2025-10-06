import numpy as np
import genesis as gs
from env_with_CMAES import CMAESOptimizer
from utils.metric import clip_score

class CalabashEnv(CMAESOptimizer):
    def __init__(self, task='task01_calabash'):
        super().__init__(task)

    def add_entities_for_task(self):
        self.dough = self.scene.add_entity(
            material=gs.materials.MPM.ElastoPlastic(E=100000, nu=0.3, rho=600.0, yield_lower = 0.0025, yield_higher = 0.0045, use_von_mises=True, von_mises_yield_stress=2000),
            morph=gs.morphs.Box(
                upper=(0.38, 0.035, 0.065), 
                lower=(0.32, -0.035, 0.005)
            ),
            surface=gs.surfaces.Rough(
                color=(231.0 / 255.0, 211.0 / 255.0, 198.0 / 255.0),
                vis_mode='recon',
                recon_backend='splashsurf-1.5-smooth-25'
            ),
        )

    def metric(self):
        import imageio
        from PIL import Image
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=4e-3, substeps=10, gravity=(0,0,-9.8),
            ),
            mpm_options=gs.options.MPMOptions(
                lower_bound=(-0.1,-0.1,-0.1), upper_bound=(0.6,0.6,0.6),
                grid_density=256, #enable_CPIC=True, 
            ),
            sph_options=gs.options.SPHOptions(
                particle_size=0.004, lower_bound=(-2,-2,0), upper_bound=(2,2,2)
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_fov=30, res=(960, 640),
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame = True
            ),
            show_viewer = False,
        )
        self.dough = scene.add_entity(
            material=gs.materials.MPM.ElastoPlastic(E=100000, nu=0.3, rho=600.0, yield_lower = 0.0025, yield_higher = 0.0045, use_von_mises=True, von_mises_yield_stress=2000),
            morph=gs.morphs.Box(
                upper=(0.38, 0.035, 0.065), 
                lower=(0.32, -0.035, 0.005)
            ),
            surface=gs.surfaces.Rough(
                color=(231.0 / 255.0, 211.0 / 255.0, 198.0 / 255.0),
                vis_mode='recon',
                recon_backend='splashsurf-1.5-smooth-25'
            ),
        )
        self.dough_pos = (0.5, 0.5, 0.5)
        n_cams, distance = 8, 0.2
        self.cam_evals = []
        for _ in range(n_cams):
            direction = np.random.rand(3)
            direction *= distance / np.linalg.norm(direction)
            cam = scene.add_camera(
                pos=self.dough_pos + direction,
                lookat=self.dough_pos,
                fov=50,
                res=(512, 512),
                GUI=False,
            )
            self.cam_evals.append(cam)

        scene.build()
        
        dough_particles = np.load("dough_particles.npy")
        com = np.mean(dough_particles, axis=0)
        dough_particles = dough_particles - com + self.dough_pos
        self.dough.set_pos(0, dough_particles)
        scene.visualizer.update()
        N, scores = len(self.cam_evals), []

        for i, cam in enumerate(self.cam_evals):
            img = cam.render()[0]
            filename = f"cam{i}.png"
            imageio.imwrite(filename, img)
            img = Image.fromarray(img)
            scr = max(clip_score(img, "calabash shape dough"), clip_score(img, "calabash"))
            scores.append(scr)
            print(scr)
        score = np.array(scores).max()

        return score

# env = ReachingEnv()
