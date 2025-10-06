import multiprocessing

import numpy as np
import genesis as gs
import time
import cma
import os
from itertools import accumulate
import imageio
import matplotlib.pyplot as plt
from genesis.constants import backend as gs_backend

def run_gs_env(env_id, input_queue, output_queue, project_path, img_steps, img_save_dir):
    gs.init(seed=0, precision='32', logging_level='error', backend=gs_backend.gpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3, substeps=10, gravity=(0,0,-9.8),
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.1,-0.1,-0.1), upper_bound=(0.6,0.6,0.6),
            grid_density=128,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_fov=30, res=(960, 640),
        ),
        vis_options=gs.options.VisOptions(
            env_separate_rigid = True
        ),
        show_viewer = False,
    ) # scene
    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
    ) # plane
    xarm = scene.add_entity(
        gs.morphs.URDF(
            file='{}/assets/xarm7_with_gripper_reduced_dof.urdf'.format(project_path), 
            fixed=True, 
            collision=True, 
            links_to_keep=["link_tcp"]),
    ) # xarm
    dough = scene.add_entity(
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
    ) # dough
    tool = scene.add_entity(
        morph = gs.morphs.Mesh(
            file="tool.obj",
            scale=0.4,
            pos=(0.4, 0.2, 0.02),
            euler=(180, 0, 0),
            fixed=False,
            collision=True,
            decompose_nonconvex=False,
            convexify=False,
            decimate=True,                                                                                                                                                                                                
            decimate_face_num=3000, 
        ),
        surface=gs.surfaces.Default(
            color    = (0.1, 0.1, 0.1),
        ),  
    ) # tool
    cam = scene.add_camera(
        pos=(3.5,0.5,2.5), lookat=(0.0,0.0,0.5), fov=50, res=(1440,1440), GUI=False,
    )
    cam_up = scene.add_camera(
        pos=(0.5,0.5,2.5), lookat=(0.5,0.5,0.5), fov=50, res=(1440,1440), GUI=False,
    )
    cam_right = scene.add_camera(
        pos=(0.5,1.0,0.02), lookat=(0.5,0.5,0.02), fov=50, res=(1440,1440), GUI=False,
    )
    cam_front = scene.add_camera(
        pos=(1.0,0.5,0.02), lookat=(0.5,0.5,0.02), fov=50, res=(1440,1440), GUI=False,
    )

    scene.build()

    for entity in scene.sim.rigid_solver.entities[2:]:
        for link in entity.links:
            link._inertial_mass = 0.015

    tool.set_dofs_kp(np.array([250, 250, 250, 100, 100, 100]) * 2)
    tool.set_dofs_kv(np.array([70, 70, 70, 30, 30, 30]) * 2)

    def control_tool(tool, pos, rot=None):
        if rot is None:
            rot = tool.get_dofs_position()[3:]
        q_tool = np.concatenate((pos, rot))
        tool.control_dofs_position(q_tool)
        for _ in range(200):
            scene.step()

    def evaluate(x, cur_iter, vis, img_steps, img_save_dir):
        x = x.reshape((4, 3))
        scene.reset()
                        
        initial_quat = x[0]

        control_tool(tool, x[1], initial_quat)
        control_tool(tool, x[2], initial_quat)
        control_tool(tool, x[3], initial_quat)
        
        dough_particles = dough.get_particles()
        penalty = dough_particles[:, 2].max()

        return penalty, img_steps
    # ) evaluate

    output_queue.put("ready")
    while True:
        if not input_queue.empty():
            command = input_queue.get()
            if command == "exit":
                output_queue.put(f"Env {env_id} exiting.")
                break  # Stop when "exit" command is received
            elif command[0] == "traj":
                x = command[1] 
                sample_id = command[2]
                cur_iter = command[3]
                vis = command[4]
                reward, img_steps = evaluate(x, cur_iter, vis, img_steps, img_save_dir)

                output_queue.put((env_id, sample_id, reward))
        time.sleep(1)


def init_gs_multiprocess(n_envs, project_path, img_save_dir):
    print('Before everything')
    processes = []
    input_queues = []
    output_queue = multiprocessing.Queue() 
    for i in range(n_envs):
        input_queue = multiprocessing.Queue()
        input_queues.append(input_queue)
        p = multiprocessing.Process(target=run_gs_env, args=(i, input_queue, output_queue, project_path, 0, img_save_dir))
        p.start()
        processes.append(p)
    print("Processes initialized")
    return processes, input_queues, output_queue


def optimize(n_envs, input_queues, output_queue):
    dim, rng, sigma, iters = 9, 1.0, 0.02, 50
    init_params = [(0, 0, 0), (0.55, 0.2, 0.15), (0.4, 0.0, 0.15), (0.4, 0.0, 0.01)]
    init_params = np.array(init_params).flatten()
    lower_bounds = init_params - rng
    upper_bounds = init_params + rng
    print(init_params.shape, init_params)
    es = cma.CMAEvolutionStrategy(
        init_params,
        sigma,
        {
            "maxiter": iters,
            "popsize": n_envs,
            "bounds": [lower_bounds, upper_bounds],
        },
    )

    tot_cnt, plot_x, plot_y = 0, [], []
    busy_env = np.zeros(n_envs)
    for ww in range(iters):
        X = es.ask()
        tell_list, tell_dict = [], {}
        for idx, x in enumerate(X):
            while np.sum(busy_env) == n_envs:
                while not output_queue.empty():
                    env_id, sample_id, reward = output_queue.get()
                    busy_env[env_id] = 0
                    tell_dict[sample_id] = reward

                time.sleep(1)
            for i in range(n_envs):
                if busy_env[i] != 1:
                    input_queues[i].put(("traj", x, idx, ww, False))
                    busy_env[i] = 1
                    break

        while np.sum(busy_env) != 0:
            while not output_queue.empty():
                env_id, sample_id, reward = output_queue.get()
                busy_env[env_id] = 0
                tell_dict[sample_id] = reward
            time.sleep(1)
        
        print(f"Iteration {ww} finished")
        print(len(tell_dict))

        while len(tell_dict) < len(X):
            while not output_queue.empty():
                env_id, sample_id, reward = output_queue.get()
                busy_env[env_id] = 0
                tell_dict[sample_id] = reward
            time.sleep(1)

        for i in range(len(X)):
            tell_list.append(tell_dict[i])
            tot_cnt += 1
            plot_x.append(tot_cnt)
            plot_y.append(tell_dict[i])

        es.tell(X, tell_list)
        es.disp()
        prefix_min = list(accumulate(plot_y, min))
        
        plt.plot(plot_x, prefix_min)
        plt.savefig(os.path.join(img_save_dir, "plot.png"))

        fbest = np.array(es.result.fbest) # penalty
        print('iter = ', ww, '   min penalty = ', fbest)
        
        xbest = np.array(es.result.xbest) # traj
        np.save(os.path.join(img_save_dir, "best_traj.npy"), xbest)
        with open(os.path.join(img_save_dir, "best_score.txt"), 'w') as fo:
            fo.write(f"{fbest:.4f}\n")

        with open(os.path.join(img_save_dir, "cmaes_ckpt.pkl"), 'wb') as f:
            f.write(es.pickle_dumps())
    
    result = es.result
    x = result.xbest

    env_id, sample_id, reward = output_queue.get()


if __name__ == '__main__':
    img_steps = 0
    project_path = os.path.dirname(os.path.join(os.path.abspath(__file__), '..'))

    log_dir = os.path.join(project_path, 'task03_flatten', 'try')
    os.makedirs(log_dir, exist_ok=True)
    n_tries = len([fil for fil in os.listdir(log_dir) if not '.' in fil])
    img_save_dir = os.path.join(log_dir, f"{n_tries:03d}")
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(os.path.join(img_save_dir, "opt_log"), exist_ok=True)
    

    multiprocessing.freeze_support()
    multiprocessing.set_start_method("spawn", force=True)
    print('Cool!!!!')

    n_envs = 3
    processes, input_queues, output_queue = init_gs_multiprocess(n_envs, project_path, img_save_dir)
    ready_cnt = 0
    while ready_cnt < n_envs:
        if not output_queue.empty() and output_queue.get() == "ready":
            ready_cnt += 1
        time.sleep(1)

    optimize(n_envs, input_queues, output_queue)

    for q in input_queues:
        q.put("exit")

    for p in processes:
        p.join()
