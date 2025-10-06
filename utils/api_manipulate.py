import genesis as gs
import imageio
import torch
import os
import numpy as np
import trimesh
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from genesis.engine.entities import MPMEntity

def euler_to_quat(euler, degree=True):
    if degree:
        euler = np.deg2rad(np.array(euler))
    cy = np.cos(euler[2] * 0.5)
    sy = np.sin(euler[2] * 0.5)
    cp = np.cos(euler[1] * 0.5)
    sp = np.sin(euler[1] * 0.5)
    cr = np.cos(euler[0] * 0.5)
    sr = np.sin(euler[0] * 0.5)

    q = np.zeros(4)
    q[0] = cy * cp * cr + sy * sp * sr
    q[1] = cy * cp * sr - sy * sp * cr
    q[2] = sy * cp * sr + cy * sp * cr
    q[3] = sy * cp * cr - cy * sp * sr
    return q

def quat_xyzw_to_wxyz(quat):
    return np.array([quat[3], quat[0], quat[1], quat[2]])

def quat_wxyz_to_xyzw(quat):
    return np.array([quat[1], quat[2], quat[3], quat[0]])

def get_transformation_from_pos_quat(pos, quat):
    quat_scipy = quat_wxyz_to_xyzw(quat)
    r = R.from_quat(quat_scipy)
    r_matrix = r.as_matrix()
    T = np.zeros((4, 4))
    T[:3, :3] = r_matrix
    T[:3, 3] = pos
    T[3, 3] = 1
    return T

def rotation_matrix_x(theta):
    """Return a 3x3 rotation matrix for a rotation around the x-axis by angle theta."""
    return R.from_matrix(np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ]))

def align_gripper_z_with_normal(normal, horizontal=False, randomize=False, flip=False):
    n_WS = normal
    Gz = n_WS 
    if horizontal:
        y = np.array([1, 0, 0])
        if flip:
            y = np.array([0.0, 1, 0])
        if randomize:
            succeed = False
            while not succeed:
                y = np.random.uniform(-1, 1, 3)
                y /= np.linalg.norm(y)
                deg1 = np.rad2deg(np.arccos(np.dot(y, np.array([0.0, -1, 0]))))
                deg2 = np.rad2deg(np.arccos(np.dot(y, np.array([0.0, -1, 0]))))
                if deg1 < 30 or deg2 < 30:
                    succeed = True
                    break
    else:
        y = np.array([0.0, 0.0, -1.0])
        if flip:
            y = np.array([0.0, 0.0, 1.0])
        if randomize:
            succeed = False
            while not succeed:
                y = np.random.uniform(-1, 1, 3)
                y /= np.linalg.norm(y)
                deg1 = np.rad2deg(np.arccos(np.dot(y, np.array([0.0, 0.0, -1.0]))))
                deg2 = np.rad2deg(np.arccos(np.dot(y, np.array([0.0, 0.0, 1.0]))))
                if deg1 < 30 or deg2 < 30:
                    succeed = True
                    break
    y = np.array([1, 0, 0])
    Gy = y - np.dot(y, Gz) * Gz
    Gx = np.cross(Gy, Gz)
    R_WG = R.from_matrix(np.vstack((Gx, Gy, Gz)).T)
    return R_WG

def align_gripper_x_with_normal(normal):
    n_WS = normal
    Gx = n_WS  
    y = np.array([0.0, -1, 0])

    Gy = y - np.dot(y, Gx) * Gx
    Gz = np.cross(Gx, Gy)
    R_WG = R.from_matrix(np.vstack((Gx, Gy, Gz)).T)
    return R_WG

def reset_robotarm(robotarm):
    # for xarm
    robotarm.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 300, 300]),
    )
    robotarm.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 30, 30]),
    )
    robotarm.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -300, -300]),
        np.array([87, 87, 87, 87, 12, 12, 12, 300, 300]),
    )
    q_robotarm = np.array([ 0.5981, -0.1136,  0.1676,  0.2658,  0.0286,  0.3049, -0.8370,  0.0187,
         0.0207])
    v_robotarm = np.array([ 7.1592e-05,  1.8199e-04,  2.2042e-05, -3.4953e-05, -3.0166e-05,
         1.2672e-05, -9.9197e-06,  1.9492e-03, -2.6091e-03])    
    robotarm.set_dofs_position(q_robotarm)
    robotarm.set_dofs_velocity(v_robotarm)

def batch_adjust_gripper_pose(scene, robotarm, pos, quat, save_img):
    pass

def adjust_gripper_pose_without_plan(scene, robotarm, pos, quat, save_img, q_gripper, verbose=False, show=False):
    '''
    Args:
        scene: the scene object
        robotarm: the robotarm object
        pos: the target position or 'same' to use the current position
        quat: the target quaternion or 'same' to use the current quaternion
        save_img: the function to save images
    '''
    ee_name = "link_tcp"
    if ee_name not in robotarm.links[-1].name:
        ee_name = "hand"
    if isinstance(quat, str):
        if quat == "same":
            quat = robotarm.get_link(ee_name).get_quat().cpu()
        else:
            raise ValueError("quat should be a quaternion or 'same'")
    if isinstance(pos, str):
        if pos == "same":
            pos = robotarm.get_link(ee_name).get_pos().cpu()
        else:
            raise ValueError("pos should be a position or 'same'")
    pos = np.array(pos)
    quat = np.array(quat)
    # cur_q = robotarm.get_qpos()
    # control_grip = 0 if cur_q[-1] < 0.01 else 0.044
    print('ik_pos', pos)
    print('ik_quat', quat)
    print('ee_name', ee_name)
    q, err = robotarm.inverse_kinematics(
        link=robotarm.get_link(ee_name),
        pos=pos,
        quat=quat,
        return_error=True,
        respect_joint_limit=False,
    )
    q[-2:] = q_gripper
    if show:
        robotarm.set_dofs_position(q)
    else:
        robotarm.control_dofs_position(q)
        print('ee_name', ee_name)
        print('pos, quat', pos, quat)
        print('q', q)
        for _ in range(130):
            scene.step()
            if verbose and _ % 25 == 0:
                save_img()

def adjust_gripper_pose(scene, robotarm, pos, quat, save_img): 
    ee_name = "link_tcp"
    if ee_name not in robotarm.links[-1].name:
        ee_name = "hand"

    pos, quat = np.array(pos), np.array(quat)
    
    cur_q = robotarm.get_qpos()
    open_cur_q = robotarm.get_qpos()
    open_cur_q[-2:] = 0
    control_grip = 0 if cur_q[-1] < 0.01 else 0.044
    q, err = robotarm.inverse_kinematics(
        link=robotarm.get_link(ee_name),
        pos=pos,
        quat=quat,
        return_error=True,
        respect_joint_limit=False,
    )
    q[-2:] = 0
    
    robotarm.set_dofs_position(open_cur_q)
    waypoints = robotarm.plan_path(q, planner='BITstar')
    robotarm.set_dofs_position(cur_q)

    for i, waypoint in enumerate(waypoints):
        q_real = waypoint
        q_real[-2:] = control_grip
        robotarm.control_dofs_position(q_real)
        for _ in range(10):
            scene.step()

def release(scene, robotarm, save_img):
    '''
    Args:
        scene: the scene object
        robotarm: the robotarm object
        save_img: the function to save images
    '''
    cur_q = robotarm.get_dofs_position()
    cur_q[-2:] = 0           # xarm
    robotarm.control_dofs_position(cur_q)
    for _ in range(50):
        scene.step()

def sample_points_on_mesh(vertices, faces, num_samples):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    sampled_points, face_indices = trimesh.sample.sample_surface(mesh, num_samples)
    return sampled_points

def my_grasp(scene, robotarm, obj, save_img, restriction_x = None, restriction_y = None, restriction_z = None, grasp_quat=None, filename=None, verbose=False):
    '''
    Args:
        scene: the scene object
        robotarm: the robotarm object
        obj: the object to grasp
        save_img: the function to save images
        restriction_x: the restriction on x axis
        restriction_y: the restriction on y axis
        verbose: whether to print debug information
    Returns:
        bool: whether the grasp is successful
    Grasp the object with the robot arm. First calculate a feasible grasping pose and the robot arm should move to the object, grasp it without lifting it up.
    '''

    ee_link, open_size = 'link_tcp', 0.0  # xarm
    if ee_link not in robotarm.links[-1].name:
        ee_link, open_size = 'hand', 0.04 # franka

    if filename is not None:
        dough_qpos = np.load(filename)
        for waypoint in dough_qpos:
            robotarm.control_dofs_position(waypoint)
            for _ in range(30):
                scene.step()
            if verbose:
                save_img('existing_grasping')
        for _ in range(60):
            scene.step()
        if verbose:
            save_img('existing_grasping')
        return
    
    com = None
    if isinstance(obj, MPMEntity):
        particles = obj.get_particles()
        org_obj_qpos = particles.mean(axis=0)
        xmn, xmx = particles[:, 0].min(), particles[:, 0].max()
        ymn, ymx = particles[:, 1].min(), particles[:, 1].max()
        zmn, zmx = particles[:, 2].min(), particles[:, 2].max()
        lis = []
        lis.append(np.array([[(xmn+xmx)/2, (ymn+ymx)/2, (zmn+zmx)/2]]))
        lis.append(particles[particles[:,0] == xmn])
        lis.append(particles[particles[:,0] == xmx])
        lis.append(particles[particles[:,1] == ymn])
        lis.append(particles[particles[:,1] == ymx])
        lis.append(particles[particles[:,2] == zmn])
        lis.append(particles[particles[:,2] == zmx])
        handle_pc = np.concatenate(lis, axis=0)
        com = particles.mean(axis=0)
        print('handle_pc', handle_pc)
    else:
        org_obj_qpos = obj.get_qpos().cpu().numpy()
        gemo_idx = 0
        vertices = obj.geoms[gemo_idx].get_verts().cpu().numpy()
        faces = np.array(obj.geoms[gemo_idx].init_faces)
        handle_pc = sample_points_on_mesh(vertices, faces, 10000)

    q_xarm = robotarm.get_dofs_position()
    def reset_obj():
        if isinstance(obj, MPMEntity):
            obj.set_pos(0, particles)
            robotarm.set_dofs_position(q_xarm)
            scene.step()
        else:
            obj.set_qpos(org_obj_qpos)
            robotarm.set_dofs_position(q_xarm)
            scene.step()


    if restriction_x is not None:
        handle_pc = handle_pc[handle_pc[:, 0] > restriction_x[0]]
        handle_pc = handle_pc[handle_pc[:, 0] < restriction_x[1]]
    if restriction_y is not None:
        handle_pc = handle_pc[handle_pc[:, 1] > restriction_y[0]]
        handle_pc = handle_pc[handle_pc[:, 1] < restriction_y[1]]
    if restriction_z is not None:
        handle_pc = handle_pc[handle_pc[:, 2] > restriction_z[0]]
        handle_pc = handle_pc[handle_pc[:, 2] < restriction_z[1]]
    else:
        mean_z = np.mean(handle_pc[:, 2])
        height, offset = mean_z, 10 #mean_z / 10
        # Get those points with z within the height range and store them in a new point cloud
        # why 20% near center, maybe tunable
        new_pc = []
        for point in handle_pc:
            if point[2] > height - offset and point[2] < height + offset:
                new_pc.append(point)
        new_pc = np.array(new_pc)
        handle_pc = new_pc

        print('mean_z', mean_z)
        print('offset', offset)
    
    print('handle_pc', handle_pc)

    # exit()
    if com is None:
        com = np.mean(handle_pc, axis=0)
    max_gripper_length = 0.043 * 2
    print('com', com)

    new_pc = handle_pc
    candidate_points = []
    for point in new_pc:
        print(point)
        # Get the farthest point from the current point
        dists = np.linalg.norm(new_pc - point, axis=1)
        max_dist = np.max(dists)
        if verbose:
            print("farthest", point, new_pc[np.argmax(dists)], max_dist)
        if max_dist < max_gripper_length:
            #Append the point pair to the candidate points
            candidate_points.append((point, new_pc[np.argmax(dists)]))
        
        # Get the closest point of the symmetry point
        sym_point = 2 * com - point
        dists = np.linalg.norm(new_pc - sym_point, axis=1)
        point_2 = new_pc[np.argmin(dists)]
        if verbose:
            print("symmetry", point, point_2, np.linalg.norm(point - point_2),np.linalg.norm(sym_point - point_2))
        if np.linalg.norm(point - point_2) < max_gripper_length and np.linalg.norm(sym_point - point_2) < 0.01:
            candidate_points.append((point, point_2))
        
        # Get the symmetry point of the current point by x axis
        sym_point_x = (2 * com[0] - point[0], point[1], point[2])
        dists = np.linalg.norm(new_pc - sym_point_x, axis=1)
        point_3 = new_pc[np.argmin(dists)]
        if verbose:
            print("symmetry_x", point, point_3, np.linalg.norm(point - point_3), np.linalg.norm(sym_point_x - point_3))
        if np.linalg.norm(point - point_3) < max_gripper_length and np.linalg.norm(sym_point_x - point_3) < 0.01:
            candidate_points.append((point, point_3))

        # Get the symmetry point of the current point by y axis
        sym_point_y = (point[0], 2 * com[1] - point[1], point[2])
        dists = np.linalg.norm(new_pc - sym_point_y, axis=1)
        point_4 = new_pc[np.argmin(dists)]
        if verbose:
            print("symmetry_y", point, point_4, np.linalg.norm(point - point_4), np.linalg.norm(sym_point_y - point_4))
        if np.linalg.norm(point - point_4) < max_gripper_length and np.linalg.norm(sym_point_y - point_4) < 0.01:
            candidate_points.append((point, point_4))

    if len(candidate_points) == 0:
        print("No candidate points found")
        return
    
    # Sort the candidate points by the distance of the mid point to the center of mass
    candidate_points = sorted(candidate_points, key=lambda x: np.linalg.norm((x[0] + x[1]) / 2 - com))
    print("Number of candidate points", len(candidate_points))
    # Pick top 10 candidate points
    sampled_points = candidate_points[:100]
    normal_list = [(0, 0, 1)]
    # normal_list = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, 1)]
    true_success = False
    grasping_waypoints = []
    print(len(candidate_points))
    print('sampled_points', sampled_points)
    print('com', com)
    # sampled_points[0] = (np.array([0.36,0,0]),np.array([0.36,0,0]))
    for point_pair in sampled_points:
        point1, point2 = point_pair
        mid_point = (point1 + point2) / 2
        if verbose:
            print("COM distance", np.linalg.norm(mid_point - com))
        
        for normal in normal_list:
            normal = np.array(normal)
            if grasp_quat is not None:
                # normal = np.array([0, 1, 0])
                normal = np.array([0, 0, 1])
                ik_orientation = grasp_quat
            else:
                target_orientation = align_gripper_z_with_normal(-normal, horizontal=False, randomize=False, flip=True).as_quat()
                target_orientation_gs = quat_xyzw_to_wxyz(target_orientation)
                ik_orientation = target_orientation_gs


            real_target_pos = mid_point - normal * 0.02
            if real_target_pos[2] < -0.006:
                real_target_pos[2] = -0.006
            mp_target_pos = mid_point + normal * 0.10

            ee_link = robotarm.get_link('link_tcp')
            ik_pos = mp_target_pos


            joint_limit_low, joint_limit_high = robotarm.q_limit[0], robotarm.q_limit[1]
            try_ik_times = 3
            random_initial_ik_joint = [np.random.uniform(joint_limit_low, joint_limit_high) for _ in range(try_ik_times)]

            ik_success = False
            for ik_try_idx in range(try_ik_times):
                if verbose:
                    print("ik_try_idx", ik_try_idx)
                reset_obj()
                q, err = robotarm.inverse_kinematics(
                    link=ee_link,
                    pos=mp_target_pos - normal * 0.05,
                    quat=ik_orientation,
                    return_error=True,
                    init_qpos=random_initial_ik_joint[ik_try_idx],
                    max_solver_iters=100,
                )

                q[-2:] = open_size
                q = q.cpu().numpy()
                q_clipped = np.clip(q, np.array(joint_limit_low), np.array(joint_limit_high))
                robotarm.set_dofs_position(q_clipped)

                collision_pairs = robotarm.detect_collision()
                scene.visualizer.update()
                collision = len(collision_pairs) > 0
                if verbose:
                    save_img(f'test_{ik_try_idx}_norm{normal[0].item()}{normal[1].item()}{normal[2].item()}_ikpos{ik_pos[0].item():.2f}{ik_pos[1].item():.2f}{ik_pos[2].item():.2f}_{collision}_{torch.norm(err).item():.2f}')
                
                if torch.norm(err).item() <= 0.1 and not collision:
                    robotarm.set_dofs_position(random_initial_ik_joint[ik_try_idx])
                    grasping_waypoints = [q_clipped]
                    q, err = robotarm.inverse_kinematics(
                        link=ee_link,
                        pos=ik_pos,
                        quat=ik_orientation,
                        return_error=True,
                        init_qpos=random_initial_ik_joint[ik_try_idx],
                        max_solver_iters=100,
                    )

                    q[-2:] = open_size
                    q = q.cpu().numpy()
                    q_clipped = np.clip(q, np.array(joint_limit_low), np.array(joint_limit_high))

                    waypoints = robotarm.plan_path(q_clipped, planner="BITstar")
                    save_img(f'waypoints{len(waypoints)}')

                    if len(waypoints) == 0:
                        continue

                    for _, waypoint in enumerate(waypoints):
                        robotarm.set_dofs_position(waypoint)
                        grasping_waypoints.append(waypoint.detach().cpu().numpy())
                        scene.step()
                    ik_success = True
                    break
                    

            if not ik_success:
                continue

            q, err = robotarm.inverse_kinematics(
                link=ee_link,
                pos=real_target_pos,
                quat=ik_orientation,
                return_error=True,
                init_qpos=q_clipped,
                max_solver_iters=100,
            )
            q[-2:] = open_size
            q = q.cpu().numpy()
            q_clipped = np.clip(q, np.array(joint_limit_low), np.array(joint_limit_high))

            robotarm.control_dofs_position(q_clipped)
            for _ in range(30):
                scene.step()
            if verbose:
                save_img('ik1')

            print("Error", np.linalg.norm(robotarm.get_link('link_tcp').get_pos().cpu() - real_target_pos), robotarm.get_link('link_tcp').get_pos().cpu(), real_target_pos)

            if np.linalg.norm(robotarm.get_link('link_tcp').get_pos().cpu() - real_target_pos) < 0.1: # 0.03
                
                grasping_waypoints.append(q_clipped)
                
                print("Grasp Success!")
                save_img('grasp')
                
                q, err = robotarm.inverse_kinematics(
                    link=ee_link,
                    pos=real_target_pos,
                    quat=ik_orientation,
                    return_error=True,
                    init_qpos=q_clipped,
                    max_solver_iters=100,
                )
                q[-2:] = 0.044
                q = q.cpu().numpy()
                q_clipped = np.clip(q, np.array(joint_limit_low), np.array(joint_limit_high))
                robotarm.control_dofs_position(q_clipped)
                grasping_waypoints.append(q_clipped)
                for _ in range(150):
                    scene.step()
                    if verbose and _ % 25 == 0:
                        save_img('ik2')
                true_success = True
                break
        
        if true_success:
            break
        break
        
    if not true_success:
        print("Grasp Failed")
        return False, None
    import uuid
    uid = uuid.uuid1()
    grasping_waypoints = np.array(grasping_waypoints)
    np.save("gw_{}.npy".format(uid), grasping_waypoints)
    return True, uid
