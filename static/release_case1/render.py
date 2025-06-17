import os
import re
import json 
import math 
import trimesh
import numpy as np
import open3d as o3d

project_path = '/home/xhrlyb/Projects/FuncAny'

tid = 3

def look_at(cam_pos, target=np.array([0, 0, 0]), up=np.array([0, 0, 1])):
    forward = (target - cam_pos)
    forward /= np.linalg.norm(forward)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    R = np.eye(4)
    R[:3, :3] = np.stack([right, up, -forward], axis=1)
    R[:3, 3] = -cam_pos
    return np.linalg.inv(R)

def render_and_save(mesh_path, output_folder, num_views=10):
    os.makedirs(output_folder, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    mesh.translate(-mesh.get_center())        
    vertices = np.asarray(mesh.vertices)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=640, height=480)
    vis.add_geometry(mesh)
    opt = vis.get_render_option()
    opt.light_on = True
    opt.background_color = np.array([1, 1, 1]) 

    for i in range(num_views):
        # Random camera position on a sphere
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        radius = max(abs(vertices.min()), vertices.max()) * 3
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        cam_pos = np.array([x, y, z])

        # Compute extrinsic
        extrinsic = look_at(cam_pos)

        # Set the view
        ctr = vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        param.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(param)

        vis.poll_events()
        vis.update_renderer()

        # Save image
        img = vis.capture_screen_float_buffer(False)
        img = (255 * np.asarray(img)).astype(np.uint8)
        o3d.io.write_image(f"{output_folder}/{tid}_{i}.png", o3d.geometry.Image(img))

    vis.destroy_window()


mesh_path = f'{tid}.obj'
json_filename = f'proposer{tid}.json'
output_folder = '.'

render_and_save(mesh_path, output_folder, num_views=5)