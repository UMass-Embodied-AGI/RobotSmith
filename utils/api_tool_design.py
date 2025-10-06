import numpy as np
from skimage import measure
import trimesh
import open3d as o3d
import igl

def primitive(primitive_name, primitive_scale):
    """
    Create a 3D primitive mesh, centered at the origin.
    
    Args:
        primitive_name (str): One of {'cube', 'ball', 'cylinder'}.
        primitive_scale (list of float):
            - If cube, [sx, sy, sz]
            - If ball, [radius]
            - If cylinder, [radius, height]
    Returns:
        trimesh.Trimesh: The resulting mesh, centered at (0, 0, 0).
    """
    primitive_name = primitive_name.lower()
    
    if primitive_name == 'cube':
        # Expecting [sx, sy, sz]
        if len(primitive_scale) != 3:
            raise ValueError("cube requires 3 scale parameters: [sx, sy, sz].")
        # box() in trimesh is centered at the origin with extents=.
        mesh = trimesh.creation.box(extents=primitive_scale)
    
    elif primitive_name == 'ball':
        # Expecting [radius]
        if len(primitive_scale) != 1:
            raise ValueError("ball requires 1 scale parameter: [radius].")
        radius = primitive_scale[0]
        # icosphere or uv_sphere are common ways to create a sphere in trimesh
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=radius)
    
    elif primitive_name == 'cylinder':
        # Expecting [radius, height]
        if len(primitive_scale) != 2:
            raise ValueError("cylinder requires 2 scale parameters: [radius, height].")
        r, h = primitive_scale
        # cylinder() in trimesh is centered on the origin, along the Z axis
        mesh = trimesh.creation.cylinder(radius=r, height=h, sections=32)
    
    else:
        raise ValueError(f"Unsupported primitive name: {primitive_name}. "
                         "Choose from 'cube', 'ball', or 'cylinder'.")
    
    return mesh


def rotate_to_align(mesh1):
    """
    Rotate and translate the mesh so that:
      - Its bounding box is centered at the origin (0,0,0).
      - The largest dimension of the bounding box aligns with +X,
      - The second-largest aligns with +Y,
      - The smallest aligns with +Z.
    The mesh remains axis-aligned and is returned in-place.

    Args:
        mesh (trimesh.Trimesh): The input mesh (modified in-place).

    Returns:
        trimesh.Trimesh: The rotated+translated mesh, for convenience.
    """
    # 1) Translate mesh so its bounding-box center is at origin.
    #    (We do this first so that when we permute axes, the shape stays around 0,0,0.)
    mesh = mesh1.copy()

    bbox_center = mesh.bounding_box.centroid
    mesh.apply_translation(-bbox_center)

    # 2) Get the bounding box extents in the current orientation.
    #    e.g. (Ex, Ey, Ez)
    ex, ey, ez = mesh.bounding_box.extents
    dims = np.array([ex, ey, ez])

    # 3) Figure out which dimension is largest, second, smallest.
    #    We'll store the indices in descending order of dimension size.
    #    Example: if ex >= ey >= ez, then desc_order = [0, 1, 2].
    desc_order = np.argsort(-dims)  # negative so largest->smallest

    # desc_order[0] = index of largest dimension
    # desc_order[1] = index of second-largest
    # desc_order[2] = index of smallest

    # 4) Build a permutation matrix that sends:
    #       old axis desc_order[0] -> new X,
    #       old axis desc_order[1] -> new Y,
    #       old axis desc_order[2] -> new Z.
    #    For instance, if desc_order = [1,0,2], that means:
    #      old Y -> new X,
    #      old X -> new Y,
    #      old Z -> new Z,
    #    which can be represented by:
    #      [[0, 1, 0],
    #       [1, 0, 0],
    #       [0, 0, 1]]

    perm = np.zeros((3,3), dtype=float)
    # Each row picks out which old axis corresponds to new X, Y, Z
    # row = new-axis index, col = old-axis index
    perm[0, desc_order[0]] = 1.0  # new X from old axis = largest dimension
    perm[1, desc_order[1]] = 1.0  # new Y from old axis = second largest
    perm[2, desc_order[2]] = 1.0  # new Z from old axis = smallest

    # 5) Ensure the permutation matrix is a right-handed transform.
    #    If the determinant is negative, flip the new Z axis (for example).
    if np.linalg.det(perm) < 0:
        perm[2, :] = -perm[2, :]

    # 6) Construct a 4x4 transformation from this 3x3 rotation.
    transform_4x4 = np.eye(4)
    transform_4x4[:3, :3] = perm

    # 7) Apply the permutation. The mesh should now have largest bounding-box
    #    dimension along +X, second largest along +Y, and smallest along +Z.
    mesh.apply_transform(transform_4x4)

    return mesh

def get_axis_align_bounding_box(mesh):
    """
    Return the axis-aligned bounding box of the mesh as (min_x, min_y, min_z, max_x, max_y, max_z).

    Args:
        mesh (trimesh.Trimesh): The mesh.

    Returns:
        tuple: (min_x, min_y, min_z, max_x, max_y, max_z)
    """
    # mesh.bounds returns an array of shape (2, 3):
    #   [[min_x, min_y, min_z],
    #    [max_x, max_y, max_z]]
    bounds = mesh.bounds
    (min_x, min_y, min_z), (max_x, max_y, max_z) = bounds
    return (min_x, min_y, min_z, max_x, max_y, max_z)

def get_volume(mesh):
    """
    Return the volume of the mesh in cubic units.

    Args:
        mesh (trimesh.Trimesh): A closed, manifold mesh.

    Returns:
        float: The volume of the mesh.
    """
    # Trimesh can compute volume if the mesh is watertight (closed).
    # If mesh isn't closed, mesh.volume might be zero or inaccurate.
    # For safety, you might want to call mesh.force_watertight() or use mesh.convex_hull.volume,
    # but here we assume it's already closed.
    return mesh.volume

def rescale(mesh, ratio):
    """
    Uniformly scale the mesh by the given ratio about the origin (0, 0, 0).

    Args:
        mesh (trimesh.Trimesh): The mesh to scale (modified in-place).
        ratio (float): The scale factor to apply.

    Returns:
        trimesh.Trimesh: The scaled mesh (for convenience).
    """
    # Create a scaling matrix
    new_mesh = mesh.copy()
    new_mesh.apply_scale(ratio)
    
    return new_mesh

def move(mesh, offset):
    """
    Translate the mesh by the given offset (x, y, z).

    Args:
        mesh (trimesh.Trimesh): The mesh to move in-place.
        offset (tuple or list of float): The translation offset (dx, dy, dz).

    Returns:
        trimesh.Trimesh: The transformed mesh (for convenience).
    """
    new_mesh = mesh.copy()
    offset = np.asarray(offset, dtype=float)
    
    # Create a translation matrix
    transform = np.eye(4)
    transform[:3, 3] = offset
    
    # Apply in-place
    new_mesh.apply_transform(transform)
    
    return new_mesh

def empty_grid():
    """
    Create an empty 256x256x256 boolean occupancy grid from -0.5 to +0.5 in each axis.

    Returns:
        dict: A dictionary containing:
            - 'data': np.ndarray of shape (256, 256, 256), dtype=bool (all False initially).
            - 'res':  integer (256).
            - 'min_bound': np.array([-0.5, -0.5, -0.5]).
            - 'max_bound': np.array([0.5, 0.5, 0.5]).
    """
    grid = {}
    grid["res"] = 256
    grid["data"] = np.zeros((256, 256, 256), dtype=bool)  # All empty at first
    grid["min_bound"] = np.array([-0.2, -0.2, -0.2])
    grid["max_bound"] = np.array([ 0.2,  0.2,  0.2])
    return grid


def add_mesh(grid, mesh):
    """
    Convert 'mesh' into a volume of occupied voxels using an SDF (signed-distance) test,
    then mark those voxels as True in 'grid'.

    Args:
        grid (dict): The grid dictionary from empty_grid().
        mesh (trimesh.Trimesh): A triangular mesh (assumed to fit in [-0.5, 0.5]^3).
    
    Returns:
        dict: The updated grid, same reference as input.
    """
    # Unpack grid data
    res = grid["res"]
    data = grid["data"]
    vmin = grid["min_bound"]
    vmax = grid["max_bound"]
    
    # Prepare query points: center of each voxel in (x,y,z)
    # shape: (res^3, 3)
    xs = np.linspace(vmin[0], vmax[0], res, endpoint=False) + (vmax[0]-vmin[0])/(2*res)
    ys = np.linspace(vmin[1], vmax[1], res, endpoint=False) + (vmax[1]-vmin[1])/(2*res)
    zs = np.linspace(vmin[2], vmax[2], res, endpoint=False) + (vmax[2]-vmin[2])/(2*res)

    # Create a full 3D grid of points
    # XX.shape = (res, res, res), etc.
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')
    points = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    
    # Convert the mesh into arrays for libigl
    # libigl signed_distance expects:
    #   points:    (#P,3) array
    #   V:         (#V,3) array of mesh vertices
    #   F:         (#F,3) array of mesh faces (integers)
    V = mesh.vertices
    F = mesh.faces
    
    # Compute signed distance with libigl
    # sdf_values < 0  => inside
    sdf_values, _, _ = igl.signed_distance(points, V, F)
    
    # Reshape back to (res, res, res)
    sdf_3d = sdf_values.reshape((res, res, res))
    
    # Occupied if sdf < 0
    inside = (sdf_3d < 0)
    
    # Combine with existing occupancy via logical OR
    data |= inside
    
    return grid


def sub_mesh(grid, mesh):
    """
    Convert 'mesh' into a volume using an SDF, then set those voxels to False
    (subtract from the grid).
    """
    res = grid["res"]
    data = grid["data"]
    vmin = grid["min_bound"]
    vmax = grid["max_bound"]
    
    xs = np.linspace(vmin[0], vmax[0], res, endpoint=False) + (vmax[0]-vmin[0])/(2*res)
    ys = np.linspace(vmin[1], vmax[1], res, endpoint=False) + (vmax[1]-vmin[1])/(2*res)
    zs = np.linspace(vmin[2], vmax[2], res, endpoint=False) + (vmax[2]-vmin[2])/(2*res)

    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')
    points = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])

    V = mesh.vertices
    F = mesh.faces
    
    sdf_values, _, _ = igl.signed_distance(points, V, F)
    sdf_3d = sdf_values.reshape((res, res, res))
    inside = (sdf_3d < 0)
    
    # Subtraction => any inside voxel becomes False
    data[inside] = False
    
    return grid

def cut_grid(grid):
    """
    Return two new grids of the same resolution (256x256x256):
      - grid_bottom: occupies only z < 0
      - grid_up: occupies only z >= 0
    by zeroing out the complementary region in each grid.

    Each returned grid has:
      - 'data': a (256,256,256) boolean array
      - same 'min_bound' and 'max_bound' as the original
      - same 'res' as the original

    Args:
        grid (dict): Must have keys 'data', 'res', 'min_bound', 'max_bound'.
            'data' is a 3D boolean array: (256,256,256).

    Returns:
        (dict, dict): (grid_up, grid_bottom)
    """
    # Unpack the original grid
    res = grid["res"]
    data = grid["data"]
    vmin = grid["min_bound"]
    vmax = grid["max_bound"]

    # Create full copies for up & bottom
    data_up = data.copy()
    data_bottom = data.copy()

    # z_cut_idx is the voxel index corresponding to z=0 in our [-0.5, 0.5] range
    # If the range is exactly 1.0 in z, then the midpoint is 0.5 * res => 128
    z_cut_idx = res // 2

    # Clear out the 'bottom half' in data_up => everything below z=0
    # i.e. for indices 0..(z_cut_idx-1), set to False
    data_up[:, :, :z_cut_idx] = False

    # Clear out the 'upper half' in data_bottom => everything above z=0
    # i.e. for indices z_cut_idx..(res-1), set to False
    data_bottom[:, :, z_cut_idx:] = False

    # Build the two new grid dicts
    grid_up = {
        "res": res,
        "data": data_up,
        "min_bound": vmin.copy(),
        "max_bound": vmax.copy()
    }
    grid_bottom = {
        "res": res,
        "data": data_bottom,
        "min_bound": vmin.copy(),
        "max_bound": vmax.copy()
    }

    return grid_up, grid_bottom

def grid_to_mesh(grid, do_simplify=True, target_num_faces=3000):
    """
    Convert a 3D occupancy grid into a surface mesh using Marching Cubes.
    Optionally simplify the mesh using Open3D's quadric decimation.

    Args:
        grid (dict): A dictionary with keys:
            - 'data': (256,256,256) boolean array (True = occupied).
            - 'res': int, resolution (e.g. 256).
            - 'min_bound': np.array([x_min, y_min, z_min]).
            - 'max_bound': np.array([x_max, y_max, z_max]).
        do_simplify (bool): Whether to perform mesh simplification (default True).
        target_num_faces (int): If simplifying, the target number of faces.

    Returns:
        trimesh.Trimesh: The extracted (and optionally simplified) mesh. 
            If the grid is empty or no surface is found, faces might be empty.
    """
    data = grid["data"]    # Boolean volume, shape = (256,256,256)
    res = grid["res"]      # Typically 256
    min_b = grid["min_bound"]
    max_b = grid["max_bound"]

    # 1) Convert boolean volume to float so marching_cubes can interpret it.
    #    Values near 1 => inside, near 0 => outside.
    volume = data.astype(np.float32)

    # 2) Extract an isosurface at 0.5
    #    verts_voxel -> Nx3 array in "voxel space" [0..res-1]
    #    faces       -> Mx3 indices
    #    normals     -> Nx3 normal vectors
    #    values      -> Nx1 (unused here)
    verts_voxel, faces, normals, _ = measure.marching_cubes(volume, level=0.5)

    if len(faces) == 0:
        # Empty or uniform grid => no surface
        return trimesh.Trimesh(vertices=[], faces=[])

    # 3) Map voxel coordinates to real-world coordinates
    box_size = max_b - min_b  # e.g. [1.0, 1.0, 1.0] if bounding box is [-0.5..0.5]^3
    scale = box_size / float(res) 
    verts_world = verts_voxel * scale + min_b  # Shift + scale

    # 4) Build a Trimesh
    mesh = trimesh.Trimesh(vertices=verts_world, faces=faces[:, ::-1], vertex_normals=normals)

    if do_simplify and len(mesh.faces) > 0:
        # 5) Convert Trimesh -> Open3D, simplify, then convert back
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

        # Optional: compute vertex normals so Open3D knows how to handle them
        o3d_mesh.compute_vertex_normals()

        # Perform quadric decimation
        simplified_o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=target_num_faces)

        # Convert back to Trimesh
        simplified_vertices = np.asarray(simplified_o3d_mesh.vertices)
        simplified_faces = np.asarray(simplified_o3d_mesh.triangles)

        mesh = trimesh.Trimesh(vertices=simplified_vertices, faces=simplified_faces)

    return mesh

import numpy as np

# -------------
# 1) Assume these API functions are already implemented somewhere:
#    - generate_3d(name, scale)  -> Trimesh
#    - rotate_to_align(mesh)     -> Trimesh
#    - move(mesh, offset)        -> Trimesh
#    - rescale(mesh, ratio)      -> Trimesh
#    - get_volume(mesh)          -> float
#    - get_axis_align_bounding_box(mesh) -> (minx, miny, minz, maxx, maxy, maxz)
#    - empty_grid()              -> dict with 'data', 'res', 'min_bound', 'max_bound'
#    - add_mesh(grid, mesh)      -> dict
#    - sub_mesh(grid, mesh)      -> dict
#    - cut_grid(grid)            -> (grid_up, grid_bottom)  # each 256x256x256
#    - grid_to_mesh(grid, ...)   -> Trimesh
#
# 2) We’ll just define a skeleton call for generate_3d here, 
#    assuming it returns a dummy Trimesh. In practice, you'd have 
#    a text-to-3D pipeline that returns a gummy bear shape.

# from my_api import (
#     generate_3d, rotate_to_align, move, rescale, get_volume,
#     get_axis_align_bounding_box, empty_grid, add_mesh, sub_mesh,
#     cut_grid, grid_to_mesh
# )

import trimesh
import os
project_path = os.path.dirname(os.path.join(os.path.abspath(__file__), '..'))
meshy_api_key = ''
with open(os.path.join(project_path, 'meshy_api_key.txt')) as fi:
    meshy_api_key = fi.readlines()[0]

def generate_3d(name):
    import io
    import json
    import requests
    import subprocess

    cmd1 =   "curl https://api.meshy.ai/openapi/v2/text-to-3d " + \
            "  -H \'Authorization: Bearer {}\' ".format(meshy_api_key) + \
            "  -H \'Content-Type: application/json\' " + \
            "  -d \'{\n" + \
            "  \"mode\": \"preview\",\n" + \
            "  \"prompt\": \"{}\",\n".format(name) + \
            "  \"art_style\": \"realistic\",\n" + \
            "  \"should_remesh\": true\n" + \
            "}\'\n" 
    result = subprocess.run(cmd1, shell=True, capture_output=True, text=True)
    task_id = json.loads(result.stdout)['result']
    # task_id = "0195c83d-6d69-7826-ac1f-e97aa7ba7541"
    
    cmd2 = "curl https://api.meshy.ai/openapi/v2/text-to-3d/{} ".format(task_id) + \
            "-H \"Authorization: Bearer {}\" ".format(meshy_api_key)
    result = subprocess.run(cmd2, shell=True, capture_output=True, text=True)
    output = json.loads(result.stdout)
    while not output['status'] == 'SUCCEEDED':
        print("Waiting for meshy to finish...")
        result = subprocess.run(cmd2, shell=True, capture_output=True, text=True)
        output = json.loads(result.stdout)
    mesh_file = output['model_urls']['obj']
    
    response = requests.get(mesh_file)
    mesh = trimesh.load(io.BytesIO(response.content), file_type='obj')
    
    return mesh