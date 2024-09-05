import numpy as np
import os
import pywavefront
from scipy.spatial import cKDTree
import open3d as o3d
from plyfile import PlyData, PlyElement

def chamfer_dist(pcd_src, pcd_dst, norm='L2', opacities=None):
    tree_dst = cKDTree(np.asarray(pcd_dst.points))
    dists_src2dst, inds = tree_dst.query(np.asarray(pcd_src.points), k=1)
    tree_src = cKDTree(np.asarray(pcd_src.points))
    dists_dst2src, inds = tree_src.query(np.asarray(pcd_dst.points), k=1)
    # dists_dst2src = dists_dst2src * opacities.squeeze()
    if norm == 'L2':
        return np.mean(dists_src2dst**2) + np.mean(dists_dst2src**2)
        # return np.mean(dists_dst2src**2)
    elif norm == 'L1':
        return np.mean(np.abs(dists_src2dst)) + np.mean(np.abs(dists_dst2src))
        # return np.mean(np.abs(dists_src2dst))
        # return np.mean(np.abs(dists_dst2src))
    else:
        raise NotImplementedError()
    
def f1_score(pcd_src, pcd_dst, threshold=0.05, opacities=None):
    res_src2dst = o3d.pipelines.registration.evaluate_registration(
        pcd_src, pcd_dst, threshold)
    precision = res_src2dst.fitness
    res_dst2src = o3d.pipelines.registration.evaluate_registration(
        pcd_dst, pcd_src, threshold)
    recall = res_dst2src.fitness
    F_score = 2 * (precision * recall) / (precision + recall)
    return F_score, precision, recall

def remove_usemtl(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    with open(filename, 'w') as file:
        for line in lines:
            if not line.startswith('usemtl') and not line.startswith('s') and not line.startswith('l'):
                file.write(line)

filename = "livingroom"

# Set the directory path
dir_path = "data/model/meshes/" + filename + "_meshes"

# WARNING: FOR OBJ FILES ONLY
# Get a list of all .obj files in the directory
# obj_files = [f for f in os.listdir(dir_path) if f.endswith('.obj')]
obj_files = ["/home/omkar/Desktop/Projetcs/gaussian-splatting-with-depth/RGBS_data_2/27-march-readings/set1/Mar27at11-05 AM-poly/textured.obj"]

# Initialize an empty list to store all vertex data
all_vertex_data = []

# For each .obj file in the directory
for obj_file in obj_files:
    # Read the .obj file and extract the vertex data
    remove_usemtl(os.path.join(dir_path, obj_file))
    vertex_data = pywavefront.Wavefront(os.path.join(dir_path, obj_file), collect_faces=True, parse=True, create_materials=False).vertices
    # Append the vertex data to the list
    all_vertex_data.extend(vertex_data)

# Convert the list of all vertex data to a numpy array
all_vertex_data = np.array(all_vertex_data)

print(all_vertex_data.shape)
# WARNING: FOR OBJ FILES ONLY

# WARNING: FOR PLY FILES ONLY
# # Get a list of all .ply files in the directory
# ply_files = [f for f in os.listdir(dir_path) if f.endswith('.ply')]

# # Initialize an empty list to store all point clouds
# all_point_clouds = []

# # For each .ply file in the directory
# for ply_file in ply_files:
#     # Read the .ply file and extract the point cloud
#     pcd = o3d.io.read_point_cloud(os.path.join(dir_path, ply_file))
#     # Append the point cloud to the list
#     all_point_clouds.append(pcd)

# # Concatenate all point clouds into a single one
# all_points = o3d.geometry.PointCloud()
# for pcd in all_point_clouds:
#     all_points += pcd

# # Convert the point cloud to a numpy array
# all_vertex_data = np.asarray(all_points.points)

# print(all_vertex_data.shape)
# WARNING: FOR PLY FILES ONLY


# # Load the .ply file
# for i in ["rgb", "echo", "fls", "rgb_colmap"]:
#     print(i)
# path = "data/model/point_cloud/" + filename + "/" + i + ".ply"
path = "/home/omkar/Desktop/Projetcs/gaussian-splatting-with-depth/RGBS_data_2/27-march-readings/set1/cropped_our_for_metrices_2.ply"
plydata = PlyData.read(path)

xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"])),  axis=1)
opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
opacities = 1 / (1 + np.exp(-opacities))
# delete points with low opacity
# opacities = np.squeeze(opacities > 0.1)
# xyz = xyz[opacities, :]
print(xyz.shape)

# calculate the chamfer distance
pcd_src = o3d.geometry.PointCloud()
pcd_src.points = o3d.utility.Vector3dVector(all_vertex_data)
pcd_dst = o3d.geometry.PointCloud()
pcd_dst.points = o3d.utility.Vector3dVector(xyz)
# chamfer_dist = chamfer_dist(pcd_src, pcd_dst)
chamfer_dist_l2 = chamfer_dist(pcd_src, pcd_dst, norm='L2', opacities=opacities)
chamfer_dist_l1 = chamfer_dist(pcd_src, pcd_dst, norm='L1', opacities=opacities)
print("chamfer:", chamfer_dist_l2)

# calculate the F1 score
f1score, precision, recall = f1_score(pcd_src, pcd_dst, threshold=0.2)
print("f1:", f1score)
print("precision:", precision)
print("recall:", recall)
print()