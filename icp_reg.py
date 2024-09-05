import open3d as o3d
import numpy as np
import copy
import os
import pywavefront
from scipy.spatial import cKDTree
import open3d as o3d
from plyfile import PlyData, PlyElement
np.random.seed(42)

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


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

# def remove_usemtl(filename):
#     with open(filename, 'r') as file:
#         lines = file.readlines()
#     with open(filename, 'w') as file:
#         for line in lines:
#             if not line.startswith('usemtl') and not line.startswith('s') and not line.startswith('l'):
#                 file.write(line)

if __name__ == "__main__":
    # Load the mesh and convert it to a point cloud
    mesh = o3d.io.read_triangle_mesh("/home/omkar/Desktop/Projetcs/gaussian-splatting-with-depth/rebuttal_data/Geometric_experiments/Cropped_mesh.obj")
    source = mesh.sample_points_poisson_disk(1000)  # Adjust the number of points as needed

    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Load the point cloud
    target = o3d.io.read_point_cloud("/home/omkar/Desktop/Projetcs/gaussian-splatting-with-depth/rebuttal_data/Geometric_experiments/rebuttal_pcd_z_splat_90.ply")

    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # get the target and transformed source as numpy arrays
    pcd_dst = np.asarray(target.points)
    pcd_src = np.asarray(source.points)

    ##############################################################################################################

    threshold = 1.0
    # trans_init = np.asarray([[0.8652033805847168, -0.030802462249994278, 0.5725960731506348, -1.4800001382827759],
    #                          [-0.6985923647880554, -0.04147971794009209, 0.7090765833854675, -1.200000286102295],
    #                          [0.0016728243790566921, -1.3910713195800781, -0.03382261097431183, 0.4999999403953552], 
    #                          [0.0, 0.0, 0.0, 1.0]])
    trans_init = np.asarray([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0], 
                                [0, 0, 0, 1]])
    # draw_registration_result(source, target, trans_init)
    # print("Initial alignment")
    # evaluation = o3d.pipelines.registration.evaluate_registration(source, target,
    #                                                     threshold, trans_init)
    # print(evaluation)

    # print("Apply point-to-point ICP")
    # reg_p2p = o3d.pipelines.registration.registration_icp(
    #     source, target, threshold, trans_init,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint())
    # print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    # print("")
    # draw_registration_result(source, target, reg_p2p.transformation)

    print("Apply point-to-plane ICP")
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(), criteria=criteria)
    # print(reg_p2l)
    # print("Transformation is:")
    # print(reg_p2l.transformation)
    # print("")
    # draw_registration_result(source, target, reg_p2l.transformation)

    # get the target and transformed source as numpy arrays
    target_points = np.asarray(target.points)
    source_points = np.asarray(source.points)

    # get the transformation matrix
    transformation = reg_p2l.transformation
    transformation = np.array(transformation).reshape((4, 4))

    # apply the transformation matrix to the source points
    source_points_transformed = np.dot(transformation[:3, :3], source_points.T).T + transformation[:3, 3]

    # calculate the chamfer distance
    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(source_points_transformed)

    pcd_dst = o3d.geometry.PointCloud()
    pcd_dst.points = o3d.utility.Vector3dVector(target_points)

    ##############################################################################################################

    chamfer_dist_l2 = chamfer_dist(pcd_src, pcd_dst, norm='L2')
    chamfer_dist_l1 = chamfer_dist(pcd_src, pcd_dst, norm='L1')

    print("chamfer l2:", chamfer_dist_l2)
    print("chamfer l1:", chamfer_dist_l1)

    # calculate the F1 score
    f1score, precision, recall = f1_score(pcd_src, pcd_dst, threshold=0.2)
    print("f1:", f1score)
    print("precision:", precision)
    print("recall:", recall)

