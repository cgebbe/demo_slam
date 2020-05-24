import open3d as o3d
import numpy as np

# params
path_xyz = "output/xyz_in_orgCS.npy"

# load xyz
xyz = np.load(path_xyz)
mask_behind = xyz[2, :] < -10
mask_too_far = xyz[2, :] > 100
mask_too_hi = -xyz[1,:] > 3
mask_exlude = np.logical_or(mask_too_hi, np.logical_or(mask_behind, mask_too_far))
xyz_filtered = xyz[:, ~mask_exlude]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz_filtered.T)

# visualize pcd
o3d.visualization.draw_geometries([pcd],
                                  #zoom=0.3412,
                                  #front=[0.4257, -0.2125, -0.8795],
                                  #lookat=[2.6172, 2.0475, 1.532],
                                  #up=[-0.0694, -0.9768, 0.2024],
                                  )