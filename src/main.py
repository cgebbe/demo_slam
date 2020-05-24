import cv2
import matplotlib.pyplot as plt
import matplotlib.colors
import mpl_toolkits.mplot3d
import os
import numpy as np
import tqdm


def get_camK():
    """ from file calib_cam_to_cam.txt """
    str = r'P_rect_02: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03'
    vals = np.array([float(x) for x in str.split()[1:]])
    vals = vals.reshape(3, 4)
    camK = vals[0:3, 0:3]
    return camK


def detect_features(gray_curr, should_plot=False):
    """ Detect Features using SIFT
    See https://www.pyimagesearch.com/2015/07/16/where-did-sift-and-surf-go-in-opencv-3/
    To install opencv contrib  https://stackoverflow.com/questions/42886286/how-to-get-opencv-contrib-module-in-anaconda
    """
    sift = cv2.xfeatures2d.SIFT_create()
    keyp_curr, desc_curr = sift.detectAndCompute(gray_curr, None)
    if should_plot:
        img2 = cv2.drawKeypoints(gray_curr, keyp_curr, img)
        plt.imshow(img2)
        plt.show()
    return keyp_curr, desc_curr


def match_features(gray_last, keyp_last, desc_last,
                   gray_curr, keyp_curr, desc_curr,
                   should_plot=False,
                   ):
    """ Match features
    See
        https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
        https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
        https://people.cs.umass.edu/~elm/Teaching/ppt/370/370_10_RANSAC.pptx.pdf

    RANSAC = RANdom SAmple Consensus:
        Do X times:
            1. Select four feature pairs (at random)
            2. Compute homography H (exact)
            3. Compute inliers where SSD(pi’, H pi) < ε
        4. Keep largest set of inliers
        5. Re-compute least-squares H estimate on all of the inliers
    """
    # find matches
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc_last, desc_curr, k=2)

    # filter matches via ratio test by Lowe
    matchesMask = [[0, 0] for i in range(len(matches))]
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.6 * n.distance:
            matchesMask[i] = [1, 0]

    # plot matches
    if should_plot:
        img3 = cv2.drawMatchesKnn(gray_last, keyp_last,
                                  gray_curr, keyp_curr,
                                  matches,
                                  outImg=None,
                                  # matchColor=(0, 255, 0),
                                  # singlePointColor=(255, 0, 0),
                                  matchesMask=matchesMask,
                                  flags=2,  # only display match lines, not feature points
                                  )
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.imshow(img3)
        plt.show()

    return matches, matchesMask


def estimate_pose(matches, matchesMask,
                  keyp_last, keyp_curr, camK,
                  ):
    """ Determine Essential matrix and thereby RT transformation from cam2->cam1+
    0 = x1^T F x2, where x1,x2 are homogeneous image coordinates
    E = K1^T F K2, where K1,K2 are camera matrices...
    see https://en.wikipedia.org/wiki/Fundamental_matrix_(computer_vision)
    See https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
    https://stackoverflow.com/questions/37810218/is-the-recoverpose-function-in-opencv-is-left-handed
    """
    # get 2D coords of keypoints
    pts_curr, pts_last = [], []
    for i, (m, n) in enumerate(matches):
        if matchesMask[i][0] == 1:
            pts_last.append(keyp_last[m.queryIdx].pt)
            pts_curr.append(keyp_curr[m.trainIdx].pt)
    pts_last = np.int32(pts_last)
    pts_curr = np.int32(pts_curr)

    # calc essential matrix and select only inliers
    E, mask = cv2.findEssentialMat(pts_last, pts_curr, camK,
                                   method=cv2.FM_LMEDS,
                                   )
    inliers_last = pts_last[mask.ravel() == 1]
    inliers_curr = pts_curr[mask.ravel() == 1]

    # extract R,t camera pose. Is curr->last
    retval, R, t, mask = cv2.recoverPose(E, inliers_last, inliers_curr, camK)
    RT_curr_to_last = np.diag(np.ones(4))
    RT_curr_to_last[0:3, 0:3] = R
    RT_curr_to_last[0:3, [3]] = t
    RT_lastCS_to_currCS = RT_curr_to_last
    return RT_lastCS_to_currCS, inliers_last, inliers_curr


def triangulate_points(camK,
                       RT_orgCS_to_lastCS, inliers_last,
                       RT_orgCS_to_currCS, inliers_curr,
                       ):
    """ Triangulate points
    """
    P_curr = np.matmul(camK, RT_orgCS_to_currCS[0:3, :])
    P_last = np.matmul(camK, RT_orgCS_to_lastCS[0:3, :])
    xyz1_in_orgCS = cv2.triangulatePoints(P_last.astype(np.float64),
                                          P_curr.astype(np.float64),
                                          inliers_last.T.astype(np.float64),
                                          # important! Otherwise strange error
                                          inliers_curr.T.astype(np.float64),
                                          )
    xyz1_in_orgCS[:, :] /= xyz1_in_orgCS[3, :]
    return xyz1_in_orgCS[0:3, :]


def plot_depthmap(xyz_in_orgCS, camK, RT_orgCS_to_currCS):
    # get uvs
    xyz_in_currCS = np.matmul(RT_orgCS_to_currCS[0:3, 0:3], xyz_in_orgCS)
    xyz_in_currCS += RT_orgCS_to_currCS[0:3, [3]]
    mask_in_front_of_cam = xyz_in_currCS[2, :] > 0
    xyz_in_currCS[:, ~mask_in_front_of_cam] = -1  # so that later invalidated!
    uv1s = np.matmul(camK, xyz_in_currCS)
    uv1s /= uv1s[2, :]
    uvs = np.round(uv1s[0:2, :]).astype(np.int)

    # get invalid mask
    height, width = gray_curr.shape
    mask_too_lo = uvs < 0
    mask_too_hi = np.empty_like(mask_too_lo)
    mask_too_hi[0, :] = uvs[0, :] >= width
    mask_too_hi[1, :] = uvs[1, :] >= height
    mask_invalid = np.logical_or(mask_too_hi, mask_too_lo)
    mask_invalid = np.any(mask_invalid, axis=0)
    print("{}/{} are outside of image".format(sum(mask_invalid), len(mask_invalid)))

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    ax.imshow(gray_curr, cmap='gray')
    artist = ax.scatter(uvs[0, ~mask_invalid], uvs[1, ~mask_invalid],
                        marker='.', s=50, alpha=0.5,  # markersize
                        c=xyz_in_currCS[2, ~mask_invalid],
                        cmap='viridis', vmin=0, vmax=50,
                        norm=matplotlib.colors.PowerNorm(gamma=0.5),
                        )
    cbar = plt.colorbar(artist, orientation="vertical", pad=0.05)
    cbar.set_label("z_CCS")
    ax.set_xlim([0, width])
    ax.set_ylim([height, 0])
    fig.tight_layout()
    fig.savefig("../output/depthmap_{:03d}.png".format(idx_filename))


# PARAMS
parent_path_images = r'../input'
should_plot_depthmap = True

# init stuff
camK = get_camK()
gray_last = None
RT_orgCS_to_currCS = np.diag(np.ones(4))
cam_positions = np.zeros((3, 0))
xyz_in_orgCS = np.zeros((3, 0))

# load images
filenames = os.listdir(parent_path_images)
filenames = [f for f in filenames if f.endswith('.png')]
filenames.sort()
nimages = len(filenames)
for idx_filename in tqdm.trange(0, 60, 3):

    # load image and detect features
    path = os.path.join(parent_path_images, filenames[idx_filename])
    img = cv2.imread(path)
    gray_curr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keyp_curr, desc_curr = detect_features(gray_curr)

    # =======================
    # Actual SLAM pipeline
    if gray_last is not None:
        matches, matchesMask = match_features(gray_last, keyp_last, desc_last,
                                              gray_curr, keyp_curr, desc_curr,
                                              )

        # estimate relative pose between last and current camera position
        RT_lastCS_to_currCS, inliers_last, inliers_curr = estimate_pose(matches, matchesMask,
                                                                        keyp_last, keyp_curr, camK,
                                                                        )
        RT_orgCS_to_currCS = np.matmul(RT_lastCS_to_currCS, RT_orgCS_to_lastCS)

        # Determine 3D coordinates of features via triangulation
        xyz_new_in_orgCS = triangulate_points(camK,
                                              RT_orgCS_to_lastCS, inliers_last,
                                              RT_orgCS_to_currCS, inliers_curr,
                                              )
        xyz_in_orgCS = np.hstack((xyz_in_orgCS, xyz_new_in_orgCS))

        # Perform bundle adjustment across last keypoints
        # TODO

        # Plot 3D points by projecting them onto current image
        if should_plot_depthmap:
            plot_depthmap(xyz_in_orgCS, camK, RT_orgCS_to_currCS)
    # =======================

    # store cam position
    RT_curr_to_org = RT_orgCS_to_currCS
    RT_org_to_curr = np.linalg.inv(RT_curr_to_org)
    t_org_to_curr = RT_org_to_curr[0:3, [3]]
    cam_positions = np.hstack((cam_positions, t_org_to_curr))

    # store relevant data from this keyframe
    RT_orgCS_to_lastCS = RT_orgCS_to_currCS
    gray_last = gray_curr
    keyp_last = keyp_curr
    desc_last = desc_curr
    plt.close('all')

""" AFTER PROCEESING ALL FRAMES...
"""
# export 3D points
np.save('../output/xyz_in_orgCS.npy', xyz_in_orgCS)

# plot 2D topview with points AND camera positions
mask_too_lo = xyz_in_orgCS[2, :] < -10
mask_too_hi = xyz_in_orgCS[2, :] > 100
mask_exlude = np.logical_or(mask_too_lo, mask_too_hi)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
artist = ax.scatter(xyz_in_orgCS[0, ~mask_exlude], xyz_in_orgCS[2, ~mask_exlude],
                    marker='.', s=1.0,  # marker size
                    c=-xyz_in_orgCS[1, ~mask_exlude],
                    vmin=0, vmax=2, cmap='viridis',
                    )
ax.plot(cam_positions[0, :], cam_positions[2, :], 'x', color='k', label='cam_positions')
cbar = plt.colorbar(artist, orientation="vertical", pad=0.05)
cbar.set_label("-y_CCS")
ax.set_xlabel('x_in_CCS')
ax.set_ylabel('z_in_CCS')
ax.set_aspect('equal')
fig.tight_layout()
# plt.show()
fig.savefig('../output/plot_topview.png')

print("=== Finished")
