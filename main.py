import cv2
import matplotlib.pyplot as plt
import os
import numpy as np


def get_camK():
    """ from file calib_cam_to_cam.txt """
    str = r'P_rect_02: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03'
    vals = np.array([float(x) for x in str.split()[1:]])
    vals = vals.reshape(3, 4)
    camK = vals[0:3, 0:3]
    return camK


# PARAMS
parent_path_images = r'd:\tmp\_homeoffice\KITTI\raw\2011_09_26\2011_09_26_drive_0011_sync\image_02\data'
camK = get_camK()
cam_positions = []

# init stuff
keyp_last = None
desc_last = None
RT_orgCS_to_currCS = np.diag(np.ones(4))

# load images
filenames = os.listdir(parent_path_images)
filenames = [f for f in filenames if f.endswith('.png')]
nimages = len(filenames)
for idx_filename in range(0, 60, 3):
    # load image
    path = os.path.join(parent_path_images, filenames[idx_filename])
    img = cv2.imread(path)
    gray_curr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    """ Detect Features using SIFT
    See https://www.pyimagesearch.com/2015/07/16/where-did-sift-and-surf-go-in-opencv-3/
    To install opencv contrib  https://stackoverflow.com/questions/42886286/how-to-get-opencv-contrib-module-in-anaconda
    """
    sift = cv2.xfeatures2d.SIFT_create()
    keyp_curr, desc_curr = sift.detectAndCompute(gray_curr, None)
    if False:
        img2 = cv2.drawKeypoints(gray_curr, keyp_curr, img)
        plt.imshow(img2)
        plt.show()

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
    if keyp_last is not None:
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

        if False:
            # plot matches
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
        RT_orgCS_to_currCS = np.matmul(RT_lastCS_to_currCS, RT_orgCS_to_lastCS)

        """ Triangulate points 
        """
        P_curr = np.matmul(camK, RT_orgCS_to_currCS[0:3, :])
        P_last = np.matmul(camK, RT_orgCS_to_lastCS[0:3, :])
        xyz1_in_orgCS = cv2.triangulatePoints(P_last, P_curr,
                                              inliers_last.T, inliers_curr.T,
                                              )
        print(xyz1_in_orgCS.shape)
        dummy = 0

    # store cam positions
    RT_curr_to_org = RT_orgCS_to_currCS
    RT_org_to_curr = np.linalg.inv(RT_curr_to_org)
    t_org_to_curr = RT_org_to_curr[0:3, 3]
    cam_positions.append(t_org_to_curr)
    print(t_org_to_curr)

    RT_orgCS_to_lastCS = RT_orgCS_to_currCS
    # P_last = P_curr
    gray_last = gray_curr
    keyp_last = keyp_curr
    desc_last = desc_curr

# HAVE PROCESSED ALL FRAMES
cam_positions = np.array(cam_positions)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(cam_positions[:, 0], cam_positions[:, 2], '.')
ax.set_aspect('equal')
ax.set_xlabel('x_in_CCS')
ax.set_ylabel('z_in_CCS')
plt.show()
fig.savefig('cam_positions.png')
print("=== Finished")
