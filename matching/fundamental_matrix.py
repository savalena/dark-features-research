import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

fx = 609.824
fy = 614.837
cx = 326.547
cy = 232.0163
CAMERA_MATRIX = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
#
#
# def bf_matching(descriptors_1, descriptors_2):
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches = bf.match(descriptors_1, descriptors_2)
#     return matches
#
# def bf_knn_matching(des1, des2, lowe_ratio=0.8):
#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(des1, des2, k=2)
#     # Apply ratio test
#     good_knn = []
#     for m, n in matches:
#         if m.distance < lowe_ratio * n.distance:
#             good_knn.append(m)
#     return good_knn
#
# def matched_keypoints(kp1, kp2, matches):
#     matched_kp1 = []
#     matched_kp2 = []
#     for m in matches:
#         matched_kp2.append(kp2[m.trainIdx].pt)
#         matched_kp1.append(kp1[m.queryIdx].pt)
#     return np.array(matched_kp1), np.array(matched_kp1)
#
#
# def extract_kp_coords(kp):
#     kp_coords = []
#     for points in kp:
#         kp_coords.append(list(points.pt))
#     return np.asarray(kp_coords)

def bf_matching(descriptors_1, descriptors_2):
    # print("BF matching model: Implemented")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_1, descriptors_2)
    return matches

def bf_knn_matching(des1, des2, lowe_ratio=0.8):
    # print("KNN matching model: Implemented")
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < lowe_ratio * n.distance:
            good.append(m)
    return good

def matched_keypoints(kp1, kp2, matches):
    pts1, pts2 = [], []
    for m in matches:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    return pts1, pts2


def extract_kp_coords(kp):
    kp_coords = []
    for points in kp:
        kp_coords.append(list(points.pt))
    return np.asarray(kp_coords)

def apply_model_filtering(filtering, matched_kp1, matched_kp2):
    if filtering == "fm":
        # print(filtering, "chosen")
        fundamental_matrix, mask = cv2.findFundamentalMat(matched_kp1, matched_kp2, cv2.FM_RANSAC)
        return mask
    elif filtering == "em":

        # print(filtering, "chosen")
        em, mask = cv2.findEssentialMat(matched_kp1, matched_kp2)
        return mask
    elif filtering == "homo":

        # print(filtering, "chosen")
        h, mask =  cv2.findHomography(matched_kp1, matched_kp2)
        return mask
    elif filtering == "em_calib":

        # print(filtering, "chosen")
        em, mask = cv2.findEssentialMat(matched_kp1, matched_kp2, CAMERA_MATRIX, cv2.FM_RANSAC)
        return mask
    else:
        print("Wrong feature filtering")
        sys.exit()
#
# def apply_model_filtering(filtering, matched_kp1, matched_kp2):
#     if filtering == "fm":
#         print(filtering, "chosen")
#         fundamental_matrix, mask = cv2.findFundamentalMat(matched_kp1, matched_kp2, cv2.FM_RANSAC)
#         return mask
#     elif filtering == "em":
#
#         print(filtering, "chosen")
#         em, mask = cv2.findEssentialMat(matched_kp1, matched_kp2)
#         return mask
#     elif filtering == "homo":
#
#         print(filtering, "chosen")
#         h, mask =  cv2.findHomography(matched_kp1, matched_kp2)
#         return mask
#     elif filtering == "em_calib":
#         print(filtering, "chosen")
#         em, mask = cv2.findEssentialMat(matched_kp1, matched_kp2, CAMERA_MATRIX, cv2.FM_RANSAC)
#         return mask
#     else:
#         print("Wrong feature filtering")
#         sys.exit()

def choose_model_matching(matching):
    if matching == "bf":
        return bf_matching
    elif matching == "knn":
        return bf_knn_matching
    else:
        print("Wrong feature matching")
        sys.exit()

def find_good_matches(kp1, kp2, ds1, ds2, idx,  filtering="fm", matching="bf"):
    try:
        # data converstion
        kp1_coords, kp2_coords = extract_kp_coords(kp1), extract_kp_coords(kp2)
        matching_model = choose_model_matching(matching)
        if ds1 is not None and ds2 is not None:
            matches = matching_model(ds1, ds2) #TODO if zero matches were found
            matches = matching_model(ds1, ds2) #TODO if zero matches were found
            if len(matches) != 0:
            # data matching
                matched_kp1, matched_kp2 = matched_keypoints(kp1,
                                                             kp2,
                                                             matches)
                mask = apply_model_filtering(filtering, matched_kp1, matched_kp2)
                return np.count_nonzero(mask == 1), mask, True
            else:
        # except:
                return 0, [], False
        else:
            return 0, [], False
    except:
        # print("Error occured")
        return 0, [], False


def plot_good_matches(img1, img2, kp1, kp2,
                      ds1, ds2,
                      title, ax,
                      matching="bf", filtering="fm"):
    # try:
    kp1_coords, kp2_coords = extract_kp_coords(kp1), extract_kp_coords(kp2)
    matching_model = choose_model_matching(matching)
    matches = matching_model(ds1, ds2)
    # data matching
    matched_kp1_coods, matched_kp2_coods = matched_keypoints(kp1,
                                                 kp2,
                                                 matches)
    mask = apply_model_filtering(filtering, matched_kp1_coods, matched_kp2_coods)
    output_image = np.concatenate([img1, img2], axis=1)
    ax.imshow(output_image)
    ax.scatter(kp1_coords[:, 0], kp1_coords[:, 1], s=1)
    ax.scatter(kp2_coords[:, 0] + img1.shape[1], kp2_coords[:, 1], s=1)
    i = 0
    # print("matches len: ", len(matches))
    for points in matches:
        # print('mask value: ',mask[i, 0])
        color = "green" if mask[i, 0] else "red"
        i = i + 1
        ax.plot([kp1_coords[int(points.queryIdx), 0], kp2_coords[int(points.trainIdx), 0] + img1.shape[1]],
                 [kp1_coords[int(points.queryIdx), 1], kp2_coords[int(points.trainIdx), 1]], linewidth=1,
                 c=color)
    ax.set_title(title)
    ax.axis('off')
    # except:
    #     output_image = np.concatenate([img1, img2], axis=1)
    #     ax.imshow(output_image)
    #     ax.set_title(title)
    #     ax.axis('off')


# def choose_model_matching(matching):
#     if matching == "bf":
#         return bf_matching
#     elif matching == "knn":
#         return bf_knn_matching
#     else:
#         print("Wrong feature matching")
#         sys.exit()
#
# def find_good_matches(kp1, kp2, ds1, ds2, idx,  filtering="fm", matching="bf"):
#     # try:
#     # data converstion
#     kp1_coords, kp2_coords = extract_kp_coords(kp1), extract_kp_coords(kp2)
#     matching_model = choose_model_matching(matching)
#     matches = matching_model(ds1, ds2)
#
#     # data matching
#     matched_kp1, matched_kp2 = matched_keypoints(kp1,
#                                                  kp2,
#                                                  matches)
#     mask = apply_model_filtering(filtering, matched_kp1, matched_kp2)
#     return np.count_nonzero(mask == 1), mask, True
#     # except:
#     #     return 0, [], False
#
#
# def plot_good_matches(img1, img2, kp1, kp2,
#                       ds1, ds2,
#                       title, ax,
#                       matching="bf", filtering="fm"):
#     try:
#         kp1_coords, kp2_coords = extract_kp_coords(kp1), extract_kp_coords(kp2)
#         matching_model = choose_model_matching(matching)
#         matches = matching_model(ds1, ds2)
#         # data matching
#         matched_kp1_coods, matched_kp2_coods = matched_keypoints(kp1_coords,
#                                                      kp2_coords,
#                                                      matches)
#         mask = apply_model_filtering(filtering, matched_kp1_coods, matched_kp2_coods)
#         output_image = np.concatenate([img1, img2], axis=1)
#         ax.imshow(output_image)
#         ax.scatter(kp1_coords[:, 0], kp1_coords[:, 1], s=1)
#         ax.scatter(kp2_coords[:, 0] + img1.shape[1], kp2_coords[:, 1], s=1)
#         i = 0
#         print("matches len: ", len(matches))
#         for points in matches:
#             print('mask value: ',mask[i, 0])
#             color = "green" if mask[i, 0] else "red"
#             i = i + 1
#             ax.plot([kp1_coords[int(points.queryIdx), 0], kp2_coords[int(points.trainIdx), 0] + img1.shape[1]],
#                      [kp1_coords[int(points.queryIdx), 1], kp2_coords[int(points.trainIdx), 1]], linewidth=1,
#                      c=color)
#         ax.set_title(title)
#         ax.axis('off')
#     except:
#         output_image = np.concatenate([img1, img2], axis=1)
#         ax.imshow(output_image)
#         ax.set_title(title)
#         ax.axis('off')
#
