import cv2

def orb_feature_exctraction(img):
    orb = cv2.ORB_create(nfeatures=2000)
    kp, descr = orb.detectAndCompute(img, None)
    orb_img = cv2.drawKeypoints(img, kp, None)
    return orb_img, kp, descr