import dlib
import numpy as np
import cv2
import matplotlib.pyplot as plt



LEFT_EYE_POINTS = list(range(42, 48))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_BROW_POINTS = list(range(17, 22))
NOSE_POINTS = list(range(27, 35))
MOUTH_POINTS = list(range(48, 61))
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]




def get_landmarks(img,PREDICTOR_PATH): # landmarks are main 68 dots
    detector = dlib.get_frontal_face_detector()
    
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    
    rects = detector(img , 1)
    
    if len(rects)> 1 :
        raise TooManyFaces
    if len(rects)== 0 :
        raise NoFaceDetected
    
    matrix = np.matrix([[p.x , p.y] for p in predictor(img , rects[0]).parts()])
    return matrix
    
    
def transformation_from_points(points1,points2): # points from both images
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    
    c1 = np.mean(points1 , axis=0)
    c2 = np.mean(points2 , axis=0)
    
    points1 -= c1
    points2 -= c2
    
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    
    
    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    
    result= np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])
    return result

def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im
    

def correct_colours(im1, im2, landmarks1,COLOUR_CORRECT_BLUR_FRAC):
    
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                              np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(np.uint8)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
                                                im2_blur.astype(np.float64))
    

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)
   
   
def get_face_mask(im, landmarks,FEATHER_AMOUNT):
    im = np.zeros(im.shape[:2], dtype=np.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im    
    
    
    
    
    
def face_swap(PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat", # downloaded and unzipped file
              COLOUR_CORRECT_BLUR_FRAC = 0.6,
              FEATHER_AMOUNT = 5,
              img1_path = 'img2_2.jpg',
              img2_path = 'house.jpg'):
    
    
    img1 = cv2.imread(img1_path)
    landmark1 = get_landmarks(img1,PREDICTOR_PATH)

    img2 = cv2.imread(img2_path)
    landmark2 = get_landmarks(img2,PREDICTOR_PATH)

    M = transformation_from_points(landmark1,landmark2)
    warpped_img2 = warp_im(img2,M,img1.shape) # warp img2 to img1
    corrected_color = correct_colours(img1,warpped_img2,landmark1,COLOUR_CORRECT_BLUR_FRAC) # correcting img2 color based on img1

    mask = get_face_mask(img2, landmark2,FEATHER_AMOUNT) # img2 mask (eyes,noise,mouth)
    warped_mask = warp_im(mask, M, img1.shape) # put mask of img2 on img1
    face_mask = get_face_mask(img1, landmark1,FEATHER_AMOUNT) # img1 mask
    combined_mask = np.max([face_mask, warped_mask],axis=0) # max of two masks

    output_im = img1 * (1.0 - combined_mask)+ corrected_color * combined_mask
    
    
    return output_im
