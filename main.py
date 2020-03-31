# -*- coding: utf-8 -*-

import argparse
from face_swap import face_swap as fs
import matplotlib.pyplot as plt
import cv2


if __name__=='__main__':

    code_parser = argparse.ArgumentParser()
    code_parser.add_argument('--predictor_path',metavar='PREDICTOR_PATH',default="./shape_predictor_68_face_landmarks.dat")
    code_parser.add_argument('--color_blure',metavar='COLOUR_CORRECT_BLUR_FRAC',default='0.6',type=float)
    code_parser.add_argument('--feather',metavar='FEATHER_AMOUNT',default='5',type=int)
    code_parser.add_argument('--first',metavar='first_img_path',default='img2_2.jpg')
    code_parser.add_argument('--second',metavar='second__img_path',default='house.jpg')
    code_parser.add_argument('--dest_name',metavar='result name',default='final.jpg')

    
    
    # for test
    '''
    args = code_parser.parse_args('--color_blure=0.6 --feather=4 --dest_name=final.jpg'.split())
    '''

    args = code_parser.parse_args()
   
    if args.feather % 2 == 0:
        args.feather +=1
        
    final_image = fs(PREDICTOR_PATH = args.predictor_path,
                     COLOUR_CORRECT_BLUR_FRAC = args.color_blure,
                     FEATHER_AMOUNT = args.feather,
                     img1_path = args.first,
                     img2_path = args.second)
    
    cv2.imwrite(args.dest_name,final_image)
  
 # downloaded and unzipped file
              