import numpy
import cv2
import os
from matplotlib import pyplot as plt
from tqdm import tqdm

def ReadImageData(path_img):
  try:
    pic = cv2.imread(path_img)
    height, width, channels = pic.shape
    return [width,height,pic]
  except:
    return []

def Run(path_link,label):
    list_image_width_child = []
    list_image_height_child = []
    for img in os.listdir(path_link):
      data_img = ReadImageData(os.path.join(path_link,img))
      if len(data_img) == 0 : continue
      list_image_width_child.append(data_img[0])
      list_image_height_child.append(data_img[1])
    return [[list_image_width_child,list_image_height_child],label]
