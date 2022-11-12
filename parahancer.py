# Input:
# parahancer <input_dir> <output_dir> <max_time_in_min> <brightness_factor> <sharpness_factor> <contrast_factor> <max_thread_count>
# parahancer E:\Desktop E:\Pictures 3 2 4 6 
# parahancer E:\Desktop E:\Pictures 5 1 2 3 2

import os
import numpy as np
import cv2
import argparse

MAX_THREAD_COUNT = 5

def is_dir_path(dir_path: str) -> bool:
   if os.path.isdir(dir_path): 
      return True
   raise NotADirectoryError(dir_path)

@staticmethod
def load_images(input_dir: str) -> np.ndarray:
   """
   Loads all images from the given file directory into a numpy array.
   """
   images = {}
   for filename in os.listdir(input_dir):
      img = cv2.imread(filename=os.path.join(input_dir, filename))
      if img is not None:
         # Saves each image with its original filename
         imgset = {img, filename}
         images.append(imgset)
   return images

@staticmethod
def save_images(image_list: np.ndarray, output_dir: str) -> None:
   """
   Saves all images in the list into the given file directory.
   """
   for imgset in image_list:
      cv2.imwrite(os.path.join(filename=imgset[1]), img=imgset[0])

def main(parser=argparse.ArgumentParser()):
   # Define command line arguments
   parser.add_argument(
     "input_dir", type=str, help="The file directory to take input images from."
   )
   parser.add_argument(
      "output_dir", type=str, help="The file directory to put output images into."
   )
   parser.add_argument(
      "max_time_in_min", type=int, help="The maximum total time that the image processing can run for."
   )
   parser.add_argument(
      "brightness_factor", type=int, help="Adjusts the brightness factor."
   )
   parser.add_argument(
      "sharpness_factor", type=int, help="Adjusts the sharpness factor."
   )
   parser.add_argument(
      "contrast_factor", type=int, help="Adjusts the contrast factor."
   )
   parser.add_argument(
      "max_thread_count", type=str, nargs="?", default=MAX_THREAD_COUNT, help="The maximum number of threads to use." # Defaults to MAX_THREAD_COUNT
   )
   # Parse the command line parameters upon executing the program
   args = parser.parse_args()
   print("\n")

   if is_dir_path(args.input_dir) and is_dir_path(args.output_dir):
      print("Input images from: %s" %args.input_dir)
      print("Output images will go to: %s" %args.input_dir)
      print("Maximum time (mins): %d" %args.max_time_in_min)
      print("Brightness: %d" %args.brightness_factor)
      print("Contrast: %d" %args.sharpness_factor)
      print("Sharpness: %d" %args.contrast_factor)
      print("Maximum number of threads: %d" %args.max_thread_count)
   else:
      print("Please double check that the given file directories exist.")

if __name__=="__main__":
   main()