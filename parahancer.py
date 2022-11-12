# Input:
# parahancer <input_dir> <output_dir> <max_time_in_min> <brightness_factor> <sharpness_factor> <contrast_factor> <max_thread_count>
# parahancer E:\Desktop E:\Pictures 3 2 4 6 
# parahancer E:\Desktop E:\Pictures 5 1 2 3 2

import os
import numpy as np
import cv2
import argparse

MAX_THREAD_COUNT = 5
PREVIEW_IMAGE_WIDTH = 800
TRACKBAR_BR = "Brightness"
TRACKBAR_SH = "Brightness"
TRACKBAR_CT = "Brightness"

# Image manipulation


   
# Image loading and saving
def show_image(imgset: np.ndarray):
   """
   Displays the given image in a new window.
   """
   img, filename = imgset.copy()[0], imgset.copy()[1]
   height, width = img.shape[0], img.shape[1]
   new_width = PREVIEW_IMAGE_WIDTH
   ratio = new_width / width
   new_height = int(height * ratio)
   cv2.namedWindow(filename) # Creates a window and names it after the image filename
   cv2.imshow(
      winname=filename,
      mat=cv2.resize(src=img, dsize=(new_width, new_height), interpolation=cv2.INTER_LINEAR)
   )
   cv2.waitKey(0)

def load_images(input_dir: str) -> np.ndarray:
   """
   Loads all images from the given file directory into a numpy array.
   """
   images = []
   for filename in os.listdir(input_dir):
      img = cv2.imread(filename=os.path.join(input_dir, filename))
      if img is not None:
         # Saves each image with its original filename
         imgset = []
         imgset.append(img)
         imgset.append(filename)
         images.append(imgset)
   return images

def save_images(image_list: np.ndarray, output_dir: str) -> None:
   """
   Saves all images in the list into the given file directory.
   """
   for imgset in image_list:
      cv2.imwrite(filename=os.path.join(output_dir, imgset[1]), img=imgset[0])

def get_image_filename(img: np.ndarray):
   """
   Returns the image filename based on the imgset structure.
   """
   return img[1]

def is_dir_path(dir_path: str) -> bool:
   """
   Returns true if the directory exists. Otherwise, it throws a NotADirectoryError.
   """
   if os.path.isdir(dir_path): 
      return True
   raise NotADirectoryError(dir_path)


# Main function
def main(parser=argparse.ArgumentParser()):
   """
   Commmand line input and main program functionality are handled here.
   """
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
      "max_thread_count", type=int, nargs="?", default=MAX_THREAD_COUNT, help="The maximum number of threads to use." # Defaults to MAX_THREAD_COUNT
   )
   # Parse the command line parameters upon executing the program
   args = parser.parse_args()
   print("")

   # Input validation
   assert is_dir_path(args.input_dir)
   assert is_dir_path(args.output_dir)

   print("Input images from: %s" %args.input_dir)
   print("Output images will go to: %s" %args.input_dir)
   print("Maximum time (mins): %d" %args.max_time_in_min)
   print("Brightness: %d" %args.brightness_factor)
   print("Contrast: %d" %args.sharpness_factor)
   print("Sharpness: %d" %args.contrast_factor)
   print("Maximum number of threads: %d" %args.max_thread_count)

   # Load images
   input_image_list = load_images(args.input_dir)
   output_image_list = []

   print("\nLoaded %d image(s): " %len(input_image_list))
   for image in input_image_list:
     print(get_image_filename(image))

   # Run image manipulation
   for image in input_image_list:
      output_image_list.append(image)

   # Save images
   save_images(output_image_list, args.output_dir) # Saves directly to output for testing


if __name__=="__main__":
   main()