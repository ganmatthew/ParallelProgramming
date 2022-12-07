# Input:
#     parahancer <input_dir> <output_dir> <max_time_in_min> <brightness_factor> <sharpness_factor> <contrast_factor> <max_thread_count>
# 
# Sample inputs:
#     parahancer E:\Input E:\Output 3 2 4 6 
#     parahancer E:\Input E:\Output 5 1 2 3 100

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import PIL.Image as Image
import PIL.ImageEnhance as ImageEnhance
import argparse

MAX_THREAD_COUNT = 61         # maximum number of threads on Windows is 61
PREVIEW_IMAGE_WIDTH = 800     # width of images in preview windows

# Container for image and metadata
class ImageSet:
   def __init__(self, original_image: np.ndarray, filename: str):
      self.original_image = original_image
      self.new_image = any
      self.filename = filename
   
   def get_original_image(self):
      return self.original_image

   def get_new_image(self):
      return self.new_image

   def get_filename(self):
      return self.filename

   def set_new_image(self, image: any):
      self.new_image = image

# Image manipulation
def apply_brightness(imgset: ImageSet, brightness_factor: int):
   brightness = ImageEnhance.Brightness(imgset.get_new_image())
   output_image = brightness.enhance(brightness_factor)
   imgset.set_new_image(output_image)
   #show_image(imgset)

def apply_contrast(imgset: ImageSet, contrast_factor: int):
   contrast = ImageEnhance.Contrast(imgset.get_new_image())
   output_image = contrast.enhance(contrast_factor)
   imgset.set_new_image(output_image)
   #show_image(imgset)

def apply_sharpness(imgset: ImageSet, sharpness_factor: int):
   sharpness = ImageEnhance.Sharpness(imgset.get_new_image())
   output_image = sharpness.enhance(sharpness_factor)
   imgset.set_new_image(output_image)
   #show_image(imgset)

def apply_brightness_contrast_sharpness(imgset: ImageSet, brightness_factor: int, contrast_factor: int, sharpness_factor: int):
   brightness = ImageEnhance.Brightness(imgset.get_new_image())
   output_image = brightness.enhance(brightness_factor)
   contrast = ImageEnhance.Contrast(output_image)
   output_image = contrast.enhance(contrast_factor)
   sharpness = ImageEnhance.Sharpness(output_image)
   output_image = sharpness.enhance(sharpness_factor)
   imgset.set_new_image(output_image)
   return imgset

# Image loading, saving, and displaying
def print_image_filenames(imgset_list: list):
   """
   Prints the filenames of all images line by line from the given ImageSet list.
   """
   imgset: ImageSet
   for imgset in imgset_list:
     print(imgset.get_filename())

def show_image(imgset: ImageSet):
   """
   Displays the given image in a new_image window.
   """
   image, filename = imgset.get_original_image(), imgset.get_filename()
   height, width = image.shape[0], image.shape[1]
   new_image_width = PREVIEW_IMAGE_WIDTH
   ratio = new_image_width / width
   new_image_height = int(height * ratio)
   cv2.namedWindow(filename) # Creates a window and names it after the image filename
   cv2.imshow(
      winname=filename,
      mat=cv2.resize(src=image, dsize=(new_image_width, new_image_height), interpolation=cv2.INTER_LINEAR)
   )
   #cv2.waitKey(0) # yield until any key is pressed

def load_images(input_dir: str) -> list:
   """
   Loads all images from the given file directory into an ImageSet list.
   """
   imgset_list = []
   for filename in os.listdir(input_dir):
      loaded_filename = os.path.join(input_dir, filename)
      #print(loaded_filename)
      img = cv2.imread(filename=loaded_filename)
      if img is not None:
         imgset_list.append(
            ImageSet(original_image=img, filename=filename)
         )
   return imgset_list

def save_images(imgset_list: list, output_dir: str) -> None:
   """
   Saves all images in the ImageSet list into the given file directory.
   """
   imgset: ImageSet
   for imgset in imgset_list:
      saved_filename = os.path.join(output_dir, imgset.get_filename())
      cv2.imwrite(filename=saved_filename, img=imgset.get_new_image())

# Image conversion between numpy array and PIL image formats
def numpy_to_image(img: np.ndarray) -> Image:
   mode = 'RGBA' if image_has_alpha(img) else 'RGB'
   return Image.fromarray(img, mode)

def image_to_numpy(img: Image) -> np.ndarray:
   return np.array(img)

# Checks for image alpha channel
def image_has_alpha(img: np.ndarray) -> bool:
   """
   Returns true if the numpy image array has an alpha channel.
   """
   _, _, c = img.shape
   return True if c == 4 else False

# Checks if the directory exists
def is_dir_path(dir_path: str) -> bool:
   """
   Returns true if the directory exists. Otherwise, it throws a NotADirectoryError.
   """
   if os.path.isdir(dir_path): 
      return True
   raise NotADirectoryError(dir_path)

# Returns true if there is still remaining time
def check_for_remaining_time(old_time: float, max_duration_in_mins: int) -> bool:
   """
   Returns true if there is still remaining time according to the maximum time argument
   """
   return True if time.time() - old_time > (max_duration_in_mins * 60) else False

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
      "max_time_in_min", type=int, default=np.int32, help="The maximum total time that the image processing can run for."
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

   print("Input images will be loaded from: %s" %args.input_dir)
   print("Output images will be saved to: %s" %args.output_dir)
   print("Maximum time: %d minute(s)" %args.max_time_in_min)
   print("Brightness: %dx" %args.brightness_factor)
   print("Contrast: %dx" %args.sharpness_factor)
   print("Sharpness: %dx" %args.contrast_factor)
   print("Maximum number of threads: %d" %args.max_thread_count)
   
   # Load images into a queue
   print("\nLoading images...")
   input_image_list = load_images(args.input_dir)
   print("\nLoaded %d image(s) " %len(input_image_list))
   #print_image_filenames(input_image_list)

   # Create a queue of images to manipulate
   output_image_list = []
   imgset: ImageSet
   for imgset in input_image_list:
      # Convert numpy image into PIL Image
      np_image = imgset.get_original_image()
      pil_image = numpy_to_image(np_image)
      imgset.set_new_image(pil_image)
      # Populate queue
      output_image_list.append(imgset)

   def method_1():
      imgset: ImageSet
      for imgset in output_image_list:
         apply_brightness(imgset, args.brightness_factor)

      imgset: ImageSet
      for imgset in output_image_list:
         apply_contrast(imgset, args.contrast_factor)

      imgset: ImageSet
      for imgset in output_image_list:
         apply_sharpness(imgset, args.sharpness_factor)

   def method_2():
      imgset: ImageSet
      for imgset in output_image_list:
         apply_brightness(imgset, args.brightness_factor)
         apply_contrast(imgset, args.contrast_factor)
         apply_sharpness(imgset, args.sharpness_factor)

   def method_3():
      imgset: ImageSet
      for imgset in output_image_list:
         t1 = threading.Thread(target=apply_brightness, args=(imgset, args.brightness_factor))
         t2 = threading.Thread(target=apply_contrast, args=(imgset, args.contrast_factor))
         t3 = threading.Thread(target=apply_sharpness, args=(imgset, args.sharpness_factor))
         t1.start()
         t2.start()
         t3.start()
         t1.join()
         t2.join()
         t3.join()

   def method_4():
      imgset: ImageSet
      for imgset in output_image_list:
         thread = threading.Thread(target=apply_brightness_contrast_sharpness, args=(imgset, args.brightness_factor, args.contrast_factor, args.sharpness_factor))
         thread.start()
         thread.join()

   def method_5():
      with ThreadPoolExecutor(max_workers=args.max_thread_count) as executor:
         results = executor.map(apply_brightness_contrast_sharpness, output_image_list, 
            [args.brightness_factor for _ in output_image_list], 
            [args.contrast_factor for _ in output_image_list],
            [args.sharpness_factor for _ in output_image_list],
            timeout=args.max_time_in_min)

      return list(results)

   # Run the processing and calculate execution time
   print("\nRunning...")
   start_time = time.time()
   output_image_list = method_5()
   #method_2()
   end_time = time.time() - start_time
   print("\nFinished in %0.4f seconds" %end_time)

   # Save images
   imgset: ImageSet
   for imgset in output_image_list:
      # Convert modified PIL Images to numpy arrays
      pil_image = imgset.get_new_image()
      np_image = image_to_numpy(pil_image)
      imgset.set_new_image(np_image)

   print("\nSaving %d image(s)..." %len(output_image_list))
   save_images(output_image_list, args.output_dir)
   print("\nDone!")


if __name__=="__main__":
   main()