# Parallel Programming Image Enhancement
A Python command line application that performs basic image brightness, contrast, and sharpness adjustments on batch files with the use of thread pools.

# Getting Started
## Prerequisites
To run this application, you will need to have the following installed on your system:
- Python 3

# Usage
1. Run the Python script with the following positional arguments:
> parahancer.py <input_dir> <output_dir> <output_dir> <max_time_in_min> <max_time_in_min> <brightness_factor> <brightness_factor> <sharpness_factor> <sharpness_factor> <contrast_factor> <max_thread_count>
  * `input_dir`: The file directory to take input images from.
  * `output_dir`: The file directory to put output images into.
  * `max_time_in_min`: The maximum total time that the image processing can run for.
  * `brightness_factor`: Adjusts the brightness factor.
  * `sharpness_factor`: Adjusts the sharpness factor.
  * `contrast_factor`: Adjusts the contrast factor.
  * `max_thread_count`: (optional) THe maximum number of threads to use.

For example, to check for images in `E:\Input` and put them in `E:\Output` and apply a brightness factor of 3, a sharpness factor of 2, a contrast factor of 4, and limit the number of threads to 6, you can run:

`parahancer E:\Input E:\Output 3 2 4 6`

2. Once done, the script will also generate a ``stats.txt`` file containing the statistics of the program run in the same directory as the Python file.

Created as part of coursework for STDISCM (Distributed Computing) at De La Salle University-Manila
