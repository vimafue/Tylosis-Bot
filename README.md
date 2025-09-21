# Tylosis-Bot
This code deals with the processing of micro-CT images of oak wood samples. Among other things, it includes a Segmentation U-Net trained by me, which removes the tylosis from the vessels. This important process facilitates the analysis of said vessels for climate-related research. This project was developed in collaboration with the German Archaeological Institute. The algorithms and the neural network are integrated into a GUI.


## How to get the Tylosis-Bot running
This is a quick guide on how to get my program running on your computer.

### 1. Python
- my version: Python 3.8.0: https://www.python.org/downloads/release/python-380/
- newest version: https://www.python.org/downloads/

The Python version may affect the functionality of the program, as in the past, the versions of the various libraries have played a role in terms of their compatibility with each other.

### 2. Anaconda
I used the paket manager anaconda simply because the compatibility of the versions of opencv, tensorflow, python and cuda can be very difficult to match. Anaconda helps with that.
-	https://www.anaconda.com/
-	within anaconda: install PyCharm
-	Add additional libraries to the interpreter in Anaconda:
    -	opencv
    - tensorflow

### 3. PyCharm: Conda Environment
Change the settings within PyCharm to add the conda environment.
- Settings -> Python Interpreter -> Add Interpreter -> Conda Environment (cross at „make available for all projects“) -> OK -> OK
- https://lucidgen.com/en/add-conda-virtual-environment-to-pycharm/

### 4. Get Model file
The model file of the trained U-Net was too large for the GitHub repository. It can be found and downloaded on Zenodo using the following link.
On your computer, the model file (model.h5) should be placed in the same folder as the main.py file to ensure that the program works properly

https://doi.org/10.5281/zenodo.17170930

### 5. Run Tylosis-Bot

- open file "main.py" with PyCharm
- start GUI by clicking the green play button in the top right corner of PyCharm
- If GUI/program crashes or freezes, click the red square in the top right corner of PyCharm (= emergency stop)

If the GUI window opens, you can decide if you want to load one singular image or a stack of images by pressing the corresponding button. The input image/images need to be grayscale images of any size. A custom cropping algorithm is implemented into the code to scale and cut the images into the right size for the Unet input. Later the cut up images are stitched back together. The original size is not always restored after the segmentation.


