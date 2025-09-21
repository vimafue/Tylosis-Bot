import cv2
import tkinter as tk
from tkinter import*
from tkinter import filedialog
from tkinter.filedialog import asksaveasfilename
from PIL import ImageTk,Image, ImageDraw, ImageFont, ImageOps
from skimage.morphology import closing
import math
import numpy as np
import skimage.measure
import skimage.io
import skimage.transform
import skimage.exposure
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
# switch off warnings related to deprecated functions
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


#evaluate all golbal variables
evaluate_single_image = TRUE                #evaluate single image or whole folder?
original_image : ImageTk.PhotoImage      #label for original image
empty_image : ImageTk.PhotoImage         #fills the labels if no other image is displayed
mask : ImageTk.PhotoImage                 #label for predicted mask
single_file_path = ""                           #path to single image to be evaluated (if "single image" was selected)
folder_path = ""                                #path to folder to be evaluated (if "whole folder" was selected)
save_folder = ""                          #path to save the predicted masks (if "whole folder" was selected)


#reads, cut to frames, predicts and preprocesses single image
def predict_image():
    img = cv2.imread(single_file_path, cv2.IMREAD_GRAYSCALE)

    img = cv2.equalizeHist(img)

    ######################################################################################################################################################################
    # Here starts Cut_frames subroutine #
    ######################################################################################################################################################################
    height = img.shape[0]  # get heigth of img
    width = img.shape[1]  # get width of img

    mustRescale = False  # initialize Rescale bool #image must be only rescaled when at least one of the image sides is < 1024
    scaleFactor = 1

    ######resize image if it doesnt fit a 1024x1024 frame#########
    if height < width:  # looks at shorter side
        if height < 1024:
            scaleFactor = 1024 / height  # scale-factor with which the whole image mst be rescaled so that the shorter side becomes 1024 pixel long
            mustRescale = True  # because ine side is < 1024, the image needs to be rescaled

            height = 1024
            width = width * scaleFactor

            if width < 1024:  # if the width is only 1023 (or so) through rounding errors, it gets corrected manually
                width = 1024

    else:
        if width < 1024:
            scaleFactor = 1024 / width
            mustRescale = True

            width = 1024
            height = height * scaleFactor

            if height < 1024:
                height = 1024

    if mustRescale:
        dim = (int(width), int(height))  # new height and width of the image
        img = cv2.resize(img, dim,interpolation=cv2.INTER_AREA)  # creates the resized image with the area interpolation (masks are not longer binary)
    ###################################################################################
    # Finds the number of 1024x1024 frames that fit into the image plus offset
    ###################################################################################
    num_x = math.floor(width / 1024)  # how many 1024-frames do fit horizontally?
    num_y = math.floor(height / 1024)  # how many 1024-frames do fit vertically?

    offset_x = math.floor((width % 1024) / 2)  # by how many pixels do I have to move my 1024^2 frames to the right so that they are centered horizontally? (after that there should be the same margin left and right)
    offset_y = math.floor((height % 1024) / 2)  # by how many pixels do I need to push my 1024^2 frames down so that they are vertically centered?

    ###################################################################################
    # Calculate startpoint of the 1024^2-frames (top left corner)
    ###################################################################################
    coordinates = []
    for i in range(num_x):
        for j in range(num_y):
            # x,y coordinates of all 1024x1024 frames that are directly nect to each other
            x = i * 1024 + offset_x
            y = j * 1024 + offset_y
            coordinates.append((x, y))
            if j < (num_y - 1):
                # x,y coordinates of all 1024x1024 frames that lie between the horizontal straight interfaces between the frames
                m = x
                n = y + 512  # 512 = 1024/2
                coordinates.append((m, n))
            if i < (num_x - 1):
                # x,y coordinates of all 1024x1024 frames that lie between the vertical straight intersections between the frames
                o = x + 512
                p = y
                coordinates.append((o, p))
            if i < (num_x - 1) and j < (num_y - 1):
                # x,y coordinates of all 1024x1024 frames lying on the corner points of the frames
                k = x + 512
                l = y + 512
                coordinates.append((k, l))
    ###################################################################################
    # Crops the frames out of the images
    ###################################################################################
    crop_images = []
    for i in range(len(coordinates)):  #
        (x, y) = coordinates[i]
        cropped = img[y:y + 1024, x:x + 1024]  # crops the image into i overlapping imgages
        crop_images.append(cropped)  # combines all cropped images in one array
    ###################################################################################
    ##Here comes that one image at a time is taken from the crop_images array and predicted and preprocessed and then stored in a new array
    ###################################################################################
    x_test = crop_images
    masks_array = []

    # Load Model #
    modelfile = os.getcwd() + "/model.h5"
    model = tf.keras.models.load_model(modelfile)

    # Take into consideration that the pixel-values are currently in the range of [0,1]
    threshold = 0.0  # <-- threshold must here not exceed 1 for avoiding a pure black mask

    for i in range(len(coordinates)):  #
        x = np.zeros((1, 1024, 1024, 1)) #placeholder for the 4D-array
        x[0][:, :, 0] = x_test[i] #write image into the first and second dimension to match the model-dimension-requirements

        estimate = model.predict(x, verbose=1)

        mask = np.squeeze(estimate, axis=(0, 3))

        # start multiplying with 255 and than switch to uint8 NOT the opposite order in order to prevent your mask from becoming purely black
        mask_pred = (mask * 255.0).astype(np.uint8)  # makes predicted mask 8bit binary

        # thresholding the predicted mask #
        (T, thresh) = cv2.threshold(mask_pred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Filling in the holes in the vessels#
        holes = thresh.copy()
        height = holes.shape[0]  # get heigth of holes image
        width = holes.shape[1]  # get width of holes image
        hasfilled=False #boolean parameter to end the for loop below

        for x in range(height):
            for y in range(width):
                if holes[x, y] == 0: #searches for a black seedPoint to floodfill from there
                    holes = thresh.copy()
                    cv2.floodFill(holes, None, (y, x), 255) #fills all black background pixels to white so that just the black holes are left
                    if np.average(holes) < 255*0.8:
                        continue
                    else:
                        hasfilled = True
                        break #ends inner for loop
            if hasfilled==True:
                break #ends outer for loop
        if hasfilled==True:
            holes = cv2.bitwise_not(holes)  # invert holes mask
            thresh = cv2.bitwise_or(thresh, holes)  # bitwise or of holes with mask to fill in holes

        # Removing Outliers #
        labeled_image = skimage.measure.label(thresh)  # perform connected component analysis
        im = skimage.morphology.remove_small_objects(labeled_image, 50)  # with the last value being the removed object size
        im = im.astype(np.uint8)  # make labeled image uint8 to put into threshold function
        (T, mask_t) = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY) #thresholds in case the image is not binary

        masks_array.append(mask_t)
    ######################################################################################################################################################################
    # then the masks are merged back into one image by ordering the overlapping areas
    ######################################################################################################################################################################

    mask = np.zeros((num_y * 1024, num_x * 1024),
                    dtype=np.uint8)  # defines big mask where all small mask-frames are combined

    for i in range(len(masks_array)):
        (x, y) = coordinates[i]
        (x, y) = (x - offset_x, y - offset_y)  # the offset is not being predicted so we have to subtract it from the endresult
        comb = cv2.bitwise_or(mask[y:y+1024, x:x+1024], masks_array[i]) #overlay prdicted mask with in mask existing frame to get whole mask with overlapping frames (alle Vessel-Pixels should be white in the end)
        mask[y:y + 1024, x:x + 1024] = comb  # saves the overlayed small masks into the big main mask
    ######################################################################################################################################################################
    # Here ends Cut_frames subroutine #
    ######################################################################################################################################################################
    return mask

#reads, cut to frames, predicts, preprocesses and saves batch of images out of the read folder
def predict_folder():
    # Load Model #
    modelfile = os.getcwd() + "/model.h5"
    model = tf.keras.models.load_model(modelfile)
    paths = []

    for file in os.listdir(folder_path):  # for every file in folder path
        if os.path.isdir(folder_path +"/"+ file):
            paths.append(folder_path +"/"+ file)  # determines subfolders in path
        else:
            filename = (folder_path +"/"+ file)  # path_name for each rgb image
            print(filename)
            save_way = (save_folder +"/"+ file)
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

            img = cv2.equalizeHist(img)

            ######################################################################################################################################################################
            # Here starts Cut_frames subroutine #
            ######################################################################################################################################################################
            height = img.shape[0]  # get heigth of img
            width = img.shape[1]  # get width of img

            mustRescale = False  # initialize Rescale bool #image must be only rescaled when at least one of the image sides is < 1024
            scaleFactor = 1

            ######resize image if it doesnt fit a 1024x1024 frame#########
            if height < width:  # looks at shorter side
                if height < 1024:
                    scaleFactor = 1024 / height  # scale-factor with which the whole image mst be rescaled so that the shorter side becomes 1024 pixel long
                    mustRescale = True  # because ine side is < 1024, the image needs to be rescaled

                    height = 1024
                    width = width * scaleFactor

                    if width < 1024:  # if the width is only 1023 (or so) through rounding errors, it gets corrected manually
                        width = 1024

            else:
                if width < 1024:
                    scaleFactor = 1024 / width
                    mustRescale = True

                    width = 1024
                    height = height * scaleFactor

                    if height < 1024:
                        height = 1024

            if mustRescale:
                dim = (int(width), int(height))  # new height and width of the image
                img = cv2.resize(img, dim,
                                 interpolation=cv2.INTER_AREA)  # creates the resized image with the area interpolation (masks are not longer binary)
            ###################################################################################
            # Finds the number of 1024x1024 frames that fit into the image plus offset
            ###################################################################################
            num_x = math.floor(width / 1024)  # how many 1024-frames do fit horizontally?
            num_y = math.floor(height / 1024)  # how many 1024-frames do fit vertically?

            offset_x = math.floor((
                                              width % 1024) / 2)  # by how many pixels do I have to move my 1024^2 frames to the right so that they are centered horizontally? (after that there should be the same margin left and right)
            offset_y = math.floor((
                                              height % 1024) / 2)  # by how many pixels do I need to push my 1024^2 frames down so that they are vertically centered?

            ###################################################################################
            # Calculate startpoint of the 1024^2-frames (top left corner)
            ###################################################################################
            coordinates = []
            for i in range(num_x):
                for j in range(num_y):
                    # x,y coordinates of all 1024x1024 frames that are directly nect to each other
                    x = i * 1024 + offset_x
                    y = j * 1024 + offset_y
                    coordinates.append((x, y))
                    if j < (num_y - 1):
                        # x,y coordinates of all 1024x1024 frames that lie between the horizontal straight interfaces between the frames
                        m = x
                        n = y + 512  # 512 = 1024/2
                        coordinates.append((m, n))
                    if i < (num_x - 1):
                        # x,y coordinates of all 1024x1024 frames that lie between the vertical straight intersections between the frames
                        o = x + 512
                        p = y
                        coordinates.append((o, p))
                    if i < (num_x - 1) and j < (num_y - 1):
                        # x,y coordinates of all 1024x1024 frames lying on the corner points of the frames
                        k = x + 512
                        l = y + 512
                        coordinates.append((k, l))
            ###################################################################################
            # Crops the frames out of the images
            ###################################################################################
            crop_images = []
            for i in range(len(coordinates)):  #
                (x, y) = coordinates[i]
                cropped = img[y:y + 1024, x:x + 1024]  # crops the image into i overlapping imgages
                crop_images.append(cropped)  # combines all cropped images in one array
            ###################################################################################
            ##Here comes that one image at a time is taken from the crop_images array and predicted and preprocessed and then stored in a new array
            ###################################################################################
            x_test = crop_images
            masks_array = []

            # Take into consideration that the pixel-values are currently in the range of [0,1]
            threshold = 0.0  # <-- threshold must here not exceed 1 for avoiding a pure black mask

            for i in range(len(coordinates)):  #
                x = np.zeros((1, 1024, 1024, 1))  # placeholder for the 4D-array
                x[0][:, :, 0] = x_test[
                    i]  # write image into the first and second dimension to match the model-dimension-requirements

                estimate = model.predict(x, verbose=1)

                mask = np.squeeze(estimate, axis=(0, 3))
                # mask = np.squeeze(estimate[:,:,:]) >= threshold  # thresholds every mask with the best threshold

                # start multiplying with 255 and than switch to uint8 NOT the opposite order in order to prevent your mask from becoming purely black
                mask_pred = (mask * 255.0).astype(np.uint8)  # makes predicted mask 8bit binary

                # thresholding the predicted mask #
                (T, thresh) = cv2.threshold(mask_pred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                # Filling in the holes in the vessels#
                holes = thresh.copy()
                height = holes.shape[0]  # get heigth of holes image
                width = holes.shape[1]  # get width of holes image
                hasfilled = False  # boolean parameter to end the for loop below

                for x in range(height):
                    for y in range(width):
                        if holes[x, y] == 0:  # searches for a black seedPoint to floodfill from there
                            holes = thresh.copy()
                            cv2.floodFill(holes, None, (y, x),
                                          255)  # fills all black background pixels to white so that just the black holes are left
                            if np.average(holes) < 255 * 0.8:
                                continue
                            else:
                                hasfilled = True
                                break  # ends inner for loop
                    if hasfilled == True:
                        break  # ends outer for loop
                if hasfilled == True:
                    holes = cv2.bitwise_not(holes)  # invert holes mask
                    thresh = cv2.bitwise_or(thresh, holes)  # bitwise or of holes with mask to fill in holes

                # Removing Outliers #
                labeled_image = skimage.measure.label(thresh)  # perform connected component analysis
                im = skimage.morphology.remove_small_objects(labeled_image,
                                                             50)  # with the last value being the removed object size
                im = im.astype(np.uint8)  # make labeled image uint8 to put into threshold function
                (T, mask_t) = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY)  # thresholds in case the image is not binary

                masks_array.append(mask_t)
            ######################################################################################################################################################################
            # then the masks are merged back into one image by ordering the overlapping areas
            ######################################################################################################################################################################

            mask = np.zeros((num_y * 1024, num_x * 1024),
                            dtype=np.uint8)  # defines big mask where all small mask-frames are combined

            for i in range(len(masks_array)):
                (x, y) = coordinates[i]
                (x, y) = (
                x - offset_x, y - offset_y)  # the offset is not being predicted so we have to subtract it from the endresult
                comb = cv2.bitwise_or(mask[y:y + 1024, x:x + 1024], masks_array[
                    i])  # overlay prdicted mask with in mask existing frame to get whole mask with overlapping frames (alle Vessel-Pixels should be white in the end)
                mask[y:y + 1024, x:x + 1024] = comb  # saves the overlayed small masks into the big main mask
            ######################################################################################################################################################################
            # Here ends Cut_frames subroutine #
            ######################################################################################################################################################################
        cv2.imwrite(save_way, mask)  # saves the mask
    return mask

class MyGUI:
    def __init__(self, master):
        self.master = master    # window=master
        master.title("Tylosis Bot") #title GUI window
        master.geometry("1200x1200")  #size GUI window
        master.configure(background='#ACD1AF') #backgroundcolor GUI window

        global original_image
        global empty_image
        global mask

        # configure frames:
        master.rowconfigure(0, weight = 1)     #creates two columns with equal width
        master.rowconfigure(1, weight=1)

        frame_top = Frame(master, bg='#ACD1AF')               #top frame holding everything but exit button
        frame_top.grid(row = 0, column=0)                 #attach to first row of master
        frame_top.columnconfigure(0, weight = 1)    #create two columns for top frame
        frame_top.columnconfigure(1, weight = 1)

        frame_left = Frame(frame_top, bg='#ACD1AF')           #left frame holding image/folder buttons + original image preview
        frame_left.grid(row=0, column=0, padx=25)                # attach to left column of top frame
        frame_left.rowconfigure(0, weight = 1)               # create two rows for left frame
        frame_left.rowconfigure(1, weight=1)

        frame_image_selection = Frame(frame_left, bg='#ACD1AF')   #contains image + folder button
        frame_image_selection.grid(row = 0, column=0)         #attach to upper row of left frame
        frame_image_selection.columnconfigure(0, weight=1)      #create column for image button
        frame_image_selection.columnconfigure(1, weight=1)      #create column for folder button

        frame_right = Frame(frame_top, bg='#ACD1AF')  # right frame holding predict/save buttons + mask preview
        frame_right.grid(row=0, column=1, sticky=E, padx=25)  # attach to right column of top frame
        frame_right.rowconfigure(0, weight=1)  # create two rows for right frame
        frame_right.rowconfigure(1, weight=1)

        frame_predict_save = Frame(frame_right, bg='#ACD1AF')  # contains predict + save button
        frame_predict_save.grid(row=0, column=0)  # attach to upper row of right frame
        frame_predict_save.columnconfigure(0, weight=1)  # create column for predict button
        frame_predict_save.columnconfigure(1, weight=1)  # create column for save button


        # Buttons
        self.open_image_button = Button(frame_image_selection, text="Open Single Image", command=self.open_image)  # when pushed, calls open image function
        self.open_image_button.grid(column=0, row=0, padx=20, pady=5)

        self.open_folder_button = Button(frame_image_selection, text="Open Folder", command=self.open_folder)      # when pushed, calls open folder function
        self.open_folder_button.grid(column=1, row=0, padx=20, pady=5)

        self.predict_button = Button(frame_predict_save, text="Predict", command=self.predict)        # when pushed, calls clean function
        self.predict_button.grid(column=0, row=0, padx=20, pady=5)

        self.savemask_button = Button(frame_predict_save, text="Save Mask", command=self.savemask)  # when pushed, calls savemask function
        self.savemask_button.grid(column=1, row=0, padx=20, pady=5)

        self.exit_button = Button(master, text="    Exit    ", command=master.quit)  # ends program
        self.exit_button.grid(column=0, row=1, sticky=NE, padx=25)

        # Labels
        empty_image = ImageTk.PhotoImage(Image.new(mode="RGB", size=(500, 500), color=(255, 255, 255))) # creates square white images to initialise the label

        self.label_original_image = Label(frame_left, image = empty_image, borderwidth=0) # label will contain the opened image
        self.label_original_image.grid(column = 0, row = 1, pady=10)

        self.label_mask = Label(frame_right, image=empty_image, borderwidth=0)  # label will contain the opened image
        self.label_mask.grid(column = 0, row = 1, pady=10)

    # Functions
    def open_image(self): # opens an Image
        global evaluate_single_image
        global single_file_path
        global original_image
        evaluate_single_image = TRUE #a single image was read in
        single_file_path = filedialog.askopenfilename(initialdir="", title="Select A File", filetypes=(
            ("PNG files", "*.png"), ("TIFF files", "*.tif"), ('JPEG files', ('*.jpg', '*.jpeg', '*.jpe', '*.jfif')), ("GIF files", "*.gif"))) #opens filedialog and saves the selected file name in filename
        im=skimage.io.imread(single_file_path)
        show_im = skimage.transform.resize(im,(500, 500))
        show_im=skimage.exposure.equalize_hist(show_im, nbins=256)
        show_im = (show_im * 255).astype("uint8")
        show_im = Image.fromarray(show_im)

        original_image = ImageTk.PhotoImage(show_im) # opens image
        self.label_original_image.configure(image=original_image, text="", compound="center", wraplength=490) # displays image into a label

    def open_folder(self): # opens a Folder
        global evaluate_single_image
        global original_image
        global folder_path
        global empty_image
        evaluate_single_image = FALSE #a folder was read in
        folder_path = filedialog.askdirectory(initialdir="", title="Select A Folder") # opens filedialog to select folder
        self.label_original_image.configure(image = empty_image, text="Source path:   "+folder_path, compound="center", wraplength=490)  # displays pathname into a label

    def predict(self): # starts the prediction and preprocessing
        global evaluate_single_image
        global mask_pred
        global show_mask
        global save_folder
        self.label_mask.configure(image=empty_image, text="Processing...", compound="center", wraplength=490)  # displays that the prediction is processing

        if evaluate_single_image==TRUE: #if a single image was read in, the function predict_image is called
            mask_pred = predict_image() #calls predict_image function
            show_mask=Image.fromarray(mask_pred)
            show_mask = show_mask.resize((500, 500))  # resizes the opened image to the labelsize
            show_mask = ImageTk.PhotoImage(show_mask)  # opens mask
            self.label_mask.configure(image=show_mask, text="", compound="center",wraplength=490)  # displays mask into a label

        else: #if a folder was read in, the function predict_folder is called
            save_folder = filedialog.askdirectory(initialdir="", title="Select A Destination Folder") # opens filedialog to select save folder
            mask_pred = predict_folder()#calls predict_folder function
            self.label_mask.configure(image=empty_image, text="Save path:   "+save_folder+"\n\n\n\n The prediction is completed", compound="center", wraplength=490)  # displays pathname into a label


    def savemask(self): # saves the mask of a single image
        global mask_pred
        save_path = asksaveasfilename(initialdir="/", title="Select A Save Path", defaultextension=".png", filetypes=(
            ('JPEG files', ('*.jpg', '*.jpeg', '*.jpe', '*.jfif')), ('PNG files', '*.png'),
            ('GIF files', '*.gif'))) # opens filedialog to select save path
        if save_path:
            cv2.imwrite(save_path, mask_pred)  # saves the predicted mask to the input file name.


root = Tk()
my_gui = MyGUI(root)
root.mainloop()