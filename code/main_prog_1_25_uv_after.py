#!/usr/bin/python3.7

from PIL import Image as Image_PIL
from PIL import ImageTk
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from subprocess import call
import board
import neopixel
from collections import deque
from bisect import insort, bisect_left
from itertools import islice
import os
from datetime import datetime
import scipy
from scipy.signal import savgol_filter, argrelmax, argrelextrema
import scipy.signal as ss
import motor_function as motor
import state as sm
import RPi.GPIO as GPIO
from tkinter import *
import threading
import ctypes
# import multiprocessing
threads = 0

pixels = neopixel.NeoPixel(board.D10, 16)
is_post_analysis = False
is_uv = False
last_iteration = 119
is_endpoint_overwrite = False
is_image_confirm = False
selection_array = list()
imgtk_frame = 0

uv_exposure_duration = 0

current_directory = 0
original_dir = 0
processed_dir = 0

def windowing_technique(moving_median_redness):
     # attempt to do windowing technique
    window_length = 3
    window_height = 10
    confidence_threshold = 6
    window_item = list()
    window_sample = list()
    window_count = 0;

    # point that passed through window vertically
    window_exceed_limit_item = list()
    window_exceed_limit_sample = list()

    endpoint_item = 0
    endpoint_sample = 0
    endpoint_offset_point = None
    final_endpoint_item = 0
    final_endpoint_sample = 0

    confidence_lvl = 0;

    end_point_reached_flag = 0

    for sample , item in enumerate(moving_median_redness):

        if sample == 0:
            # creating window
            left = sample
            right = sample + window_length
            up = window_height/2 + item
            down = item - window_height/2

        if (item > up or item < down or sample > right):

            # if the new data point passes through the window from top
            if (item > up):
                window_exceed_limit_item.append(item)
                window_exceed_limit_sample.append(sample)
                confidence_lvl += 1

                # if it passes through top window 3 consecutive times, it's actual etching point
                if confidence_lvl >= confidence_threshold:
                    endpoint_item = item
                    endpoint_sample = sample
                    endpoint_offset_point = round(sample * 0.10) + sample
                    #print(f"endpoint point is : {endpoint_offset_point}")

            else:
                # else, it is just a noise and we need to collect again
                confidence_lvl = 0

            # create new window if new data point passes through window from sides
            left = sample
            right = sample + window_length
            up = window_height/2 + item
            down = item - window_height
            window_count += 1

            window_item.append(item)
            window_sample.append(sample)

        # determining the the final endpoint (from last vertical crossed sample
        # + 15% offset of previous period)
        if (sample == endpoint_offset_point):
            final_endpoint_item = item
            final_endpoint_sample = sample
            #print(f"END POINT REACHED.")
            end_point_reached_flag = 1

    return window_item, window_sample, window_exceed_limit_item, window_exceed_limit_sample, final_endpoint_item, final_endpoint_sample, end_point_reached_flag

def running_median_insort(seq, window_size):
    """Contributed by Peter Otten"""
    seq = iter(seq)
    d = deque()
    s = []
    result = []
    for item in islice(seq, window_size):
        d.append(item)
        insort(s, item)
        result.append(s[len(d)//2])
    m = window_size // 2
    for item in seq:
        old = d.popleft()
        d.append(item)
        del s[bisect_left(s, old)]
        insort(s, item)
        result.append(s[m])
    return result


def nothing(x):
    pass

def _fill_image(img, connectivity):
    """Fills all holes in connected components in a binary image.

    Parameters
    ------
    img : numpy array
        binary image to fill

    Returns
    ------
    filled : numpy array
        The filled image
    """
    # Copy the image with an extra border
    h, w = img.shape[:2]
    img_border = np.zeros((h + 2, w + 2), np.uint8)
    img_border[1:-1, 1:-1] = img

    floodfilled = img_border.copy()
    mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(floodfilled, mask, (0, 0), 255, flags=connectivity)
    floodfill_inv = cv2.bitwise_not(floodfilled)
    filled = img_border | floodfill_inv
    filled = filled[1:-1, 1:-1]
    return filled

def colour_calculator_np(image, non_zero, non_zero_pixels_only = True):

    # converting into numpy version
    pix = np.array(image)

    if non_zero_pixels_only  == False:
        non_zero = pix.size/3

    # check for redness (R)
    #redness = pix[:,:,0] - (pix[:,:,1]+pix[:,:,2])/2
    redness = pix[:,:,2]#+pix[:,:,1] +pix[:,:,2] # only taking the red pixels value
#     print(non_zero)
    redness[redness<0] = 0
    avg_redness = np.sum(redness)/non_zero


    # check for greenness (G)
    greenness = pix[:,:,1]# - (pix[:,:,0]+pix[:,:,2])/2
    greenness[greenness<0] = 0
    avg_greenness = np.sum(greenness)/non_zero

    # check for blueness (B)
    blueness = pix[:,:,2]# - (pix[:,:,0]+pix[:,:,1])/2
    blueness[blueness<0] = 0
    avg_blueness = np.sum(blueness)/non_zero

    return avg_redness,avg_greenness,avg_blueness

def masking_preview(image_dir):

    global is_image_confirm, imgtk_frame

    # Load in image
    image = cv2.imread(image_dir)

    # Create a window
    cv2.namedWindow('image')
#     cv2.namedWindow('picture_output')

    # create trackbars for color change
    cv2.createTrackbar('HMin','image',0,179,nothing) # Hue is from 0-179 for Opencv
    cv2.createTrackbar('SMin','image',0,255,nothing)
    cv2.createTrackbar('VMin','image',0,255,nothing)
    cv2.createTrackbar('HMax','image',0,179,nothing)
    cv2.createTrackbar('SMax','image',0,255,nothing)
    cv2.createTrackbar('VMax','image',0,255,nothing)
#     cv2.createTrackbar('Colour','image',0,1,nothing)

    # Set default value for MAX HSV trackbars.
    cv2.setTrackbarPos('HMax', 'image', 179)
    cv2.setTrackbarPos('SMax', 'image', 255)
    cv2.setTrackbarPos('VMax', 'image', 255)
#     cv2.setTrackbarPos('Colour', 'image', 0)

    # Initialize to check if HSV min/max value changes
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = colour = 0

    wait_time = 33

    while(1):

        # get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin','image')
        sMin = cv2.getTrackbarPos('SMin','image')
        vMin = cv2.getTrackbarPos('VMin','image')

        hMax = cv2.getTrackbarPos('HMax','image')
        sMax = cv2.getTrackbarPos('SMax','image')
        vMax = cv2.getTrackbarPos('VMax','image')
#         colour = cv2.getTrackbarPos('Colour','image')

        # Set minimum and max HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])


        # Print if there is a change in HSV value
        if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
            print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax
            # Create HSV Image and threshold into a range.
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)

            # perform "closing" morphology on the mask
            kernel = np.ones((10,10),np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            # cv2.imwrite(f"MASK_without_flood.jpg", mask)

            # perform filling on the mask
            mask = _fill_image(mask,8)
            # cv2.imwrite(f"MASK.jpg", mask)
            # mask = cv2.bitwise_not(mask)

            # flood fill again
            mask = cv2.bitwise_not(mask)
            mask = _fill_image(mask,8)
            mask = cv2.bitwise_not(mask)

            non_zero = np.count_nonzero(mask)

            output = cv2.bitwise_and(image,image, mask= mask)
            cv2.imwrite(f"MASK_2nd_flood.jpg", mask)

            output_smaller = cv2.resize(output,(320,240))

            imgtk_raw = Image_PIL.fromarray(cv2.cvtColor(output_smaller,cv2.COLOR_BGR2RGB))
            imgtk_frame = ImageTk.PhotoImage(image = imgtk_raw)
            imgtk_frame_window = label_image.create_image(0, 0, anchor=NW, image=imgtk_frame)

        # Wait longer to prevent freeze for videos.
        if is_image_confirm == True or cv2.waitKey(wait_time) & 0xFF == ord('q'):
#             pixels.fill((0,0,0)) #yellow
            is_image_confirm = False
            break

    cv2.destroyAllWindows()

    return mask,non_zero



def focus_preview():
    global is_image_confirm

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

#     print("wait 5 seconds")
#     time.sleep(5)
    pixels.fill((255,0,0)) #yellow 247, 202, 24
    time.sleep(0.1)
    motor.relax_motion(27, 17, 22, 18)
    time.sleep(0.1)
    motor.relax_motion(9, 25, 11, 8, vertical = False)
    counter = 0
    while True:
        counter = (counter + 1)%1000
        if counter == 0:
            try:
                label_image.delete(focus_preview_window)
                frame = None
                frame_smaller = None
                imgtk_raw  = None
                imgtk = None

            except:
                pass
    #         pixels = neopixel.NeoPixel(board.D10, 16)
    #         pixels.fill((255,0,0)) #yellow 247, 202, 24
        ret,frame = cap.read()
    #     frame = cv2.resize(frame,None, fx = 0.5, fy = 0.5, interpolation=cv2.INTER_AREA)
        frame_smaller = cv2.resize(frame,(320,240))
#         cv2.imshow('Input',frame_smaller)
        imgtk_raw = Image_PIL.fromarray(cv2.cvtColor(frame_smaller,cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image = imgtk_raw)
#         label_image.imgtk = imgtk
#         label_image.configure(image=imgtk)
        focus_preview_window = label_image.create_image(0, 0, anchor=NW, image=imgtk)
        if is_image_confirm == True:
#             pixels.fill((0,0,0)) #yellow
            is_image_confirm = False
            time.sleep(0.1)
            motor.relax_motion(27, 17, 22, 18)
            time.sleep(0.1)
            motor.relax_motion(9, 25, 11, 8, vertical = False)
            cap.release()
            cv2.destroyAllWindows()
            break

def emergency_stop():
    global threads

    print("STOPPPPPPPPP")
    my_label.config(text="Emergency stop")
    motor.stop_flag = 1
    threads.raise_exception()
    print("Done raising flag")
#     threads.join()
    print("Emergency settled")
    sm.initialisation()
    sm.emergency_stop_motion()# this function should be changed to able to keep track current vertical count. Able to stop half way.

def loading_pos():
    # motor movement
    print("hello")
    sm.initialisation()
    sm.original_pos_check()
    my_label.config(text="Back to loading position")

class thread_with_exception(threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name

    def run(self):

        # target function of the thread class
        try:
            global current_directory, original_dir, processed_dir, uv_exposure_duration
            global is_endpoint_overwrite, last_iteration, is_uv, is_image_confirm
            
            savgol_endpoint = 0
            savgol_endpoint_value = 0

            # create a new directory whenever a new operation started
            # new directory = 'current date' 'current time'
            current_directory, original_dir, processed_dir, plot_dir = motor.create_new_directory()
            try:
                sm.initialisation()
            except:
                my_label.config(text="Back to loading position")

            sm.station_1()
            my_label.config(text="Reaches station 1")
            if is_post_analysis != True:
                my_label.config(text="Please ensure the view is focussed.")
                # First to preview the image to help focus
                # Press "q" to exit
                focus_preview()

            max_iter = 5
            hue_handler = list()
            saturation_handler = list()
            value_handler = list()
            redness_avg = list()
            greenness_avg = list()
            blueness_avg = list()


            # capture picture from "webcam"
            if is_post_analysis != True:
                snapshot_string = original_dir + '/' + "snapshot.jpg"
                cap = cv2.VideoCapture(0)
                ret,frame = cap.read()
                cap.release()
                cv2.imwrite(snapshot_string, frame)

            else:
                sample_no = var.get()
                if (var.get() == 1):
                    directory = os.path.join(current_directory, r'sample_1')
                    last_iteration = 119

                elif (var.get() == 2):
                    directory = os.path.join(current_directory, r'sample_2')
                    last_iteration = 186

                elif (var.get() == 3):
                    directory = os.path.join(current_directory, r'sample_3')
                    last_iteration = 187

                elif (var.get() == 4):
                    directory = os.path.join(current_directory, r'sample_4')
                    last_iteration = 109

                else:
                    directory = os.path.join(current_directory, r'sample_6/Original')#'sample_1')
                    last_iteration = 658

#                 directory = os.path.join(current_directory, sample_str)
                snapshot_string = directory + '/' + "snapshot.jpg"# directory of the downloaded image

            # enpoint flag
            end_point_reached_flag = 0

            my_label.config(text="Please adjust the limit for masking.")
            mask,non_zero = masking_preview(snapshot_string)
            my_label.config(text="Etching endpoint monitoring ongoing.")

            start_time = time.time()

            i = 0

            while (1):
                iteration_starttime = time.time()
                if is_post_analysis != True:
                    # Live analysis
                    # capture picture from "webcam"
                    snapshot_string = original_dir + '/' + "microscope_" + str(i) + ".jpg"
                    # call(["fswebcam","-r", "1280x700","--no-banner", snapshot_string])

                    cap = cv2.VideoCapture(0)
#                     if is_uv == True:
#                         pixels.fill((0,0,0)) #turn off led
#                         motor.uv_led_on();
#                         uv_start_time = time.time()
#                         time.sleep(0.05)
                        
                    ret,frame_1 = cap.read()
                    
#                     if is_uv == True:
#                         motor.uv_led_off();
#                         uv_end_time = time.time()
#                         uv_exposure_duration = (uv_end_time - uv_start_time) + uv_exposure_duration

                    if is_uv == True and i%2 == 0:
                        motor.uv_led_on();
                        uv_start_time = time.time()
                        time.sleep(0.15)
                        motor.uv_led_off();
                        uv_end_time = time.time()
                        uv_exposure_duration = (uv_end_time - uv_start_time) + uv_exposure_duration
        
                    cap.release()
                    cv2.imwrite(snapshot_string, frame_1)


                else:
                    # Post analysis
                    snapshot_string = directory + '/' + "microscope_" + str(i) + ".jpg" # directory of the downloaded image with counter

                # this is to open precaptured image
                image = cv2.imread(snapshot_string)

                # mask the image to extract chips region and save
                image = cv2.bitwise_and(image, image, mask = mask)
                processed_string = processed_dir + '/' + "sample_masked_" + str(i) + ".jpg"
                cv2.imwrite(processed_string, image)

                # this is to open precaptured image
                masked_image = image
                masked_image_smaller = cv2.resize(masked_image,(320,240))

                if end_point_reached_flag == 1:
                    font                   = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (0,200)
                    fontScale              = 1
                    fontColor              = (255,0,0)
                    lineType               = 2

                    masked_image_smaller = cv2.putText(masked_image_smaller,'##END POINT!##', bottomLeftCornerOfText, font, fontScale,fontColor,lineType)

        #         cv2.imshow('picture_output',masked_image_smaller )

                imgtk_raw = Image_PIL.fromarray(cv2.cvtColor(masked_image_smaller ,cv2.COLOR_BGR2RGB))
                imgtk_endpoint = ImageTk.PhotoImage(image = imgtk_raw)
#                 label_image.configure(image=imgtk_endpoint)
                label_image.create_image(0, 0, anchor=NW, image=imgtk_endpoint)

                # It converts the BGR color space of image to HSV color space
                hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)

                np_hue = np.resize(hsv[:,:,0],(1,-1))
                hue_avg = ((np.mean(np_hue))/180)
                hue_handler.append(hue_avg)
                # print(hue_avg)

                np_saturation = np.resize(hsv[:,:,1],(1,-1))
                saturation_avg = np.mean(np_saturation)/255
                saturation_handler.append(saturation_avg)

                np_value = np.resize(hsv[:,:,2],(1,-1))
                value_avg = np.mean(np_value)/255
                value_handler.append(value_avg)

                # calculate redness, greenness, blueness (RGB)
                redness, greenness, blueness = colour_calculator_np(masked_image,non_zero)
                redness_avg.append(redness)
            #     greenness_avg.append(greenness)
            #     blueness_avg.append(blueness)

            #     if(len(redness_avg)>1):
            #         redness_gradient_numpy = np.gradient(np.array(redness_avg),2)

                if(len(redness_avg)>=7):

                    median_window_size = 7
                    # eleminate noise using moving median
                    moving_median_redness = running_median_insort(redness_avg, median_window_size)

                    # window method to analyse filtered redness graph
                    window_item, window_sample, window_exceed_limit_item, window_exceed_limit_sample, final_endpoint_item, final_endpoint_sample, end_point_reached_flag = windowing_technique(moving_median_redness)

                    redness_gradient_numpy = np.gradient(np.array(moving_median_redness),2)

                # inflection computation for endpoint deterination
                if(len(redness_avg)>=11):
                    div = ss.savgol_filter(moving_median_redness,9, 2, deriv = 1)
                    maxima = argrelmax(div)

                    if len(maxima[0]) != 0:
                        maxima_value = max(div[maxima[0]])
                        location = maxima[0][np.argmax(div[maxima[0]])]

                        # setting a threshold for derevative inflection determination
                        if maxima_value >= 1: #5
                            
                            # predicting the endpoint from inflection point
                            predicted_endpoint = location*1.25 # should still be around 1.25
                            end_string = "Change detected!" + "End in " + str(round(predicted_endpoint-i))
                            my_label.config(text=end_string)
                            
                            if i >= predicted_endpoint:
                                print("HELLO!!!!!!!!!!!!!!!!!!!!!")
                                my_label.config(text="Endpoint Reached!")
                                my_label.config(bg = 'yellow')
                                savgol_endpoint = predicted_endpoint
                                savgol_endpoint_value = moving_median_redness[int(savgol_endpoint)]
                                
                                if is_endpoint_overwrite:
                                    # manual
                                    pass
                                else:
                                    # auto termination
                                    pixels.fill((0,0,0)) #turn off led
                                    my_label.config(bg = 'white')
                                    cv2.destroyAllWindows()
                                    break
                                
                                

                i = i + 1
                if is_image_confirm:
                    is_image_confirm = False
                    pixels.fill((0,0,0)) #turn off led
                    my_label.config(bg = 'white')
                    cv2.destroyAllWindows()
                    break

                if is_post_analysis == True:
                    if i == last_iteration:
                        break

                iteration_endtime = time.time()
                # add in delay to make each iteration exactly 1 s each
                if is_post_analysis == False:
                    try:
                        time.sleep(1-(iteration_endtime - iteration_starttime))
                    
                    except:
                        pass

            end_time = time.time()
            print("%d iterations took %.2f seconds, which corresponds to %.2fs/iteration" % (i, end_time - start_time, (end_time - start_time)/i))
            print("Samples exposed to UV %d times and took %.2f seconds, which corresponds to %.2fs/iteration" % (i, uv_exposure_duration, uv_exposure_duration/i))
        # #np.savez('redness_data_sample_5.npz', sample_1=moving_median_redness)
        #
            # motor move to the next station
            my_label.config(text="Water bath")
            sm.station_2()
            time.sleep(10)
            my_label.config(text="Acetone bath")
            sm.station_3()
            time.sleep(10)
            my_label.config(text="IPA bath")
            sm.station_4()
            time.sleep(10)
            my_label.config(text="Back to loading position")
            sm.back_to_loading_pos()
            my_label.config(text="Sample ready to be collected")

            x = np.arange(0,int(len(hue_handler)),1)
            smoothed_red = savgol_filter(moving_median_redness, 9, 3)

            fig = plt.figure()
#             ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(211)
            ax3 = fig.add_subplot(212)

#             ax1.plot(x, hue_handler, marker='o')
#             ax1.plot(x, saturation_handler, marker='o')
#             ax1.plot(x, value_handler, marker='o')

            ax2.plot(x, redness_avg, marker= 'o')
            # ax2.plot(x, greenness_avg, marker= 'o')
            # ax2.plot(x, blueness_avg, marker= 'o')

            x_moving_avg = np.arange(0,int(len(moving_median_redness)),1)
            ax3.plot(x_moving_avg,moving_median_redness, marker= 'o')
            ax3.plot(window_sample,window_item, marker= 'o')
            ax3.plot(window_exceed_limit_sample,window_exceed_limit_item, marker= 'o')
            # ax3.plot(endpoint_sample, endpoint_item, marker= 'o')
            ax3.plot(final_endpoint_sample, final_endpoint_item, marker= 'o')

#             ax1.set_title("HSV parameters of images")
#             ax1.set_xlabel('Iterations')
#             ax1.legend(['hue','saturation','lightness'], fontsize='small',labelspacing=0.2)

            ax2.set_title("Average RGB of images")
            ax2.set_xlabel('Iterations')
            ax2.legend(['R','G','B'], fontsize='small',labelspacing=0.2)

            ax3.set_title("Smoothen version of redness graph (range = 0-255)")
            ax3.set_xlabel('Iterations')
            ax3.set_ylabel('Avg. red pixels value')
            ax3.legend(['Smothen R pixels graph','Windows','Datapoint that passed through window vertically', "Expected endpoint (+10% intial time)"], fontsize='small',labelspacing=0.2)

            fig.tight_layout(pad = 1.0)
            
            plot_string = plot_dir + '/' + "Red_pixels_trend_smoothed_and_raw.png"
            fig.savefig(plot_string)

            fig2 = plt.figure()
            ax = fig2.add_subplot(111)
            ax.plot(x_moving_avg ,redness_gradient_numpy, marker='o')
            ax.set_title("Raw (Numpy) derivative of the red (R) pixels trend")
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Raw (Numpy) derivative')
            plot_string = plot_dir + '/' + "numpy_derivative.png"
            fig2.savefig(plot_string)
            

            fig3 = plt.figure()
            ax = fig3.add_subplot(111)
            ax.plot(x_moving_avg ,div, marker='o')
            ax.set_title("Savgol derivative of the red (R) pixels trend")
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Savgol derivative')
            plot_string = plot_dir + '/' + "savgol_derivative.png"
            fig3.savefig(plot_string)

            fig4 = plt.figure()
            ax = fig4.add_subplot(111)
            ax.plot(x_moving_avg,moving_median_redness, marker='o')
            ax.plot(savgol_endpoint ,savgol_endpoint_value, marker='o')
            ax.set_title("Smoothen version of redness graph (range = 0-255)")
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Avg. red pixels value')
            ax.legend(['Smothen R pixels graph', 'Endpoint from Savgol derivative'], fontsize='small',labelspacing=0.2)
            plot_string = plot_dir + '/' + "savgol_derivative_endpoint.png"
            fig4.savefig(plot_string)

            
            plt.show()
        finally:
            print('ended')

    def get_id(self):

        # returns id of the respective thread
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    def raise_exception(self):
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id,
              ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print('Exception raise failure')

def thread_creation():
    global threads
    t1 = thread_with_exception('Thread 1')
    t1.start()

    threads = t1

def toggle():
    global is_post_analysis

    if toggle_btn.config('relief')[-1] == 'sunken':
        toggle_btn.config(relief="raised")
        toggle_btn.config(text="Live monitor")
        is_post_analysis = False

        for i in selection_array:
            i.grid_remove()

        selection_array.clear()


    else:
        toggle_btn.config(relief="sunken")
        toggle_btn.config(text="Post analysis")
        is_post_analysis = True
        sample_choice = Label(text = "Sample no?")
        sample_choice.grid(row = 6, column = 0)

        R1 = Radiobutton(root, text="1", variable=var, value=1,
                          command=sel)
        R1.grid(row = 6, column = 1)

        R2 = Radiobutton(root, text="2", variable=var, value=2,
                          command=sel)
        R2.grid(row = 6, column = 2)

        R3 = Radiobutton(root, text="3", variable=var, value=3,
                          command=sel)
        R3.grid(row = 7, column = 1)

        R4 = Radiobutton(root, text="4", variable=var, value=4,
                          command=sel)
        R4.grid(row = 7, column = 2)

        selection_array.extend((sample_choice, R1, R2, R3, R4))
def toggle_uv():
    global is_uv

    if toggle_btn_uv.config('relief')[-1] == 'sunken':
        toggle_btn_uv.config(relief="raised", bg = 'yellow')
        toggle_btn_uv.config(text="Visible\nlight")
        is_uv = False

    else:
        toggle_btn_uv.config(relief="sunken", bg = 'violet')
        toggle_btn_uv.config(text="UV\nlight")
        is_uv= True


def toggle_endpoint():
    global is_endpoint_overwrite

    if toggle_btn_endpoint.config('relief')[-1] == 'sunken':
        toggle_btn_endpoint.config(relief="raised")
        toggle_btn_endpoint.config(text="Auto\nendpoint")
        is_endpoint_overwrite = False

    else:
        toggle_btn_endpoint.config(relief="sunken")
        toggle_btn_endpoint.config(text="Manual\nendpoint")
        is_endpoint_overwrite = True

def confirm_image():
    global is_image_confirm

    is_image_confirm = True

def sel():
    selection = "You selected the option " + str(var.get())
#    label.config(text = selection)
    print(str(var.get()))

root = Tk()

root.title("Learn To Code at Codemy.com")
root.geometry("380x480")

current_display = Label(root, text="Current state:", anchor=N)
current_display.grid(row = 0, column = 0)

my_label = Label(root, width=35, text="Welcome to wet etching process bench!")
my_label.config(bg = 'white')
my_label.grid(row = 1, column = 0, columnspan = 2, pady = 10, sticky = 'nsew')

next_btn_img = PhotoImage(file = "green_button.PNG")
# Create A Button
next_btn = Button(root, image = next_btn_img , bd = 0, command = confirm_image)
next_btn.grid(row = 1, column = 2, sticky = 'nsew')

my_button1 = Button(root, text="Loading\nstation", command= loading_pos)#lambda: threading.Thread(target=loading_pos).start())
my_button1.grid(row = 3, column = 0, sticky = 'nsew')

toggle_btn_endpoint = Button(root, text="Auto\nendpoint", width = 9, relief="raised", command=toggle_endpoint)
toggle_btn_endpoint.grid(row = 2, column = 0, sticky = 'nsew')

# toggle_btn = tk.Button(text="Toggle", width=12, relief="raised")
toggle_btn = Button(root, text="Live monitor", width = 9, relief="raised", command=toggle)
toggle_btn.grid(row = 2, column = 1, sticky = 'nsew')

toggle_btn_uv = Button(root, text="Visible\nlight", width = 9, relief="raised", bg = 'yellow', command=toggle_uv)
toggle_btn_uv.grid(row = 2, column = 2, sticky = 'nsew')

# t1 = threading.Thread(target=start)
# thread_with_exception
# t1 = thread_with_exception('Thread 1')

my_button2 = Button(root, text="Start", command= lambda: thread_creation())#multiprocessing.Process(target=start).start())#threading.Thread(target=start).start())
my_button2.grid(row = 3, column = 1, sticky = 'nsew')#.pack(padx=50, pady = 5, side=LEFT, expand='yes', anchor = N)

my_button3 = Button(root, text="Emergency\nStop", bg='red', command= lambda: threading.Thread(target=emergency_stop).start())#emergency_stop)#lambda: threading.Thread(target=emergency_stop).start())
my_button3.grid(row = 3, column = 2, sticky = 'nsew')#.pack(padx=50, pady = 5, side=RIGHT, expand='yes', anchor = N)

label_image = Canvas(root, width=320, height = 240)
label_image.grid(row = 5, column = 0, columnspan = 3)

var = IntVar()

root.mainloop()





