import RPi.GPIO as GPIO
import time
import board
from datetime import datetime
import os

# for stepper motor 
GPIO.cleanup()

# Sensor_flag
sensor_1_flag = 0
sensor_2_flag = 0
sensor_3_flag = 0
sensor_4_flag = 0
sensor_5_flag = 0
stop_flag = 0

# movement counter
vertical_counter = 0
horizontal_counter = 0
sign = 1
vertical_current_pos = 0

# Pin Assignment
out1 = 27 #pin 13
out2 = 17 #pin 11
out3 = 22 #pin 15
out4 = 18 #pin 12
uv_led = 26 #pin37

out1a = 9 #pin 21
out2a = 25 #pin 22 
out3a = 11 #pin 23
out4a = 8 #pin 24

sensor1 = 23 #pin16
sensor2 = 5 #pin29
sensor3 = 6 #pin31
sensor4 = 12 #pin32
sensor5 = 13 #pin33

# setup the allocated pin and set it to be input/ouput
GPIO.setmode(GPIO.BCM)
GPIO.setup(out1,GPIO.OUT)
GPIO.setup(out2,GPIO.OUT)
GPIO.setup(out3,GPIO.OUT)
GPIO.setup(out4,GPIO.OUT)
GPIO.setup(uv_led,GPIO.OUT)

GPIO.setup(out1a,GPIO.OUT)
GPIO.setup(out2a,GPIO.OUT)
GPIO.setup(out3a,GPIO.OUT)
GPIO.setup(out4a,GPIO.OUT)

GPIO.setup(sensor1, GPIO.IN)  # Set our input pin to be an input
GPIO.setup(sensor2, GPIO.IN)  # Set our input pin to be an input
GPIO.setup(sensor3, GPIO.IN)  # Set our input pin to be an input
GPIO.setup(sensor4, GPIO.IN)  # Set our input pin to be an input
GPIO.setup(sensor5, GPIO.IN)  # Set our input pin to be an input

GPIO.output(out1,GPIO.LOW) 
GPIO.output(out2,GPIO.LOW)
GPIO.output(out3,GPIO.LOW)
GPIO.output(out4,GPIO.LOW)
GPIO.output(uv_led,GPIO.HIGH)
GPIO.output(out1a,GPIO.LOW) 
GPIO.output(out2a,GPIO.LOW)
GPIO.output(out3a,GPIO.LOW)
GPIO.output(out4a,GPIO.LOW)


def horizontal_right_motion(p1, p2, p3, p4):
    """Horizontal stepper motor moves to the right. 

    Parameters:
    p1 (int): Stepper motor signal pin1
    p2 (int): Stepper motor signal pin2
    p3 (int): Stepper motor signal pin3
    p4 (int): Stepper motor signal pin4

   """

    global horizontal_counter
    
    if horizontal_counter==0:
        GPIO.output(p1,GPIO.HIGH)
        GPIO.output(p2,GPIO.LOW)
        GPIO.output(p3,GPIO.LOW)
        GPIO.output(p4,GPIO.LOW)
        time.sleep(0.001)

    elif horizontal_counter==1:
        GPIO.output(p1,GPIO.HIGH)
        GPIO.output(p2,GPIO.HIGH)
        GPIO.output(p3,GPIO.LOW)
        GPIO.output(p4,GPIO.LOW)
        time.sleep(0.001)

    elif horizontal_counter==2:  
        GPIO.output(p1,GPIO.LOW)
        GPIO.output(p2,GPIO.HIGH)
        GPIO.output(p3,GPIO.LOW)
        GPIO.output(p4,GPIO.LOW)
        time.sleep(0.001)

    elif horizontal_counter==3:    
        GPIO.output(p1,GPIO.LOW)
        GPIO.output(p2,GPIO.HIGH)
        GPIO.output(p3,GPIO.HIGH)
        GPIO.output(p4,GPIO.LOW)
        time.sleep(0.001)

    elif horizontal_counter==4:  
        GPIO.output(p1,GPIO.LOW)
        GPIO.output(p2,GPIO.LOW)
        GPIO.output(p3,GPIO.HIGH)
        GPIO.output(p4,GPIO.LOW)
        time.sleep(0.001)

    elif horizontal_counter==5:
        GPIO.output(p1,GPIO.LOW)
        GPIO.output(p2,GPIO.LOW)
        GPIO.output(p3,GPIO.HIGH)
        GPIO.output(p4,GPIO.HIGH)
        time.sleep(0.001)

    elif horizontal_counter==6:    
        GPIO.output(p1,GPIO.LOW)
        GPIO.output(p2,GPIO.LOW)
        GPIO.output(p3,GPIO.LOW)
        GPIO.output(p4,GPIO.HIGH)
        time.sleep(0.001)

    elif horizontal_counter==7:    
        GPIO.output(p1,GPIO.HIGH)
        GPIO.output(p2,GPIO.LOW)
        GPIO.output(p3,GPIO.LOW)
        GPIO.output(p4,GPIO.HIGH)
        time.sleep(0.001)

    if horizontal_counter==0:
        horizontal_counter=7
    else:
        horizontal_counter=horizontal_counter-1

def horizontal_left_motion (p1, p2, p3, p4):
    """Horizontal stepper motor moves to the left. 

    Parameters:
    p1 (int): Stepper motor signal pin1
    p2 (int): Stepper motor signal pin2
    p3 (int): Stepper motor signal pin3
    p4 (int): Stepper motor signal pin4

   """
    global horizontal_counter
    
    if horizontal_counter==0:
        GPIO.output(p1,GPIO.HIGH)
        GPIO.output(p2,GPIO.LOW)
        GPIO.output(p3,GPIO.LOW)
        GPIO.output(p4,GPIO.LOW)
        time.sleep(0.001)
    #time.sleep(1)
    elif horizontal_counter==1:
        GPIO.output(p1,GPIO.HIGH)
        GPIO.output(p2,GPIO.HIGH)
        GPIO.output(p3,GPIO.LOW)
        GPIO.output(p4,GPIO.LOW)
        time.sleep(0.001)
    #time.sleep(1)
    elif horizontal_counter==2:  
        GPIO.output(p1,GPIO.LOW)
        GPIO.output(p2,GPIO.HIGH)
        GPIO.output(p3,GPIO.LOW)
        GPIO.output(p4,GPIO.LOW)
        time.sleep(0.001)

    elif horizontal_counter==3:    
        GPIO.output(p1,GPIO.LOW)
        GPIO.output(p2,GPIO.HIGH)
        GPIO.output(p3,GPIO.HIGH)
        GPIO.output(p4,GPIO.LOW)
        time.sleep(0.001)

    elif horizontal_counter==4:  
        GPIO.output(p1,GPIO.LOW)
        GPIO.output(p2,GPIO.LOW)
        GPIO.output(p3,GPIO.HIGH)
        GPIO.output(p4,GPIO.LOW)
        time.sleep(0.001)

    elif horizontal_counter==5:
        GPIO.output(p1,GPIO.LOW)
        GPIO.output(p2,GPIO.LOW)
        GPIO.output(p3,GPIO.HIGH)
        GPIO.output(p4,GPIO.HIGH)
        time.sleep(0.001)

    elif horizontal_counter==6:    
        GPIO.output(p1,GPIO.LOW)
        GPIO.output(p2,GPIO.LOW)
        GPIO.output(p3,GPIO.LOW)
        GPIO.output(p4,GPIO.HIGH)
        time.sleep(0.001)

    elif horizontal_counter==7:    
        GPIO.output(p1,GPIO.HIGH)
        GPIO.output(p2,GPIO.LOW)
        GPIO.output(p3,GPIO.LOW)
        GPIO.output(p4,GPIO.HIGH)
        time.sleep(0.001)

    if horizontal_counter==7:
        horizontal_counter=0
    else:
        horizontal_counter=horizontal_counter+1

def vertical_down_motion(p1, p2, p3, p4, magnitude):
    """Vertical stepper motor moves downwards by a certain count. 

    Parameters:
    p1 (int): Stepper motor signal pin1
    p2 (int): Stepper motor signal pin2
    p3 (int): Stepper motor signal pin3
    p4 (int): Stepper motor signal pin4
    magnitude (int): Stepper motor total move down count

   """
    global vertical_counter, vertical_current_pos

    for stepmotor_turns in range (magnitude):
#         delay = (magnitude - stepmotor_turns)/magnitude*0.01
#         
#         if delay<0.005:
#             delay = 0.005
        delay = 0.005
            
        if vertical_counter==0:
            GPIO.output(p1,GPIO.HIGH)
            GPIO.output(p2,GPIO.LOW)
            GPIO.output(p3,GPIO.LOW)
            GPIO.output(p4,GPIO.LOW)
            time.sleep(delay)
        #time.sleep(1)
        elif vertical_counter==1:
            GPIO.output(p1,GPIO.HIGH)
            GPIO.output(p2,GPIO.HIGH)
            GPIO.output(p3,GPIO.LOW)
            GPIO.output(p4,GPIO.LOW)
            time.sleep(delay)
        #time.sleep(1)
        elif vertical_counter==2:  
            GPIO.output(p1,GPIO.LOW)
            GPIO.output(p2,GPIO.HIGH)
            GPIO.output(p3,GPIO.LOW)
            GPIO.output(p4,GPIO.LOW)
            time.sleep(delay)

        elif vertical_counter==3:    
            GPIO.output(p1,GPIO.LOW)
            GPIO.output(p2,GPIO.HIGH)
            GPIO.output(p3,GPIO.HIGH)
            GPIO.output(p4,GPIO.LOW)
            time.sleep(delay)

        elif vertical_counter==4:  
            GPIO.output(p1,GPIO.LOW)
            GPIO.output(p2,GPIO.LOW)
            GPIO.output(p3,GPIO.HIGH)
            GPIO.output(p4,GPIO.LOW)
            time.sleep(delay)

        elif vertical_counter==5:
            GPIO.output(p1,GPIO.LOW)
            GPIO.output(p2,GPIO.LOW)
            GPIO.output(p3,GPIO.HIGH)
            GPIO.output(p4,GPIO.HIGH)
            time.sleep(delay)

        elif vertical_counter==6:    
            GPIO.output(p1,GPIO.LOW)
            GPIO.output(p2,GPIO.LOW)
            GPIO.output(p3,GPIO.LOW)
            GPIO.output(p4,GPIO.HIGH)
            time.sleep(delay)

        elif vertical_counter==7:    
            GPIO.output(p1,GPIO.HIGH)
            GPIO.output(p2,GPIO.LOW)
            GPIO.output(p3,GPIO.LOW)
            GPIO.output(p4,GPIO.HIGH)
            time.sleep(delay)

        if vertical_counter==7:
            vertical_counter=0
        else:
            vertical_counter=vertical_counter+1
            
        vertical_current_pos = vertical_current_pos - 1
        
def vertical_up_motion(p1, p2, p3, p4, magnitude):
    """Vertical stepper motor moves upwards by a certain count. 

    Parameters:
    p1 (int): Stepper motor signal pin1
    p2 (int): Stepper motor signal pin2
    p3 (int): Stepper motor signal pin3
    p4 (int): Stepper motor signal pin4
    magnitude (int): Stepper motor total move up count

   """
    global vertical_counter, vertical_current_pos

    for stepmotor_turns in range (magnitude):
            
        delay = 0.003
            
        if vertical_counter==0:
            GPIO.output(p1,GPIO.HIGH)
            GPIO.output(p2,GPIO.LOW)
            GPIO.output(p3,GPIO.LOW)
            GPIO.output(p4,GPIO.LOW)
            time.sleep(delay)

        elif vertical_counter==1:
            GPIO.output(p1,GPIO.HIGH)
            GPIO.output(p2,GPIO.HIGH)
            GPIO.output(p3,GPIO.LOW)
            GPIO.output(p4,GPIO.LOW)
            time.sleep(delay)

        elif vertical_counter==2:  
            GPIO.output(p1,GPIO.LOW)
            GPIO.output(p2,GPIO.HIGH)
            GPIO.output(p3,GPIO.LOW)
            GPIO.output(p4,GPIO.LOW)
            time.sleep(delay)

        elif vertical_counter==3:    
            GPIO.output(p1,GPIO.LOW)
            GPIO.output(p2,GPIO.HIGH)
            GPIO.output(p3,GPIO.HIGH)
            GPIO.output(p4,GPIO.LOW)
            time.sleep(delay)

        elif vertical_counter==4:  
            GPIO.output(p1,GPIO.LOW)
            GPIO.output(p2,GPIO.LOW)
            GPIO.output(p3,GPIO.HIGH)
            GPIO.output(p4,GPIO.LOW)
            time.sleep(delay)

        elif vertical_counter==5:
            GPIO.output(p1,GPIO.LOW)
            GPIO.output(p2,GPIO.LOW)
            GPIO.output(p3,GPIO.HIGH)
            GPIO.output(p4,GPIO.HIGH)
            time.sleep(delay)

        elif vertical_counter==6:    
            GPIO.output(p1,GPIO.LOW)
            GPIO.output(p2,GPIO.LOW)
            GPIO.output(p3,GPIO.LOW)
            GPIO.output(p4,GPIO.HIGH)
            time.sleep(delay)

        elif vertical_counter==7:    
            GPIO.output(p1,GPIO.HIGH)
            GPIO.output(p2,GPIO.LOW)
            GPIO.output(p3,GPIO.LOW)
            GPIO.output(p4,GPIO.HIGH)
            time.sleep(delay)

        if vertical_counter==0:
            vertical_counter=7
        else:
            vertical_counter=vertical_counter-1

        vertical_current_pos = vertical_current_pos + 1

# def vertical_up_motion(p1, p2, p3, p4):
#     """Stepper motor reverse motion. Counter decrease by 1.
# 
#     Parameters:
#     i (int): Current motion state
# 
#     Returns:
#     i (int): Next motion state
# 
#    """
#     global vertical_counter
#     if vertical_counter==0:
#         GPIO.output(p1,GPIO.HIGH)
#         GPIO.output(p2,GPIO.LOW)
#         GPIO.output(p3,GPIO.LOW)
#         GPIO.output(p4,GPIO.LOW)
#         time.sleep(0.003)
# 
#     elif vertical_counter==1:
#         GPIO.output(p1,GPIO.HIGH)
#         GPIO.output(p2,GPIO.HIGH)
#         GPIO.output(p3,GPIO.LOW)
#         GPIO.output(p4,GPIO.LOW)
#         time.sleep(0.003)
# 
#     elif vertical_counter==2:  
#         GPIO.output(p1,GPIO.LOW)
#         GPIO.output(p2,GPIO.HIGH)
#         GPIO.output(p3,GPIO.LOW)
#         GPIO.output(p4,GPIO.LOW)
#         time.sleep(0.003)
# 
#     elif vertical_counter==3:    
#         GPIO.output(p1,GPIO.LOW)
#         GPIO.output(p2,GPIO.HIGH)
#         GPIO.output(p3,GPIO.HIGH)
#         GPIO.output(p4,GPIO.LOW)
#         time.sleep(0.003)
# 
#     elif vertical_counter==4:  
#         GPIO.output(p1,GPIO.LOW)
#         GPIO.output(p2,GPIO.LOW)
#         GPIO.output(p3,GPIO.HIGH)
#         GPIO.output(p4,GPIO.LOW)
#         time.sleep(0.003)
# 
#     elif vertical_counter==5:
#         GPIO.output(p1,GPIO.LOW)
#         GPIO.output(p2,GPIO.LOW)
#         GPIO.output(p3,GPIO.HIGH)
#         GPIO.output(p4,GPIO.HIGH)
#         time.sleep(0.003)
# 
#     elif vertical_counter==6:    
#         GPIO.output(p1,GPIO.LOW)
#         GPIO.output(p2,GPIO.LOW)
#         GPIO.output(p3,GPIO.LOW)
#         GPIO.output(p4,GPIO.HIGH)
#         time.sleep(0.003)
# 
#     elif vertical_counter==7:    
#         GPIO.output(p1,GPIO.HIGH)
#         GPIO.output(p2,GPIO.LOW)
#         GPIO.output(p3,GPIO.LOW)
#         GPIO.output(p4,GPIO.HIGH)
#         time.sleep(0.003)
# 
#     if vertical_counter==0:
#         vertical_counter=7
#     else:
#         vertical_counter=vertical_counter-1


def relax_motion(p1, p2, p3, p4, vertical = True):
    
    """Stepper motor off mode. Motor not energised.

    Parameters:
    p1 (int): Stepper motor signal pin1
    p2 (int): Stepper motor signal pin2
    p3 (int): Stepper motor signal pin3
    p4 (int): Stepper motor signal pin4
    vertical (boolean): True: vertical or False: horizontal

   """
    if vertical == True:
        time.sleep(0.1)
    GPIO.output(p1,GPIO.LOW) 
    GPIO.output(p2,GPIO.LOW)
    GPIO.output(p3,GPIO.LOW)
    GPIO.output(p4,GPIO.LOW)
    
    if vertical == True:
        time.sleep(0.1)
    
def hold_motion():
    """Stepper motor holding mode. Motor maintains current position."""
    
    global vertical_counter
    
    vertical_counter = vertical_counter - vertical_counter%2
    
    if vertical_counter==0:
        GPIO.output(out1,GPIO.HIGH)
        GPIO.output(out2,GPIO.LOW)
        GPIO.output(out3,GPIO.LOW)
        GPIO.output(out4,GPIO.LOW)

    elif vertical_counter==1:
        GPIO.output(out1,GPIO.HIGH)
        GPIO.output(out2,GPIO.HIGH)
        GPIO.output(out3,GPIO.LOW)
        GPIO.output(out4,GPIO.LOW)

    elif vertical_counter==2:  
        GPIO.output(out1,GPIO.LOW)
        GPIO.output(out2,GPIO.HIGH)
        GPIO.output(out3,GPIO.LOW)
        GPIO.output(out4,GPIO.LOW)

    elif vertical_counter==3:    
        GPIO.output(out1,GPIO.LOW)
        GPIO.output(out2,GPIO.HIGH)
        GPIO.output(out3,GPIO.HIGH)
        GPIO.output(out4,GPIO.LOW)

    elif vertical_counter==4:  
        GPIO.output(out1,GPIO.LOW)
        GPIO.output(out2,GPIO.LOW)
        GPIO.output(out3,GPIO.HIGH)
        GPIO.output(out4,GPIO.LOW)

    elif vertical_counter==5:
        GPIO.output(out1,GPIO.LOW)
        GPIO.output(out2,GPIO.LOW)
        GPIO.output(out3,GPIO.HIGH)
        GPIO.output(out4,GPIO.HIGH)

    elif vertical_counter==6:    
        GPIO.output(out1,GPIO.LOW)
        GPIO.output(out2,GPIO.LOW)
        GPIO.output(out3,GPIO.LOW)
        GPIO.output(out4,GPIO.HIGH)

    elif vertical_counter==7:    
        GPIO.output(out1,GPIO.HIGH)
        GPIO.output(out2,GPIO.LOW)
        GPIO.output(out3,GPIO.LOW)
        GPIO.output(out4,GPIO.HIGH)
    
def horizontal_shake(p1, p2, p3, p4, magnitude, shake_time = 10):
    """Horizontal stepper motor shaking mode.Can input the shake movement magnitude and shake times.

    Parameters:
    p1 (int): Stepper motor signal pin1
    p2 (int): Stepper motor signal pin2
    p3 (int): Stepper motor signal pin3
    p4 (int): Stepper motor signal pin4
    magnitude (int): Shaking magnitude
    shake_time (int): Shake motion total repetition times

   """

    global horizontal_counter
    
    for shake_counter in range (shake_time):
        for stepmotor_turns in range (magnitude):
            horizontal_right_motion(p1, p2, p3, p4)
            
        relax_motion(p1, p2, p3, p4, vertical = False)
        time.sleep(0.5)
        
        for stepmotor_turns in range (magnitude):
            horizontal_left_motion(p1, p2, p3, p4)
            
        relax_motion(p1, p2, p3, p4, vertical = False)
        
        time.sleep(0.5)
        
def horizontal_right_motion_count(p1, p2, p3, p4, magnitude):
    """Horizontal stepper motor moves to the right for a certain counts. 

    Parameters:
    p1 (int): Stepper motor signal pin1
    p2 (int): Stepper motor signal pin2
    p3 (int): Stepper motor signal pin3
    p4 (int): Stepper motor signal pin4
    magnitude (int): Stepper motor total move right count

   """

    global horizontal_counter
    
    for stepmotor_turns in range (magnitude):
        horizontal_right_motion(p1, p2, p3, p4)
        
    relax_motion(p1, p2, p3, p4, vertical = False)
    
def horizontal_left_motion_count(p1, p2, p3, p4, magnitude):
    """Horizontal stepper motor moves to the left for a certain counts. 

    Parameters:
    p1 (int): Stepper motor signal pin1
    p2 (int): Stepper motor signal pin2
    p3 (int): Stepper motor signal pin3
    p4 (int): Stepper motor signal pin4
    magnitude (int): Stepper motor total move left count

   """
    global horizontal_counter
    
    for stepmotor_turns in range (magnitude):
        horizontal_left_motion(p1, p2, p3, p4)
        
    relax_motion(p1, p2, p3, p4, vertical = False)
        
def sensor_flag_reset():
    """Reset all the sensor flags. All flag values will be set to 0."""
    
    global sensor_1_flag, sensor_2_flag, sensor_3_flag, sensor_4_flag, sensor_5_flag
    
    sensor_1_flag = 0
    sensor_2_flag = 0
    sensor_3_flag = 0
    sensor_4_flag = 0
    sensor_5_flag = 0

def create_new_directory():
    """Create new file directory whenever a new operation started.

    Returns:
    current_dir (str): directory of the current execution file
    original_dir (str): directory for raw image
    processed_dir (str): directory for framed image

   """
    
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d-%m-%Y %H:%M")
    current_dir = os.getcwd()
    dir_counter = 0

    while True:
        final_directory = os.path.join(current_dir, dt_string)
        if dir_counter != 0:
            final_directory = final_directory + '(' + str(dir_counter) + ')'
            
        if not os.path.exists(final_directory):
            os.makedirs(final_directory)
            original_dir = os.path.join(final_directory, r'Original')
            processed_dir = os.path.join(final_directory, r'Framed')
            plot_dir = os.path.join(final_directory, r'Plots')
            os.makedirs(original_dir)
            os.makedirs(processed_dir)
            os.makedirs(plot_dir)
            break
        else:
            dir_counter+=1
        
    return current_dir, original_dir, processed_dir, plot_dir
    
def uv_led_on():
    """Turn on the 11V powered UV LED."""
    
    GPIO.output(uv_led,GPIO.LOW)
    
def uv_led_off():
    """Turn off the 11V powered UV LED."""
    
    GPIO.output(uv_led,GPIO.HIGH)

def sensor_1_func(channel):
    """Sensor 1 callback function. Will set the sensor flag to 1."""

    global sensor_1_flag
    sensor_1_flag = 1
    print("\rsensor_1", end='');
#     GPIO.remove_event_detect(sensor1)
    
def sensor_2_func(channel):
    """Sensor 2 callback function. Will set the sensor flag to 1."""

    global sensor_2_flag
    sensor_2_flag = 1
    print("\rsensor_2", end='');
#     GPIO.remove_event_detect(sensor2)
    
def sensor_3_func(channel):
    """Sensor 3 callback function. Will set the sensor flag to 1."""

    global sensor_3_flag
    sensor_3_flag = 1
    print("\rsensor_3", end='');
#     GPIO.remove_event_detect(sensor3)
    
def sensor_4_func(channel):
    """Sensor 4 callback function. Will set the sensor flag to 1."""

    global sensor_4_flag
    sensor_4_flag = 1
    print("\rsensor_4", end='');
#     GPIO.remove_event_detect(sensor4)
    
def sensor_5_func(channel):
    """Sensor 5 callback function. Will set the sensor flag to 1."""

    global sensor_5_flag
    sensor_5_flag = 1
    print("\rsensor_5", end='');
#     GPIO.remove_event_detect(sensor5)