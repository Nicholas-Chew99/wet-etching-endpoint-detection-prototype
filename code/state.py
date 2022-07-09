import motor_function as motor
import RPi.GPIO as GPIO
import time
    

def initialisation():
    """Setup all the edge detection pin. """
    
    motor.sensor_flag_reset()
    
    GPIO.remove_event_detect(motor.sensor5)
    GPIO.remove_event_detect(motor.sensor1)
    GPIO.remove_event_detect(motor.sensor2)
    GPIO.remove_event_detect(motor.sensor3)
    GPIO.remove_event_detect(motor.sensor4)

    GPIO.add_event_detect(motor.sensor5, GPIO.FALLING, callback=motor.sensor_5_func, bouncetime=1000) 
    GPIO.add_event_detect(motor.sensor1, GPIO.FALLING, callback=motor.sensor_1_func, bouncetime=2000) 
    GPIO.add_event_detect(motor.sensor2, GPIO.FALLING, callback=motor.sensor_2_func, bouncetime=2000) 
    GPIO.add_event_detect(motor.sensor3, GPIO.FALLING, callback=motor.sensor_3_func, bouncetime=2000) 
    GPIO.add_event_detect(motor.sensor4, GPIO.FALLING, callback=motor.sensor_4_func, bouncetime=2000) 
    
    # flag = 0
    # time.sleep(7)
    print("\nInitializing motor variables...")
    
def remove_event():
    """Remove the event detection. Prepare for next iteration. """
 
    GPIO.remove_event_detect(motor.sensor5)
    GPIO.remove_event_detect(motor.sensor1)
    GPIO.remove_event_detect(motor.sensor2)
    GPIO.remove_event_detect(motor.sensor3)
    GPIO.remove_event_detect(motor.sensor4)
    
    # flag = 0
    # time.sleep(7)
    print("\nEvent detection removed. Ready for next iteration.")

def original_pos_check():
    """Check whether or not the handle is initially located at the loading position."""

    print("\nChecking handle current position...")

    if GPIO.input(motor.sensor5):
        print("\nMoving back to the loading station...")
        
        motor.vertical_up_motion(27, 17, 22, 18, 1200)
        
        while True:
            if motor.sensor_5_flag != 1:
        #         motor_counter2 = motor.reverse_motion(motor_counter2, 9, 25, 11, 8) # horizontal right motion 
                motor.horizontal_left_motion(9, 25, 11, 8)
                
            else:
#                 print("hello")
                motor.relax_motion(9, 25, 11, 8, vertical = False)
                motor.vertical_down_motion(27, 17, 22, 18, 1200)
                break

    motor.relax_motion(27, 17, 22, 18)
    motor.sensor_flag_reset()
    
def station_1():
    """Moves towards station 1 - aluminum etchant. """

    print("\nMoving to first station... ")
    motor.vertical_up_motion(27, 17, 22, 18, 1200)
        
    motor.hold_motion()

    while True:
        if motor.sensor_1_flag != 1:
    #         motor_counter2 = motor.reverse_motion(motor_counter2, 9, 25, 11, 8) # horizontal right motion 
            motor.horizontal_right_motion(9, 25, 11, 8)
            
        else:
            print("\nstation_1")
            motor.relax_motion(9, 25, 11, 8, vertical = False)
            break


    motor.vertical_down_motion(27, 17, 22, 18, 1200)
    time.sleep(0.1)
    motor.relax_motion(27, 17, 22, 18)
    time.sleep(0.1)
    motor.relax_motion(9, 25, 11, 8, vertical = False)
# 
# 
# time.sleep(10)
# 
# # go to next location
#
def station_2():
    """Moves towards station 2 - water bath. """

    motor.sensor_flag_reset()

    motor.vertical_up_motion(27, 17, 22, 18, 1200)
        
    motor.hold_motion()

    while True:
        if motor.sensor_2_flag != 1:
    #         motor_counter2 = motor.reverse_motion(motor_counter2, 9, 25, 11, 8) # horizontal right motion 
            motor.horizontal_right_motion(9, 25, 11, 8)
            
        else:
            motor.relax_motion(9, 25, 11, 8, vertical = False)
            print("\nstation_2")
            break

    motor.vertical_down_motion(27, 17, 22, 18, 1200)
    motor.relax_motion(27, 17, 22, 18)
    motor.horizontal_shake(9, 25, 11, 8, 100, 24)
# 
# 
# time.sleep(20)
# # go to next location
#
def station_3():
    """Moves towards station 3 - Acetone bath. """
    motor.sensor_flag_reset()
    motor.vertical_up_motion(27, 17, 22, 18, 1200)
        
    motor.hold_motion()

    while True:
        if motor.sensor_3_flag != 1:
    #         motor_counter2 = motor.reverse_motion(motor_counter2, 9, 25, 11, 8) # horizontal right motion 
            motor.horizontal_right_motion(9, 25, 11, 8)
            
        else:
            motor.relax_motion(9, 25, 11, 8, vertical = False)
            print("\nstation_3")
            break

    motor.vertical_down_motion(27, 17, 22, 18, 1200)
        
    motor.relax_motion(27, 17, 22, 18)

    motor.horizontal_shake(9, 25, 11, 8, 100, 24)
# 
# time.sleep(20)
# # go to next location
#

def station_4():
    """Moves towards station 4 - IPA bath. """
    motor.sensor_flag_reset()
    motor.vertical_up_motion(27, 17, 22, 18, 1200)
        
    motor.hold_motion()

    while True:
        if motor.sensor_4_flag != 1: 
            motor.horizontal_right_motion(9, 25, 11, 8)
            
        else:
            motor.relax_motion(9, 25, 11, 8, vertical = False)
            print("\nstation_4")
            break

    motor.horizontal_right_motion_count(9, 25, 11, 8, 600)

    motor.vertical_down_motion(27, 17, 22, 18, 1200)
        
    motor.relax_motion(27, 17, 22, 18)
    motor.relax_motion(9, 25, 11, 8, vertical = False)

    motor.horizontal_shake(9, 25, 11, 8, 100, 24)
# 
# time.sleep(20)
#

def back_to_loading_pos():
    """Moves towards original loading station for recollection. """
    
    motor.vertical_up_motion(27, 17, 22, 18, 1200)
        
    motor.hold_motion()
    motor.sensor_flag_reset()

    while True:
        if motor.sensor_5_flag != 1:
    #         motor_counter2 = motor.reverse_motion(motor_counter2, 9, 25, 11, 8) # horizontal right motion 
            motor.horizontal_left_motion(9, 25, 11, 8)
            
        else:
            print("")
            motor.relax_motion(9, 25, 11, 8, vertical = False)
            break


    motor.vertical_down_motion(27, 17, 22, 18, 1200)
        
    motor.relax_motion(27, 17, 22, 18)
    motor.relax_motion(9, 25, 11, 8, vertical = False)

def emergency_stop_motion():
    
    motor.vertical_up_motion(27, 17, 22, 18, 1200 - motor.vertical_current_pos)
    motor.hold_motion()
    motor.sensor_flag_reset()
    
    while True:
        if motor.sensor_5_flag != 1:
    #         motor_counter2 = motor.reverse_motion(motor_counter2, 9, 25, 11, 8) # horizontal right motion 
            motor.horizontal_left_motion(9, 25, 11, 8)
            
        else:
            print("")
            motor.relax_motion(9, 25, 11, 8, vertical = False)
            break


    motor.vertical_down_motion(27, 17, 22, 18, 1200)
        
    motor.relax_motion(27, 17, 22, 18)
    motor.relax_motion(9, 25, 11, 8, vertical = False)
