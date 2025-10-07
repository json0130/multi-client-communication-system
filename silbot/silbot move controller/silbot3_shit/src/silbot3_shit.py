#!/usr/bin/env python

import random
import time
import rospy
from std_msgs.msg import String
from std_msgs.msg import Int32
import sys
import os
import socket
import struct
import select
sys.path.append('/home/silbot3/gptpy2')

def log_info(msg):
    """Helper function for consistent logging"""
    rospy.loginfo(f"[SILBOT3] {msg}")
    print(f"[SILBOT3] {msg}")  # Print to terminal for direct visibility

if __name__ == '__main__':
    try:
        # Initialize ROS node
        rospy.init_node('gptpy', anonymous=True)
        log_info("ROS Node initialized")
        
        # Setup publisher
        gesture_pub = rospy.Publisher('gpttopic', String, queue_size=1000)
        log_info("Gesture publisher created")
        
        # Socket setup
        HOST = '0.0.0.0'
        PORT = 7790
        
        last_gesture = ""
        last_gesture_time = time.time()
        GESTURE_COOLDOWN = 3.0
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen()
            log_info(f"Listening for Docker connection on {HOST}:{PORT}")
            
            conn, addr = s.accept()
            log_info(f"New connection from {addr}")
            
            with conn:
                while not rospy.is_shutdown():
                    # Non-blocking socket read with timeout
                    ready = select.select([conn], [], [], 0.1)
                    
                    if ready[0]:
                        data = conn.recv(1024)
                        if not data:
                            log_info("Connection closed by client")
                            break
                        
                        message = data.decode('utf-8').lower()
                        log_info(f"Received message: {message}")
                        
                        # Gesture control with cooldown
                        current_time = time.time()
                        if current_time - last_gesture_time >= GESTURE_COOLDOWN:
                            
                            # Gesture selection logic
                            if "hello" in message or "hi" in message:
                                gesture = "wave"
                            elif "i think" in message or "i believe" in message:
                                gesture = "Think"
                            else:
                                gesture = "Speak_Start"
                            
                            # Only publish new gestures
                            if gesture != last_gesture:
                                log_info(f"Publishing gesture: {gesture}")
                                gesture_pub.publish(gesture)
                                last_gesture = gesture
                                last_gesture_time = current_time
                            else:
                                log_info(f"Skipping duplicate gesture: {gesture}")
                        else:
                            log_info(f"Gesture cooldown active. Waiting {GESTURE_COOLDOWN - (current_time - last_gesture_time):.1f}s")
                    
                    # Small sleep to prevent CPU hogging
                    rospy.sleep(0.1)
                    
    except Exception as e:
        log_info(f"❌ Error: {str(e)}")
        rospy.logerr(f"Error in silbot3_shit.py: {str(e)}")
    finally:
        log_info("Shutting down SILBOT3 node")