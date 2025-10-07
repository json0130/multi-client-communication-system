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

if __name__ == '__main__':
    try:
        # Initialize ROS node
        rospy.init_node('gptpy', anonymous=True)
        gesture_pub = rospy.Publisher('gpttopic', String, queue_size=1)  # Reduce queue to 1
        
        # Socket setup
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        HOST = '0.0.0.0'
        PORT = 7790
        
        sock.bind((HOST, PORT))
        sock.listen(1)
        
        last_gesture = None  # Changed from empty string
        last_gesture_time = time.time()
        GESTURE_COOLDOWN = 3.0
        is_connected = False  # Flag to track connection status

        while not rospy.is_shutdown():
            if not is_connected:
                try:
                    # Set socket to non-blocking for accept
                    sock.setblocking(False)
                    conn, addr = sock.accept()
                    conn.setblocking(True)  # Set connection back to blocking
                    is_connected = True
                except socket.error:
                    # No connection yet, sleep and continue
                    rospy.sleep(1)
                    continue

            try:
                if is_connected:
                    ready = select.select([conn], [], [], 0.1)
                    
                    if ready[0]:
                        data = conn.recv(1024)
                        if not data:
                            # Connection closed
                            is_connected = False
                            conn.close()
                            continue
                        
                        message = data.decode('utf-8').lower()
                        current_time = time.time()
                        
                        if current_time - last_gesture_time >= GESTURE_COOLDOWN:
                            # Only process gestures if enough time has passed
                            gesture = None  # Default to no gesture
                            
                            if "hello" in message or "hi" in message:
                                gesture = "wave"
                            elif "i think" in message or "i believe" in message:
                                gesture = "Think"
                            elif len(message) > 0:  # Only do Speak_Start if there's actual content
                                gesture = "Speak_Start"
                            
                            if gesture and gesture != last_gesture:
                                gesture_pub.publish(String(gesture))
                                last_gesture = gesture
                                last_gesture_time = current_time
                    
                    rospy.sleep(0.1)  # Small sleep to prevent CPU hogging
                    
            except Exception as e:
                # Handle connection errors
                is_connected = False
                try:
                    conn.close()
                except:
                    pass
                rospy.sleep(1)
                
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        try:
            sock.close()
        except:
            pass
        sys.exit(0)