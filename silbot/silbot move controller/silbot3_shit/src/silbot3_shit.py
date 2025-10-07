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

class GestureController:
    def __init__(self):
        # Initialize ROS node with unique name
        rospy.init_node('gptpy_gesture', anonymous=True)
        
        # Create publisher with small queue like the reference code
        self.gesture_pub = rospy.Publisher('gpttopic', String, queue_size=1)
        
        # Socket setup
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        HOST = '0.0.0.0'
        PORT = 7790
        
        self.sock.bind((HOST, PORT))
        self.sock.listen(1)
        print(f"Waiting for connection on {HOST}:{PORT}")
        
        # Accept first connection (blocking)
        self.conn, addr = self.sock.accept()
        print(f"Connected to {addr}")
        
        self.last_gesture = None
        self.last_gesture_time = time.time()
        self.GESTURE_COOLDOWN = 3.0

    def process_message(self, message):
        current_time = time.time()
        
        # Only process if cooldown has passed
        if current_time - self.last_gesture_time >= self.GESTURE_COOLDOWN:
            # First publish Think gesture like reference code
            if len(message) > 0:
                print("Processing message...")
                self.gesture_pub.publish("Think")
                rospy.sleep(1)  # Give time for think gesture
                
                # Then publish appropriate gesture
                if "hello" in message.lower() or "hi" in message.lower():
                    self.gesture_pub.publish("wave")
                elif "i think" in message.lower() or "i believe" in message.lower():
                    self.gesture_pub.publish("Think")
                else:
                    self.gesture_pub.publish("Speak_Start")
                
                # Update tracking
                self.last_gesture_time = current_time
                print("Processing complete")

    def run(self):
        try:
            while not rospy.is_shutdown():
                # Wait for message with timeout
                ready = select.select([self.conn], [], [], 0.1)
                if ready[0]:
                    data = self.conn.recv(1024)
                    if not data:
                        break
                    
                    message = data.decode('utf-8')
                    print(f"Received: {message}")
                    self.process_message(message)
                
                rospy.sleep(0.1)  # Small sleep to prevent CPU hogging
                
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        print("Cleaning up...")
        if hasattr(self, 'conn'):
            self.conn.close()
        self.sock.close()

if __name__ == '__main__':
    controller = GestureController()
    controller.run()