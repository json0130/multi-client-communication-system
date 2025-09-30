#!/usr/bin/env python

import random
import time
import rospy
from std_msgs.msg import String
from std_msgs.msg import Int32
from playsound import playsound
import sys
import os
import socket
import struct
import select
sys.path.append('/home/silbot3/gptpy2')

from gptpy2.chat_server import ChatServer
from gptpy2.stt import RecordAudio

if __name__ == '__main__':
    rospy.init_node('gptpy', anonymous=True)
    gesture_pub = rospy.Publisher('gpttopic', String, queue_size=1000)
    # gesture_sub = rospy.Subscriber('gesturetopic', Int32, thunk_callback)
    
    # Set up a server on the host to listen for the Docker container
    HOST = '0.0.0.0' # Listen on all available network interfaces
    PORT = 7790      # Choose a new port for this communication

    
    # while True:
    #     gesture_pub.publish("Think")
    #     time.sleep(3)
    #     gesture_pub.publish("wave")
        
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Listening for Docker connection on {HOST}:{PORT}")
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                message = data.decode('utf-8')
                if message == "Think":
                    gesture_pub.publish("Think")
                elif message == "wave":
                    gesture_pub.publish("wave")
                # Add more logic here to handle other commands