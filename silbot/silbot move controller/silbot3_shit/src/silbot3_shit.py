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
    try:
        rospy.init_node('gptpy', anonymous=True)
        gesture_pub = rospy.Publisher('gpttopic', String, queue_size=1000)
        
        HOST = '0.0.0.0'
        PORT = 7790
        
        # Create socket without 'with' statement
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        print("Listening for Docker connection on {}:{}".format(HOST, PORT))
        
        conn, addr = s.accept()
        print("Connected by {}".format(addr))
        
        while True:
            data = conn.recv(1024)
            if not data:
                break
            message = data.decode('utf-8').lower()
            print("Received full response: {}".format(message))
            
            if "hello" in message or "hi" in message:
                gesture_pub.publish("wave")
            elif "i think" in message or "i believe" in message:
                gesture_pub.publish("Think")
            else:
                gesture_pub.publish("Speak_Start")
                
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print("Error: {}".format(e))
    finally:
        # Clean up
        try:
            conn.close()
        except:
            pass
        try:
            s.close()
        except:
            pass