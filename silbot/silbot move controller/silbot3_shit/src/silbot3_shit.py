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
        
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print("Listening for Docker connection on {}:{}".format(HOST, PORT))
        
        # Outer loop: Keep listening for new connections
        while not rospy.is_shutdown():
            conn = None # Initialize conn for error handling
            try:
                # Blocks until a client connects
                conn, addr = s.accept()
                print("Connected by {}".format(addr))
                
                # Inner loop: Handle communication with the current client
                while not rospy.is_shutdown():
                    data = conn.recv(1024)
                    
                    # If data is empty (b''), the client has disconnected
                    if not data:
                        print("Client {} disconnected.".format(addr))
                        break # Exit inner loop, go back to s.accept()
                        
                    message = data.decode('utf-8').lower()
                    print("Received full response: {}".format(message))
                    
                    if "hello" in message or "hi" in message:
                        gesture_pub.publish("wave")
                    elif "i think" in message or "i believe" in message:
                        gesture_pub.publish("Think")
                    else:
                        gesture_pub.publish("Speak_Start")
                        
            except KeyboardInterrupt:
                # Allow outer loop to break cleanly
                raise
            except socket.error as e:
                print("Socket error during communication: {}".format(e))
            except Exception as e:
                print("General error during communication: {}".format(e))
            finally:
                # Close the connection with the client before accepting a new one
                if conn:
                    conn.close()
                    conn = None
        
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print("Initialization Error: {}".format(e))
    finally:
        # Final cleanup of the main listening socket
        if 's' in locals() and s:
            s.close()