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
                conn, addr = s.accept()
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                # conn.settimeout(1.0)  # ❌ REMOVE/COMMENT OUT THIS LINE
                print("Connected by {}".format(addr))

                # Inner loop: Handle communication with the current client
                while not rospy.is_shutdown():
                    # Wrap in try/except to handle socket errors, but let recv block
                    try:
                        data = conn.recv(1024)
                        # ✅ CRITICAL FIX: Check for zero bytes (client disconnect)
                        if not data:
                            print("Client {} disconnected.".format(addr))
                            break # Exit inner loop
                        message = data.decode('utf-8').lower()
                        print("Received full response: {}".format(message))

                        # Handle the handshake message sent by the client
                        if message == "client_connected_ok":
                            print("Handshake received. Connection confirmed and waiting for real data.")
                            continue # Skip to the next recv() without publishing a gesture
                        
                        if "hello" in message or "hi" in message:
                            print("received hello or hi")
                            gesture_pub.publish("wave")
                        elif "think" in message or "believe" in message:
                            gesture_pub.publish("Think")
                            print("received i think")
                        #else:
                            #gesture_pub.publish("Speak_Start")
                            
                    except socket.timeout:
                        # This is normal - no data received within timeout period
                        # Just continue the loop to check for new data or shutdown
                        continue
                    except socket.error as e:
                        print("Socket error during communication: {}".format(e))
                        break
                        
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
                print("Connection closed, waiting for new connections...")
        
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print("Initialization Error: {}".format(e))
    finally:
        # Final cleanup of the main listening socket
        if 's' in locals() and s:
            s.close()