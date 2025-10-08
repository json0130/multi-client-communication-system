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

class GptpyClass:

    def __init__(self):

        def thunk_callback(msg):
            print("received")
            pass

        rospy.init_node('gptpy', anonymous=True)
        self.gesture_pub = rospy.Publisher('gpttopic', String, queue_size=1000)
        self.gesture_sub = rospy.Subscriber('gesturetopic', Int32, thunk_callback)

        self.gptserver = ChatServer(host="192.168.0.100", port=7788)
        self.gptserver.start()

        (HOST,PORT) = ("192.168.0.100",7789)
        self.sttserver = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sttserver.settimeout(600)
        self.sttserver.bind((HOST, PORT))
        self.sttserver.listen(1)
        self.conn, addr = self.sttserver.accept()
        print("Connected to " + str(addr))


if __name__ == '__main__':
    gptpy = GptpyClass()
    initprompt = "You are a receptionist robot working in the CARES lab for The University of Auckland, your name is Silbot. The next prompt will be the start of the conversation"
    gptpy.gptserver.send(initprompt)           # Send initial prompt to ChatGPT

    # Wait until a connection from the client is made
    while (gptpy.gptserver.halt_flag == 1):

        pass

    while (True):
        try:
            # gptpy.thunk_var = 0
            # prompt = raw_input("Enter prompt: ")    # User input (IN CASE SPEECH-TO-TEXT NO LONGER WORKS)
            prev_audio_mtime = os.stat("/home/silbot3/tts/output.mp3").st_mtime # Record current TTS audio file metadata
            ready = select.select([gptpy.conn], [], [], 600)    # Wait 10 minutes for an input before timout
            if ready[0]:
                buf = gptpy.conn.recv(1024)
            if not buf:
                break

            prompt = buf.decode('utf8')
            print(prompt)
            print("\nProcessing...\n")

            gptpy.gesture_pub.publish("Think")  # Initiate "Think" gesture
            gptpy.gptserver.send(prompt)           # Send prompt to ChatGPT

            # # Wait until TTS audio is updated and silbot has finished a motion
            while(prev_audio_mtime == os.stat("/home/silbot3/tts/output.mp3").st_mtime):
                pass

            gptpy.gesture_pub.publish("Think_finish")
            playsound('/home/silbot3/tts/output.mp3')
            gptpy.conn.sendall(b'0')
        except KeyboardInterrupt:
            break

    gptpy.gptserver.join()
