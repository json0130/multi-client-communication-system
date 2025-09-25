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
    while True:
        gesture_pub.publish("Think")
        time.sleep(3)
        gesture_pub.publish("wave")