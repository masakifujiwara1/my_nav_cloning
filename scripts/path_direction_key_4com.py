#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import select
import tty
import termios
import random

from cv2 import SUBDIV2D_NEXT_AROUND_RIGHT
from std_srvs.srv import SetBool, SetBoolResponse
from std_srvs.srv import Empty
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Int8MultiArray
from nav_msgs.msg import Path
from std_srvs.srv import Trigger
from std_msgs.msg import Int8
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Twist
from skimage.transform import resize
from nav_cloning_net_cross import *
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import rospy
import roslib
roslib.load_manifest('nav_cloning')

msg = """
Control path direction!
--------------------------
        w
    a   s   d
        x

w : Input straight direction [0 100 0 0]
a : Input left direction [0 0 100 0]
s : Moving stop
d : Input right direction [0 0 0 100]
x : Moving start

q : Exit
"""

err = """
Communications Failed
"""


def getKey():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def outdata(data):
    data = "Input_data : " + str(data.data)
    return data


if __name__ == "__main__":
    settings = termios.tcgetattr(sys.stdin)
    rospy.init_node('path_direction')
    pub = rospy.Publisher('cmd_data_key', Int8MultiArray, queue_size=1)
    f_pub = rospy.Publisher('running_flag', Int8, queue_size=1)

    data = [
        [100, 0, 0, 0],
        [0, 100, 0, 0],
        [0, 0, 100, 0],
        [0, 0, 0, 100]
    ]
    flag = 1
    ldata = [0, 0, 100, 0]
    rdata = [0, 0, 0, 100]
    cdata = [100, 0, 0, 0]
    sdata = [0, 100, 0, 0]
    pubdata = Int8MultiArray()

    STRAIGHT_MODE = True

    try:
        print(msg)
        while(1):
            key = getKey()
            if key == 'a':
                pubdata.data = ldata
                print(outdata(pubdata))
            elif key == 'd':
                pubdata.data = rdata
                print(outdata(pubdata))
            elif key == 'w':
                pubdata.data = sdata
                print(outdata(pubdata))
            elif key == 's':
                flag = 0
            elif key == 'x':
                flag = 1
            elif key == 'q':
                break
            else:
                if STRAIGHT_MODE:
                    pubdata.data = cdata
                else:
                    rand = random.randint(0, 2)
                    pubdata.data = data[rand]
                print(outdata(pubdata))
            pub.publish(pubdata)
            f_pub.publish(flag)
    except:
        print(err)

    finally:
        print(err)
