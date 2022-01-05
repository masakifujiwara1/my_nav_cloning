#!/usr/bin/env python
from __future__ import print_function
import copy
import time
import os
import csv
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
import cv2
import rospy
import roslib
roslib.load_manifest('nav_cloning')


class cource_following_learning_node:
    def __init__(self):
        rospy.init_node('cource_following_learning_node', anonymous=True)
        self.action_num = rospy.get_param(
            "/LiDAR_based_learning_node/action_num", 1)
        print("action_num: " + str(self.action_num))
        print("use_model.py called")
        self.dl = deep_learning(n_action=self.action_num)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "/camera/rgb/image_raw", Image, self.callback)
        self.image_left_sub = rospy.Subscriber(
            "/camera_left/rgb/image_raw", Image, self.callback_left_camera)
        self.image_right_sub = rospy.Subscriber(
            "/camera_right/rgb/image_raw", Image, self.callback_right_camera)
        self.vel_sub = rospy.Subscriber("/nav_vel", Twist, self.callback_vel)
        self.action_pub = rospy.Publisher("action", Int8, queue_size=1)
        self.nav_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.srv = rospy.Service('/training', SetBool,
                                 self.callback_dl_training)
        self.pose_sub = rospy.Subscriber(
            "/amcl_pose", PoseWithCovarianceStamped, self.callback_pose)
        self.path_sub = rospy.Subscriber(
            "/move_base/NavfnROS/plan", Path, self.callback_path)
        self.dir_cmd_sub = rospy.Subscriber(
            "/cmd_data_key", Int8MultiArray, self.callback_cmd)
        self.min_distance = 0.0
        self.action = 0.0
        self.episode = 0
        self.vel = Twist()
        self.path_pose = PoseArray()
        self.cv_image = np.zeros((480, 640, 3), np.uint8)
        self.cv_left_image = np.zeros((480, 640, 3), np.uint8)
        self.cv_right_image = np.zeros((480, 640, 3), np.uint8)
        self.learning = False
        self.select_dl = False
        self.flag = False
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.path = roslib.packages.get_pkg_dir(
            'nav_cloning') + '/data/result/'
        self.save_path = roslib.packages.get_pkg_dir(
            'nav_cloning') + '/data/model/'
        self.load_path = roslib.packages.get_pkg_dir(
            'nav_cloning') + '/data/model/20220105_03:42:26/model.net'
        self.previous_reset_time = 0
        self.start_time_s = rospy.get_time()
        os.makedirs(self.path + self.start_time)

        with open(self.path + self.start_time + '/' + 'reward.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(
                ['step', 'mode', 'loss', 'angle_error(rad)', 'distance(m)'])

    def callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_left_camera(self, data):
        try:
            self.cv_left_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_right_camera(self, data):
        try:
            self.cv_right_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_path(self, data):
        self.path_pose = data

    def callback_pose(self, data):
        distance_list = []
        pos = data.pose.pose.position

        for pose in self.path_pose.poses:
            path = pose.pose.position
            distance = np.sqrt(abs((pos.x - path.x)**2 + (pos.y - path.y)**2))
            distance_list.append(distance)

        if distance_list:
            self.min_distance = min(distance_list)

    def callback_cmd(self, data):
        self.dir_cmd_data = data.data
        self.flag = True

    def callback_vel(self, data):
        self.vel = data
        self.action = self.vel.angular.z

    def callback_dl_training(self, data):
        resp = SetBoolResponse()
        self.learning = data.data
        resp.message = "Training: " + str(self.learning)
        resp.success = True
        return resp

    def loop(self):
        if self.cv_image.size != 640 * 480 * 3:
            return
        if self.cv_left_image.size != 640 * 480 * 3:
            return
        if self.cv_right_image.size != 640 * 480 * 3:
            return

        if self.vel.linear.x == 0:
            return

        img = resize(self.cv_image, (48, 64), mode='constant')
        r, g, b = cv2.split(img)
        imgobj = np.asanyarray([r, g, b])

        img_left = resize(self.cv_left_image, (48, 64), mode='constant')
        r, g, b = cv2.split(img_left)
        imgobj_left = np.asanyarray([r, g, b])

        img_right = resize(self.cv_right_image, (48, 64), mode='constant')
        r, g, b = cv2.split(img_right)
        imgobj_right = np.asanyarray([r, g, b])

        ros_time = str(rospy.Time.now())

        if self.episode == 0:
            self.learning = False
            # self.dl.save(self.save_path)
            self.dl.load(self.load_path)
            # print("model_called")

        if self.learning:
            dir_cmd = None
            target_action = self.action
            distance = self.min_distance

            """
            # conventional method
            if distance > 0.1:
                self.select_dl = False
            elif distance < 0.05:
                self.select_dl = True
            if self.select_dl and self.episode >= 0:
                target_action = 0
            action, loss = self.dl.act_and_trains(imgobj, target_action)
            if abs(target_action) < 0.1:
                action_left,  loss_left  = self.dl.act_and_trains(imgobj_left, target_action - 0.2)
                action_right, loss_right = self.dl.act_and_trains(imgobj_right, target_action + 0.2)
            angle_error = abs(action - target_action)
            """
            """
            # proposed method (new)
            action, loss = self.dl.act_and_trains(imgobj, target_action)
            if abs(target_action) < 0.1:
                action_left,  loss_left  = self.dl.act_and_trains(imgobj_left, target_action - 0.2)
                action_right, loss_right = self.dl.act_and_trains(imgobj_right, target_action + 0.2)
                angle_error = abs(action - target_action)
            if distance > 0.1:
                self.select_dl = False
            elif distance < 0.05:
                self.select_dl = True
            if self.select_dl and self.episode >= 0:
                target_action = 0
            """
            """
            # proposed method (old)
            action, loss = self.dl.act_and_trains(imgobj, target_action)
            if abs(target_action) < 0.1:
                action_left,  loss_left  = self.dl.act_and_trains(imgobj_left, target_action - 0.2)
                action_right, loss_right = self.dl.act_and_trains(imgobj_right, target_action + 0.2)
            angle_error = abs(action - target_action)
            if distance > 0.1:
                self.select_dl = False
            elif distance < 0.05:
                self.select_dl = True
            if self.select_dl and self.episode >= 0:
                target_action = action
            """

            # follow line method
            action, loss = self.dl.act_and_trains(
                imgobj, dir_cmd, target_action)
            if abs(target_action) < 0.1:
                action_left,  loss_left = self.dl.act_and_trains(
                    imgobj_left,  dir_cmd, target_action - 0.2)
                action_right, loss_right = self.dl.act_and_trains(
                    imgobj_right, dir_cmd, target_action + 0.2)
            angle_error = abs(action - target_action)

            # end method

            print(" episode: " + str(self.episode) + ",dir:" + str(dir_cmd) + ", loss: " +
                  str(loss) + ", angle_error: " + str(angle_error) + ", distance: " + str(distance))
            self.episode += 1
            line = [str(self.episode), "training", str(
                loss), str(angle_error), str(distance)]
            with open(self.path + self.start_time + '/' + 'reward.csv', 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(line)
            self.vel.linear.x = 0.2
            self.vel.angular.z = target_action
            self.nav_pub.publish(self.vel)

        elif self.flag:
            dir_cmd = np.asanyarray(self.dir_cmd_data)
            target_action = self.dl.act(imgobj, dir_cmd)
            distance = self.min_distance
            print("USE DIRECTION MODE: " + " angular:" +
                  str(target_action) + ", distance: " + str(distance), str(dir_cmd))

            self.episode += 1
            angle_error = abs(self.action - target_action)
            line = [str(self.episode), "use_model", "0",
                    str(angle_error), str(distance)]
            with open(self.path + self.start_time + '/' + 'reward.csv', 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(line)
            self.vel.linear.x = 0.2
            self.vel.angular.z = target_action
            self.nav_pub.publish(self.vel)

        else:
            pass

        temp = copy.deepcopy(img)
        cv2.imshow("Resized Image", temp)
        temp = copy.deepcopy(img_left)
        cv2.imshow("Resized Left Image", temp)
        temp = copy.deepcopy(img_right)
        cv2.imshow("Resized Right Image", temp)
        cv2.waitKey(1)


if __name__ == '__main__':
    rg = cource_following_learning_node()
    DURATION = 0.25
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()
