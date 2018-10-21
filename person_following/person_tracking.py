#!/usr/bin/env python

import rospy
import rospkg
from sensor_msgs.msg import Image
from neato_node.msg import Bump
from cv_bridge import CvBridge
import cPickle as pickle
import numpy as np
import cv2
import detect_people
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3

MODEL_PATH = '/Data/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
MAX_TURN = .75
MAX_PIXELS = 200

class TrackPerson:
    def __init__(self):
        rospy.init_node('person_tracker')
        self.turn_velocity = 0
        self.forward_velocity = 0
        self.threshold_person = 0.5
        self.threshold_direction = 20 #in pixels
        self.bridge = CvBridge()
        self.get_image = False
        self.image = None
        self.model = detect_people.DetectorAPI(path_to_ckpt = MODEL_PATH)
        self.publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('camera/image_raw', Image, self.process_image)

    def process_image(self, m):
        if self.get_image:
            self.image = self.bridge.imgmsg_to_cv2(m, desired_encoding="bgr8")
            self.get_image = False

    def send_speed(self,move_forward = None, turn_left = None):
        if move_forward == None:
            move_forward = 0
        if turn_left == None:
            turn_left = 0
        my_point_stamped = Twist(linear=Vector3(move_forward,0,0), angular=Vector3(0,0,turn_left))
        self.publisher.publish(my_point_stamped)

    def run(self):
        r = rospy.Rate(3)
        while not rospy.is_shutdown():
            if self.image is not None:
                boxes, scores, classes, num = self.model.processFrame(self.image)
                x_max = self.image.shape[0]
                box = None
                for i in range(len(boxes)):
                    # Class 1 represents human
                    if classes[i] == 1 and scores[i] > self.threshold_person:
                        box = boxes[i]
                        break
                cv2.rectangle(self.image,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
                cv2.imshow("image", self.image)
                if box is None: continue
                rectange_center = (box[1] + box[3])//2
                box_dir = x_max - rectange_center
                if abs(box_dir) > self.threshold_direction:
                    self.turn_velocity=MAX_TURN * box_dir/MAX_TURN
                elif abs(box_dir) <= self.threshold_direction:
                    self.turn_velocity = 0
                self.send_speed(self.forward_velocity, self.turn_velocity)
                self.image = None
            self.get_image = True
            r.sleep()




if __name__ == "__main__":
    node = TrackPerson()
    node.run()
