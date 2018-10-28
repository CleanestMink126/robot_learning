#!/usr/bin/env python
import sys
sys.path.append('./../face_prediction/')
import trainFaces
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
import matplotlib.pyplot as plt

MODEL_PATH = '/Data/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
MAX_TURN = .46
MAX_MOVE = .2
MAX_PIXELS = 240
IMAGE_SIZE = 448,256,3


class TrackPerson:
    def __init__(self):
        rospy.init_node('person_tracker')
        print('init')
        self.turn_velocity = 0
        self.forward_velocity = .2
        self.threshold_person = 0.5
        self.threshold_direction = 10 #in pixels
        self.bridge = CvBridge()
        self.get_image = False
        self.image = None
        self.model = detect_people.DetectorAPI(path_to_ckpt = MODEL_PATH)
        self.sess, self.x, self.output = trainFaces.build_model_inference()
        print('model loaded')

        self.publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('camera/image_raw', Image, self.process_image)
        print('ros connected')

    def detect_looking(self, img):
        looking = self.sess.run(self.output,feed_dict={self.x:img})
        return np.squeeze(looking)

    def process_image(self, m):
        # print('processing')
        if self.get_image:
            # print('GOT IMAGE')
            self.image = self.bridge.imgmsg_to_cv2(m, desired_encoding="bgr8")
            # print(type(self.image))
            # self.image = np.array(np.flip(self.image,axis = 0))
            # print(self.image.shape)
            self.get_image = False

    def send_speed(self,move_forward = None, turn_left = None):
        if move_forward == None:
            move_forward = 0
        if turn_left == None:
            turn_left = 0
        my_point_stamped = Twist(linear=Vector3(move_forward,0,0), angular=Vector3(0,0,turn_left))
        self.publisher.publish(my_point_stamped)

    def run(self):
        r = rospy.Rate(4)
        directory = '/Data/PersonTracking/test/away/'
        prefix_name = 'away_'
        while not rospy.is_shutdown():
            # print(self.image)
            # print(self.get_image)
            self.send_speed(self.forward_velocity, self.turn_velocity)

            if self.image is not None:
                boxes, scores, classes, num = self.model.processFrame(self.image)
                x_max = self.image.shape[0]
                box = None
                classes=np.array(classes)
                scores = np.array(scores)
                classes_index = np.where(classes == 1)[0]
                if len(classes_index):
                    max_person = np.argmax(scores[classes_index])
                    max_person_index = classes_index[max_person]
                    if scores[max_person_index] > self.threshold_person:
                        box = boxes[max_person_index]

                if box is None:
                    self.image = None
                    # self.forward_velocity = 0
                    continue
                # for i in range(len(boxes)):
                #     # Class 1 represents human
                #     if classes[i] == 1 and scores[i] > self.threshold_person:
                #         box = boxes[i]
                #         break


                # print(img.shape)
                # plt.imshow(img)
                # plt.show()
                img = self.image
                img = img[box[0]:box[2],box[1]:box[3]]
                img = cv.resize(img, (IMAGE_SIZE[0],IMAGE_SIZE[1],IMAGE_SIZE[2]))
                cv2.imshow("image",img)
                looking = self.detect_looking(img)


                # print(prefix_name + str(rospy.Time.now())+'.png')
                # cv2.imwrite(directory + prefix_name + str(rospy.Time.now())+'.png',img)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    self.send_speed(0,0)
                    break
                # print('Box Values', box[1] ,box[3])
                rectange_center = (box[1] + box[3])//2
                box_dir = float(x_max//2 - rectange_center)
                if abs(box_dir) > self.threshold_direction:
                    self.turn_velocity=MAX_TURN * box_dir/MAX_PIXELS
                    self.forward_velocity = MAX_MOVE * (1- abs(box_dir/MAX_PIXELS))
                elif abs(box_dir) <= self.threshold_direction:
                    self.turn_velocity = 0
                    self.forward_velocity = MAX_MOVE
                # print(box_dir)
                print('Looking:', looking)
                if looking == 1:
                    self.forward_velocity = 0
                    self.send_speed(self.forward_velocity, self.turn_velocity)
                print('Turn Speed:',self.turn_velocity)
                self.image = None
            self.get_image = True
            r.sleep()




if __name__ == "__main__":
    node = TrackPerson()
    node.run()
