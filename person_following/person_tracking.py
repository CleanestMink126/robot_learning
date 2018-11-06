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
MAX_TURN = .46 #Used for conversion from pixels to turning speed
MAX_MOVE = .2 #Used for conversion betweeen pixels and forward speed
MAX_PIXELS = 240 #Makes difference in pixels between -1 and 1
IMAGE_SIZE = 448,256,3 #Input image size


class TrackPerson:
    def __init__(self):
        rospy.init_node('person_tracker')
        print('init')
        self.turn_velocity = 0 #start turn velocity
        self.forward_velocity = 0 #start forward velocity
        self.threshold_person = 0.5 #How sure the pre-trained model is that there's a person in screen
        self.threshold_direction = 10 #in pixels, how far the person's bounding box is to decide to turn
        self.bridge = CvBridge() #used to get images
        self.get_image = False #whether to graab newest image
        self.image = None #newest image
        self.model = detect_people.DetectorAPI(path_to_ckpt = MODEL_PATH) #Load person detector model
        self.sess, self.x, self.output,_ = trainFaces.build_model_inference() #Load looking detection model
        self.publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10) #Velocity Publisher
        rospy.Subscriber('camera/image_raw', Image, self.process_image) #Camera subscriber

    def detect_looking(self, img):
        '''Accepts image (or list of images) and returns the output of the model'''
        looking = self.sess.run(self.output,feed_dict={self.x:img})
        return np.squeeze(looking)

    def process_image(self, m):
        '''Decide whether or not to save the most recent message. We use the bool
        self.get_image so that we don't spend the compute of saving every image,
        rather only when we're ready for another image'''
        if self.get_image:
            self.image = self.bridge.imgmsg_to_cv2(m, desired_encoding="bgr8")
            self.get_image = False

    def send_speed(self,move_forward = None, turn_left = None):
        '''Standard function to send velocity inputs to the Neato. Negative means
        go backwards/ turn right'''
        if move_forward == None:
            move_forward = 0
        if turn_left == None:
            turn_left = 0
        my_point_stamped = Twist(linear=Vector3(move_forward,0,0), angular=Vector3(0,0,turn_left))
        self.publisher.publish(my_point_stamped)

    def get_box(self):
        '''Return a bounding box if there's a person in the image '''
        boxes, scores, classes, num = self.model.processFrame(self.image) #check for people
        x_max = self.image.shape[0]
        classes=np.array(classes)
        scores = np.array(scores)
        classes_index = np.where(classes == 1)[0] #find all guesses that predict people
        if len(classes_index):
            #find the guess that has the most confidence that it's a person
            max_person = np.argmax(scores[classes_index])
            max_person_index = classes_index[max_person]
            if scores[max_person_index] > self.threshold_person:
                #if the max confiednce is over a threshold return the max box
                return boxes[max_person_index]
        return None

    def set_speed(self, box):
        '''Set proportional Control based on where the bounding box is'''
        x_max = self.image.shape[0]

        rectange_center = (box[1] + box[3])//2 #get center of box
        box_dir = float(x_max//2 - rectange_center) #find where center of box is in relation to center of image
        if abs(box_dir) > self.threshold_direction:
            #Set a proportional velocity if the bounding box is enough to the side
            self.turn_velocity=MAX_TURN * box_dir/MAX_PIXELS
            self.forward_velocity = MAX_MOVE * (1- abs(box_dir/MAX_PIXELS))
        elif abs(box_dir) <= self.threshold_direction:
            #Full steam ahead if bounding box is in center
            self.turn_velocity = 0
            self.forward_velocity = MAX_MOVE

    def decide_stop(self, looking):
        soft_sum = np.sum(np.exp(looking)) #Get softmax probability of the person is looking
        curr_val = np.exp(looking[1])/soft_sum
        if curr_val >= .3: #stop robot if probability is above .3
            self.forward_velocity = 0
            self.send_speed(self.forward_velocity, self.turn_velocity)

    def run(self):
        '''This is the main loop for the Neato to follow a person infront of it'''
        import time
        # time.sleep(10)
        r = rospy.Rate(3)#How fast to run the loop
        save_data=False #Whether or not to colelct training data
        directory = '/Data/PersonTracking/test/away/' #directory to save the data
        prefix_name = 'away_'
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

        while not rospy.is_shutdown():
            self.send_speed(self.forward_velocity, self.turn_velocity)#start off by sending the current speed
            if self.image is not None: #if we have gotten an image
                box = self.get_box()
                if box is None:#if we did not find a suitable box, try again later
                    self.image = None
                    self.get_image = True
                    continue

                img = self.image[box[0]:box[2],box[1]:box[3]] #get box of interest
                img = cv2.resize(img, (IMAGE_SIZE[1],IMAGE_SIZE[0])) # make box right size
                looking = self.detect_looking(np.expand_dims(img,0)) #get prediction of box

                if save_data: #save image if we are collecting data
                    cv2.imwrite(directory + prefix_name + str(rospy.Time.now())+'.png',img)

                cv2.imshow("image",img) #show box
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'): #stop neato and quit if 'q'
                    self.send_speed(0,0)
                    break

                self.set_speed(box) # determine proportional speed
                self.decide_stop(looking) #determine whether or not to stop
                self.image = None #make sure to wait for a new image
            self.get_image = True
            r.sleep()




if __name__ == "__main__":
    node = TrackPerson()
    node.run()
