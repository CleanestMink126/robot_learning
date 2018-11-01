
from glob import glob
import os
import numpy as np
from PIL import Image
from random import shuffle
import matplotlib.pyplot as plt
import cv2
import trainFaces
'''
This script is similar to convertFaces with the change that it will test haar
classifier instead of saving the images
'''

class FromJPEG:
    '''Object from another Repo that will load images in appropriate format'''
    @staticmethod
    def get_image(image_path, width, height, mode, box = None):
        """
        Read image from image_path
        """
        image = Image.open(image_path)
        image = image.resize([width, height], Image.BILINEAR)
        return np.array(image.convert(mode))
    @staticmethod
    def get_batch(image_files, width, height, box = None, mode='RGB'):
        """
        Get a single image
        """
        data_batch = np.array(
            [FromJPEG.get_image(sample_file, width, height, mode, box=box) for sample_file in image_files]).astype(np.uint8)
        # Make sure the images are in 4 dimensions
        if len(data_batch.shape) < 4:
            data_batch = data_batch.reshape(data_batch.shape + (1,))
        return data_batch

if __name__ == '__main__':
    '''
    We should get a glob to get all the image files we want to examine
    We should get a dictionary to map file paths from the glob to indexes
    Each folder represents a different class
    Then we should load the images and should write to a binary file
    '''
    DATA_PATH = '/Data/PersonTracking/test/'
    TRAIN_INPUT_SAVE = '/Data/PersonTracking/test/test_images'
    TRAIN_LABEL_SAVE = '/Data/PersonTracking/test/test_labels'
    WIDTH, HEIGHT = 256, 448
    DEBUG=False
    #how many images to process at a time
    #I wrote this code pretty simply so it'll chop off N%batch_size data points
    #So lower batch size is better
    ALL_DATA_PATHS = glob(os.path.join(DATA_PATH,'*','*.png')) #Get all data
    CLASS_PATHS = glob(os.path.join(DATA_PATH,'*/')) #get all classes
    shuffle(ALL_DATA_PATHS) #shuffle data for good measure
    files_dict = {v.split('/')[4]: i for i, v in enumerate(CLASS_PATHS)}
    print(files_dict)
    i = 0
    #Declare Haar Classifier
    face_cascade = cv2.CascadeClassifier('/home/gtower/.local/lib/python2.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    #Declare my model
    sess, x, output, importance_map = trainFaces.build_model_inference()
    #For the haar classier
    t_p = 0#True Positives
    t_n = 0#True Negatives
    p = 0.0#Total Positives
    n = 0.0#Total Negatives
    #For my classifier
    m_t_p = 0#True Positives
    m_t_n = 0#True Negatives
    m_p = 0.0#Total Positives
    m_n = 0.0#Total Negatives
    while i < len(ALL_DATA_PATHS):#main loop to walk through the files
        class_name = ALL_DATA_PATHS[i].split('/')[4]#get the name of the class
        label = bool(files_dict[class_name])#get the index of the class
        image = FromJPEG.get_image(ALL_DATA_PATHS[i], WIDTH, HEIGHT, 'RGB')#get the images

        #Run my model on the image
        image_float = image.astype(dtype=np.float32)
        looking, importance_map_ex = sess.run([output,importance_map],feed_dict={x:np.expand_dims(image_float[:,:,::-1],0)})
        looking = np.squeeze(looking)
        importance_map_ex = np.squeeze(importance_map_ex) #The ouput of the convolutional layers

        soft_sum = np.sum(np.exp(looking)) #Get softmax probability of the person is looking
        curr_val = np.exp(looking[1])/soft_sum + .3
        curr_val = np.round(curr_val) #Get the final 'guess'
        if curr_val == label: #Some logic to increment the correct values
            m_t_p +=curr_val
            m_t_n+= 1-curr_val
        m_p +=curr_val
        m_n+= 1-curr_val

        #Run the haar model
        image_g =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image_g, 1.3, 2)
        guess = bool(len(faces))#Get whether we detect a face
        if guess == label:#Some logic to increment the correct values for haar
            t_p +=guess
            t_n+= 1-guess
        p +=guess
        n+= 1-guess

        print('ITERS:' ,i)
        if DEBUG and guess==False and label==True:
            #Potentially show some wrong choices of the haar classifier
            for (x,y,w,h) in faces:
                cv2.rectangle(image_g,(x,y),(x+w,y+h),(0),2)
            while True:
                cv2.imshow("image",image_g) #show box
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'): #stop neato and quit if 'q'
                    break

        i += 1
    print('TOTAL ACC:', (t_p+t_n)/(p+n))
    print('TRUE POSITIVE:',t_p/p)
    print('TRUE NEGATIVE:',t_n/n)
    print('TOTAL ACC:', (m_t_p+m_t_n)/(m_p+m_n))
    print('TRUE POSITIVE:',m_t_p/m_p)
    print('TRUE NEGATIVE:',m_t_n/m_n)
