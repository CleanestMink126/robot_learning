from glob import glob
import os
import numpy as np
from PIL import Image
from random import shuffle
import matplotlib.pyplot as plt

'''
This script will convert from a bunch of JPEG files in the file system to a binary file
Binary Files allow for much more efficient training through tensorflow.
'''

def append_binary_file(file_name, bytes_):
    with open(file_name,"ab") as f:
        f.write(bytes_)

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
    batch_size= 10 #how many images to process at a time
    #I wrote this code pretty simply so it'll chop off N%batch_size data points
    #So lower batch size is better
    ALL_DATA_PATHS = glob(os.path.join(DATA_PATH,'*','*.png')) #Get all data
    CLASS_PATHS = glob(os.path.join(DATA_PATH,'*')) #get all classes
    shuffle(ALL_DATA_PATHS) #shuffle data for good measure
    files_dict = {v.split('/')[4]: i for i, v in enumerate(CLASS_PATHS)}
    print(files_dict)
    i = 0
    while i != -1:#main loop to walk through the files
        if i+batch_size <= len(ALL_DATA_PATHS):
            j = i + batch_size
        else:
            break
        labels = np.full(batch_size, 0, dtype = np.int32)
        for k in range(i,j):#define the class based on the filepath of each image
            class_name = ALL_DATA_PATHS[k].split('/')[4]
            labels[k-i] = files_dict[class_name]
        images = FromJPEG.get_batch(ALL_DATA_PATHS[i:j], WIDTH, HEIGHT)#get the images
        if len(labels) != len(images):#make sure nothing weird is going on
            print('MISMATCH-------------------------------')
            i = j
            continue
        append_binary_file(TRAIN_INPUT_SAVE,images.tobytes())#write the info
        append_binary_file(TRAIN_LABEL_SAVE,labels.tobytes())
        print('ITERS:' ,i)
        i = j
