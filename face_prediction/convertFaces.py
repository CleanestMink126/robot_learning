
#We should get a glob to get all the image files we want to examine
#We should get a dictionary to map file paths from the glob to indexes
#We should read the text files to generate labels for these images_file
#then we should load the images. When we reach a max value we should write to a binary file
from glob import glob
import os
import numpy as np
from PIL import Image
from random import shuffle
import matplotlib.pyplot as plt

def append_binary_file(file_name, bytes_):
    with open(file_name,"ab") as f:
        f.write(bytes_)

class FromJPEG:
    DEBUG = False
    # Image configuration
    data_dir = './Models/data'
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
        # print('get file')
        data_batch = np.array(
            [FromJPEG.get_image(sample_file, width, height, mode, box=box) for sample_file in image_files]).astype(np.uint8)
        # Make sure the images are in 4 dimensions
        if len(data_batch.shape) < 4:
            data_batch = data_batch.reshape(data_batch.shape + (1,))

        return data_batch
    @staticmethod
    def get_batches(batch_size,folder,IMAGE_WIDTH,IMAGE_HEIGHT, box = None):
        """
        Generate batches
        """
        # print('start get_batches')
        current_index = 0
        data_files = glob(folder)
        shape = len(data_files), IMAGE_WIDTH, IMAGE_HEIGHT, 3
        #TODO
        labels = None
        # print(shape[0])
        while current_index + batch_size <= shape[0]:
            data_batch = get_batch(
                data_files[current_index:current_index + batch_size],
                *shape[1:3], box=box)
            labels_batch = labels[current_index:current_index + batch_size]
            # print('got files')
            current_index += batch_size
            yield data_batch, labels

if __name__ == '__main__':
    np.set_printoptions(threshold=np.nan)
    DATA_PATH = '/Data/PersonTracking/test/'
    TRAIN_INPUT_SAVE = '/Data/PersonTracking/test/test_images'
    TRAIN_LABEL_SAVE = '/Data/PersonTracking/test/test_labels'
    WIDTH, HEIGHT = 256, 448
    batch_size= 10
    ALL_DATA_PATHS = glob(os.path.join(DATA_PATH,'*','*.png'))
    CLASS_PATHS = glob(os.path.join(DATA_PATH,'*'))
    NUM_CLASSES = 1000
    shuffle(ALL_DATA_PATHS)
    files_dict = {v.split('/')[4]: i for i, v in enumerate(CLASS_PATHS)}
    print(files_dict)
    i = 0
    while i != -1:
        if i+batch_size <= len(ALL_DATA_PATHS):
            j = i + batch_size
        else:
            break
            j = -1
        # print(i)
        labels = np.full(batch_size, 0, dtype = np.int32)
        for k in range(i,j):
            class_name = ALL_DATA_PATHS[k].split('/')[4]
            labels[k-i] = files_dict[class_name]
        images = FromJPEG.get_batch(ALL_DATA_PATHS[i:j], WIDTH, HEIGHT)
        print(images.shape)
        exit()
        # for k in range(10):
        #     print(labels[k])
        #     plt.imshow(images[k])
        #     plt.show()
        # exit()
        if len(labels) != len(images):
            print('MISMATCH-------------------------------')
            i = j
            continue
        append_binary_file(TRAIN_INPUT_SAVE,images.tobytes())
        append_binary_file(TRAIN_LABEL_SAVE,labels.tobytes())
        print('ITERS:' ,i)
        i = j
