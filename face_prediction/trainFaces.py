import tensorflow as tf
import numpy as np
from tensorlayer.layers import *

# --------------------
#ARCHITECTURE INFO
IMAGE_SIZE = 448,256,3
CONVOLUTIONS = [-32, -64] #Convolutional Layers, negative means downsampling
HIDDEN_SIZE = 100
NUM_CLASSES = 1
# ---------------------
#TRAINING INFO
LEARNING_RATE = 1e-3
MOMENTUM = .5 #For Adam Optimizer
KEEP_PROB = .5 #For dropout
BATCH_SIZE = 12
TEST_BATCH_SIZE = 250
# ----------------------
#FILEPATH INFO
FILEPATH = '/Data/PersonTracking/'
TRAIN_INPUT = FILEPATH + 'train/'+ 'train_images'
TEST_INPUT = FILEPATH + 'test/' + 'test_images'
TRAIN_LABEL = FILEPATH + 'train/'+ 'train_labels'
TEST_LABEL = FILEPATH + 'test/' + 'test_labels'
PERM_MODEL_FILEPATH = '/Models/PersonTracking/model.ckpt' #filepaths to model and summaries
SUMMARY_FILEPATH ='/Models/PersonTracking/Summaries/'
# ----------------------
#MISC INFO
RESTORE = False
WHEN_DISP = 50 #How often to visualize images
WHEN_TEST = 50 # How often to run on test set
NUM_OUTPUTS = 10 #Image viewing
WHEN_SAVE = 500 #How often to save the model
ITERATIONS = 25000 #How long to trian max
NUM_EXAMPLES = 3500 #~total amount of data

def decode_image(image):
    '''Convert from Binary and Normalize from [0, 255] to [0.0, 1.0]'''
    image = tf.decode_raw(image, tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]])
    return image / 255.0

def decode_label(label):
    '''Decode labels from binary file'''
    label = tf.decode_raw(label, tf.int32)
    label = tf.reshape(label, [1])
    return  label

def return_datatset_train():
    '''Load the training dataset from a Binary File'''
    images = tf.data.FixedLengthRecordDataset(
      TRAIN_INPUT, IMAGE_SIZE[0] * IMAGE_SIZE[1] * IMAGE_SIZE[2]).map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(
      TRAIN_LABEL, 1* 4).map(decode_label)
    return tf.data.Dataset.zip((images, labels))

def return_datatset_test():
    '''Load the testing dataset from a binary file'''
    images = tf.data.FixedLengthRecordDataset(
      TEST_INPUT, IMAGE_SIZE[0] * IMAGE_SIZE[1] * IMAGE_SIZE[2]).map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(
      TEST_LABEL, 1* 4).map(decode_label)
    return tf.data.Dataset.zip((images, labels))

def create_main_model(x_image, reuse = False, is_train = None):
    '''Create a discrimator, note the convolutions may be negative to represent
        downsampling
        reuse    --> Whether or not to reuse variables
        is_train --> Whether to use dropout'''
    if is_train is None:
        is_train = not reuse
    with tf.variable_scope("main_model") as scope:
        if reuse: #get previous variable if we are reusing the discriminator but with fake images
            scope.reuse_variables()
        xs, ys = IMAGE_SIZE[0],IMAGE_SIZE[1]
        inputs = InputLayer(x_image, name= 'inputs')
        convVals = inputs # Convals represents the input to a layer
        for i,v in enumerate(CONVOLUTIONS):
            '''Similarly tile for constant reference to class'''
            if v < 0:
                v *= -1
                convVals = Conv2d(convVals,v, (3, 3), act=tf.nn.relu,strides =(2,2),name='conv_%i'%(i))
            else:
                convVals = Conv2d(convVals,v, (3, 3), act=tf.nn.relu,strides =(1,1),name='conv_%i'%(i))
        flat3 = FlattenLayer(convVals, name ='flatten')
        hid3 = DenseLayer(flat3, HIDDEN_SIZE,act = tf.nn.relu, name ='fcl')
        hid3 = DropoutLayer(hid3, keep=KEEP_PROB,is_train=is_train, is_fix=True, name='drop1')
        y_conv = DenseLayer(hid3, 2,  name = 'output').outputs
        importance_map = tf.nn.tanh(convVals.outputs)
        return y_conv,importance_map

def build_model(x, og_classes, reuse = False):
    '''Build a model for training and testing'''
    classes = tf.squeeze(og_classes, 1)
    classes_one =tf.one_hot(classes, 2)
    if not reuse:
        prefix = 'train_'
    else:
        prefix = 'test_'
    main_outputs,_ = create_main_model(x,reuse=reuse)#create model
    with tf.variable_scope('logistics') as scope:
        image_summary = tf.summary.image(prefix + "inputs", x,max_outputs = NUM_OUTPUTS)#get some example images
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=classes_one, logits=main_outputs))#get cost
        cross_entropy_summary = tf.summary.scalar(prefix + 'loss',cross_entropy)#get summary of cost for tensorboard
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(main_outputs,-1,output_type=tf.int32),classes), tf.float32))#determine accuracy
        accuracy_summary = tf.summary.scalar(prefix + 'accuracy',accuracy) #get summary of accuracy for tensorboard
    if not reuse: #if we're training use an atom optimizer
        train_step = tf.train.AdamOptimizer(LEARNING_RATE,beta1=MOMENTUM).minimize(cross_entropy)
    else:
        train_step = None
    scalar_summary = tf.summary.merge([cross_entropy_summary,accuracy_summary]) #compile summaries
    return scalar_summary, image_summary, train_step

def train_model():
    '''Start training a model'''
    sess = tf.Session()#start the session
    ##############GET DATA###############
    train_ship = return_datatset_train().repeat().batch(BATCH_SIZE)
    train_iterator = train_ship.make_initializable_iterator()
    train_input, train_label = train_iterator.get_next()

    test_ship = return_datatset_test().repeat().batch(TEST_BATCH_SIZE)
    test_iterator = test_ship.make_initializable_iterator()
    test_input, test_label = test_iterator.get_next()

    sess.run([train_iterator.initializer,test_iterator.initializer])
    ####################DEFINE MODEL############################################
    scalar_summary, image_summary, train_step = build_model(train_input, train_label)
    test_scalar_summary, test_image_summary, _ = build_model(test_input, test_label, reuse = True)
    ####################LOAD MODEL AND SUMMARY#########################################
    sess.run(tf.global_variables_initializer())
    saver_perm = tf.train.Saver()
    if PERM_MODEL_FILEPATH is not None and RESTORE: #if we load a model
        saver_perm.restore(sess, PERM_MODEL_FILEPATH)
    else:
        print('SAVE')
        saver_perm.save(sess, PERM_MODEL_FILEPATH)
    train_writer = tf.summary.FileWriter(SUMMARY_FILEPATH,
                                  sess.graph)
    #####################TRAIN##############################################
    for i in range(ITERATIONS):
        if not i % WHEN_DISP: #Whether to save some example images
            input_summary_ex, image_summary_ex,_= sess.run([scalar_summary, image_summary, train_step])
            train_writer.add_summary(image_summary_ex, i)
        else:
            input_summary_ex, _= sess.run([scalar_summary, train_step])
        train_writer.add_summary(input_summary_ex, i)

        if not i % WHEN_TEST: #whether to test on the test set
            input_summary_ex, image_summary_ex= sess.run([test_scalar_summary, test_image_summary])
            train_writer.add_summary(image_summary_ex, i)
            train_writer.add_summary(input_summary_ex, i)

        if not i % WHEN_SAVE: #whether to save the model
            print('SAVE')
            saver_perm.save(sess, PERM_MODEL_FILEPATH)

        if not i % (NUM_EXAMPLES // BATCH_SIZE): #rough estimate of number of epochs
            #Get a vague idea of probability of overfitting
            print('EPOCH:' , i / (NUM_EXAMPLES // BATCH_SIZE))

def build_model_inference():
    '''Build a model for live use'''
    sess = tf.Session()#start the session
    #define a placeholder (empty box) where we're going to put our data
    x = tf.placeholder(dtype=tf.float32, shape=(1,IMAGE_SIZE[0],IMAGE_SIZE[1],IMAGE_SIZE[2]))
    x_rev = tf.reverse(x,axis=[3])/255.0 #If we're using the neato, convert from BGR to RGB and to [0,1] range
    main_outputs, importance_map = create_main_model(x_rev,is_train=False) #Build model
    sess.run(tf.global_variables_initializer())
    saver_perm = tf.train.Saver()
    saver_perm.restore(sess, PERM_MODEL_FILEPATH)#Restore model
    return sess, x, main_outputs,importance_map

if __name__ == '__main__':
    sess, x,binary_output = build_model_inference()
