import scipy

import keras
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.layers import Input, RepeatVector, GlobalAveragePooling2D, Dropout
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing import image
import  cv2
import numpy as np
import os
import scipy
import tensorflow as tf



(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train/255.0
X_test = X_test/255.0


X_train = np.transpose(X_train, axes=[0,3,1,2])

a = []
c = 0
per = 0
for j in range(0,500):
    image = X_train[j]
    b = []
    for i in range(0,image.shape[0]):
       img = np.pad(image[i], ((96,96),(96,96)), 'constant', constant_values=0)
       b.append(img)
    c+=1
    a.append(b)

    if(c%50==0):
        per +=10
        print(per," %")

X_train = np.array(a)
X_train = np.transpose(X_train, axes=[0,2,3,1])
print(X_train.shape)

X_test = np.transpose(X_test, axes=[0,3,1,2])

a = []
c = 0
per = 0
for j in range(0,100):
    image = X_test[j]
    b = []
    for i in range(0,image.shape[0]):
       img = np.pad(image[i], ((96,96),(96,96)), 'constant', constant_values=0)
       b.append(img)
    c+=1
    a.append(b)

    if(c%10==0):
        per +=10
        print(per," %")


X_test = np.array(a)
X_test = np.transpose(X_test, axes=[0,2,3,1])
print(X_test.shape)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


input_img = Input(shape = (224, 224, 3))

def Conv_Block(kernel, kernel_size, pad, stride, input_layer):

    temp = input_layer
    if pad is not 'same':
        temp = tf.pad(temp, [[0,0],[0,pad],[0,pad],[0,0]],'constant')

    layer = Conv2D(kernel, (kernel_size, kernel_size), padding='valid', strides=stride)(temp)
    layer_bn = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(layer)
    layer_ac = keras.layers.Activation('relu')(layer_bn)
    return layer_ac


def Dense_Block(num_layer, input_layer, bottleneck_width, growth_rate = 32):

    layer = input_layer
    x = (int)(growth_rate/2)
    x = x * bottleneck_width
    for i in range(0, num_layer):

        dense_block_1_1 = Conv_Block(x, 1,'same', 1, layer)
        dense_block_1_2 = Conv_Block((int)(x/2), 3, 2, 1, dense_block_1_1)

        dense_block_2_1 = Conv_Block(x, 1,'same', 1, layer)
        dense_block_2_2 = Conv_Block((int)(x/2), 3, 2, 1, dense_block_2_1)
        dense_block_2_3 = Conv_Block((int)(x/2), 3, 2, 1, dense_block_2_2)
        dense_filter_con = keras.layers.concatenate([layer, dense_block_1_2, dense_block_2_3], axis=3)

        layer = dense_filter_con

    return layer


dense_layer=[3,4,8,6]
bottleneck_width = [1,2,4,4]




# Stage - 0
# STEM BLOCK - based on Inception v4

# input - 224*224*3

block = Conv_Block(32, 3, 1 ,2, input_img)
block_1_1 = Conv_Block(16, 1, 'same',1, block)
block_1_2 = Conv_Block(32, 3, 1, 2, block_1_1)
block_2_1 = MaxPooling2D((2,2), strides=2, padding='same')(block)
filter_con = keras.layers.concatenate([block_1_2,block_2_1], axis=3)
output_stem = Conv_Block(32, 1,'same', 1, filter_con)

# output - 56*56*32


#Stage - 1
# input - 56*56*32
#DENSE LAYER - originally total 3


s1_dense_block = Dense_Block(dense_layer[0], output_stem, bottleneck_width[0])

#TRANSITION LAYER

s1_trans_1 = Conv2D(128,(1,1), padding='same', activation='relu', strides=1)(s1_dense_block)
s1_trans_2 = MaxPooling2D((2,2), strides=2, padding='same')(s1_trans_1)
output_stage_1 = s1_trans_2

# output - 28*28*128


#Stage - 2
# input - 28*28*128
#DENSE LAYER - originally total 4

s2_dense_block = Dense_Block(dense_layer[1], output_stage_1, bottleneck_width[1])

#TRANSITION LAYER

s2_trans_1 = Conv2D(256,(1,1), padding='same', activation='relu', strides=1)(s2_dense_block)
s2_trans_2 = MaxPooling2D((2,2), strides=2, padding='same')(s2_trans_1)
output_stage_2 = s2_trans_2

# output - 14*14*256


#Stage - 3
# input - 14*14*256
#DENSE LAYER - originally total 8

s3_dense_block = Dense_Block(dense_layer[2], output_stage_2 , bottleneck_width[2])

#TRANSITION LAYER

s3_trans_1 = Conv2D(512,(1,1), padding='same', activation='relu', strides=1)(s3_dense_block)
s3_trans_2 = MaxPooling2D((2,2), strides=2, padding='same')(s3_trans_1)
output_stage_3 = s3_trans_2

# output - 7*7*512


#Stage - 4
# input - 7*7*512
#DENSE LAYER - originally total 6

s4_dense_block = Dense_Block(dense_layer[3], output_stage_3, bottleneck_width[3])

#TRANSITION LAYER

s4_trans_1 = Conv2D(704,(1,1), padding='same', activation='relu', strides=1)(s4_dense_block)
output_stage_4 = s4_trans_1

# output - 7*7*704


#Stage - 5
# input - 7*7*704

output_stage_5 = GlobalAveragePooling2D()(output_stage_4)

# output - 1*1*704


#CLASSIFICATION LAYER - SOFTMAX

output = keras.layers.Dense(10, activation='softmax')(output_stage_5)
model = Model(inputs = input_img, outputs = output)

epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)

#model_json = model.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(model_json)
#model.save_weights(os.path.join(os.getcwd(), 'model.h5'))

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))



