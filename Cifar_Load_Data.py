import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils

def Load_Data():

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

    y_train = y_train[:500]
    y_test = y_test[:100]

    return  X_train, X_test, y_train, y_test