import os

from gluoncv import data, utils
from matplotlib import pyplot as plt
import gluoncv

train_dataset = data.VOCDetection('VOCdevkit/',splits=[(2007, 'trainval')])
#val_dataset = data.VOCDetection('VOCdevkit/',splits=[(2007, 'test')])
print('Num of training images:', len(train_dataset))


path = "VOCdevkit\VOC2007\JPEGImages/"
dirs = os.listdir( path )




def redo_bound():
    i = 0
    bounding_box_list = []
    for train_image, train_label in train_dataset:
      #print(dirs[i],' Image size (height, width, RGB):', train_image.shape)
      #bounding_boxes = train_label[:, :4]
      #print(bounding_boxes)
      new_img = gluoncv.data.transforms.image.imresize(train_image, 224, 224, interp=1)
      #train_image = new_img
      #print(dirs[i], 'new Image size (height, width, RGB):', new_img.shape)
      bounding_boxes = gluoncv.data.transforms.bbox.resize(train_label[:, :4], (train_image.shape[1], train_image.shape[0]), (224, 224))
     # print(bounding_boxes)
      for item in bounding_boxes:
          bounding_box_list.append(item)
      i+=1

      if(i%100==0):
          print(i, 'items in bounding_box_dict done')

    return bounding_box_list

