import csv
import os
import xml.etree.cElementTree as et
import SSD.data_resize as dr
path = "VOCdevkit\VOC2007\Annotations/"
dirs = os.listdir(path)
dic= {
    'person':0,'bird':1, 'cat':2, 'cow':3, 'dog':4, 'horse':5, 'sheep':6,
    'aeroplane':7, 'bicycle':8, 'boat':9, 'bus':10, 'car':11, 'motorbike':12, 'train':13,
    'bottle':14, 'chair':15, 'diningtable':16, 'pottedplant':17, 'sofa':18, 'tvmonitor':19
}
xml_data_to_csv = open('voc07_train.csv', 'w')
c = 0
count = 0
i = 0

obj_bound_list = dr.redo_bound()

Csv_writer = csv.writer(xml_data_to_csv)
list_head = ['image_name','class','class_id','xmin','ymin','xmax','ymax']
Csv_writer.writerow(list_head)

for item in dirs:

    if os.path.isfile(path + item):

        c+=1
        if(c%100==0):
            print(c," done")

        tree = et.parse(path+item)
        root = tree.getroot()

        list_head = []


        for element in root.findall('object'):
            list_nodes =[]


            try:
                name = root.find('filename').text
                list_nodes.append(name)

                name = element.find('name').text
                list_nodes.append(name)

                list_nodes.append(dic[name])

                xmin = (int)(obj_bound_list[i][0])
                list_nodes.append(xmin)


                ymin = (int)(obj_bound_list[i][1])
                list_nodes.append(ymin)


                xmax = (int)(obj_bound_list[i][2])
                list_nodes.append(xmax)

                ymax = (int)(obj_bound_list[i][3])
                list_nodes.append(ymax)

            except:
                list_nodes = []
                for part in element.findall('part'):

                    name = root.find('filename').text
                    list_nodes.append(name)

                    name = element.find('name').text
                    list_nodes.append(name)

                    list_nodes.append(dic[name])

                    xmin = (int)(obj_bound_list[i][0])
                    list_nodes.append(xmin)

                    ymin = (int)(obj_bound_list[i][1])
                    list_nodes.append(ymin)

                    xmax = (int)(obj_bound_list[i][2])
                    list_nodes.append(xmax)

                    ymax = (int)(obj_bound_list[i][3])
                    list_nodes.append(ymax)

                    if (len(list_nodes) > 0):
                        Csv_writer.writerow(list_nodes)

                    list_nodes = []

            i+=1
            #Write List_nodes to csv
            if(len(list_nodes)>0):
                Csv_writer.writerow(list_nodes)


xml_data_to_csv.close()

