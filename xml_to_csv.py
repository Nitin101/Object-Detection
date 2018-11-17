import csv
import os
import xml.etree.cElementTree as et

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
for item in dirs:

    if os.path.isfile(path + item):

        c+=1
        if(c%100==0):
            print(c," done")

        tree = et.parse(path+item)
        root = tree.getroot()

        list_head = []

        Csv_writer=csv.writer(xml_data_to_csv)


        for element in root.findall('object'):
            list_nodes =[]


            try:
                name = root.find('filename').text
                list_nodes.append(name)

                name = element.find('name').text
                list_nodes.append(name)

                list_nodes.append(dic[name])

                xmin = element[4][0].text
                list_nodes.append(xmin)


                ymin = element[4][1].text
                list_nodes.append(ymin)


                xmax = element[4][2].text
                list_nodes.append(xmax)

                ymax = element[4][3].text
                list_nodes.append(ymax)

            except:
                list_nodes = []
                for part in element.findall('part'):

                    name = root.find('filename').text
                    list_nodes.append(name)

                    name = element.find('name').text
                    list_nodes.append(name)

                    list_nodes.append(dic[name])

                    xmin = part[1][0].text
                    list_nodes.append(xmin)

                    ymin = part[1][1].text
                    list_nodes.append(ymin)

                    xmax = part[1][2].text
                    list_nodes.append(xmax)

                    ymax = part[1][3].text
                    list_nodes.append(ymax)

                    if (len(list_nodes) > 0):
                        Csv_writer.writerow(list_nodes)

                    list_nodes = []

            #Write List_nodes to csv
            if(len(list_nodes)>0):
                Csv_writer.writerow(list_nodes)


xml_data_to_csv.close()

