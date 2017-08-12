import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
ROOT = "/home/chuck/bittiger_course_mxnet/DATA/VOCdevkit/VOC2012/JPEGImages/"
N = 11540


def cp_alliamge(savimg=False):
    catlist = []
    with open("cat_val.txt", "r") as f:
        for x in f.readlines():
            split = x.split(' ')
            if len(split)==3:
                catlist.append(split[0])
                if savimg:
                    img_path = ROOT+split[0]+'.jpg'
                    img = plt.imread(img_path)
                    plt.imsave('DATA/'+split[0]+'.jpg', img)
    print ("There are {} cat image".format(len(catlist)))
    print (catlist)
    return catlist


def get_all_annotation(path, imglist, save=True):
    annotation_list = {}
    for name in imglist:
        filename = path + name + '.xml'
        tree = ET.parse(filename)
        root = tree.getroot()
        objs = root.findall('object')
        bboxes = []
        for _, obj in enumerate(objs):
            cls = obj.find('name').text.lower().strip()
            if cls == 'cat':
                bbox = obj.find('bndbox')
                # make pixel indexes 0-based
                x1 = int(bbox.find('xmin').text) - 1
                y1 = int(bbox.find('ymin').text) - 1
                x2 = int(bbox.find('xmax').text) - 1
                y2 = int(bbox.find('ymax').text) - 1

                cx = (x2+x1)/2.
                cy = (y2+y1)/2
                w = x2-x1
                h = y2-y1

                bboxes.append([cx, cy, w, h])
        annotation_list[name]=bboxes
    np.save("annotation_list.npy", annotation_list)
    return annotation_list



path = "/home/chuck/bittiger_course_mxnet/DATA/VOCdevkit/VOC2012/Annotations/"
catlist = cp_alliamge()
box = get_all_annotation(path, catlist)
print box.keys()
print [len(box[key]) for key in box.keys()]
print [len(box[key]) for key in box.keys()]