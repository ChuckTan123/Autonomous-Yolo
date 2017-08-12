import mxnet as mx
import numpy as np
import cv2


# get iterator
def get_iterator(path, data_shape, label_width, batch_size, shuffle=False):
    iterator = mx.io.ImageRecordIter(path_imgrec=path,
                                    data_shape=data_shape,
                                    label_width=label_width,
                                    batch_size=batch_size,
                                    shuffle=shuffle)
    return iterator


# Convert data to rec file
def get_YOLO_xy(bxy, grid_size=(7,7), dscale=32, sizet=224):
    cx, cy = bxy
    assert cx<=1 and cy<=1, "All should be < 1, but get {}, and {}".format(cx,cy)

    j = int(np.floor(cx/(1.0/grid_size[0])))
    i = int(np.floor(cy/(1.0/grid_size[1])))
    xyolo = (cx * sizet - j * dscale) / dscale
    yyolo = (cy * sizet - i * dscale) / dscale
    return [i, j, xyolo, yyolo]


# Get YOLO label
def imgResizeBBoxTransform(img, bbox, sizet, grid_size=(7,7,5), dscale=32):

    himg, wimg = img.shape[:2]
    imgR = cv2.resize(img, dsize=(sizet, sizet))
    bboxyolo = np.zeros(grid_size)
    for eachbox in bbox:
        cx, cy, w, h = eachbox
        cxt = 1.0*cx/wimg
        cyt = 1.0*cy/himg
        wt = 1.0*w/wimg
        ht = 1.0*h/himg
        assert wt<1 and ht<1
        i, j, xyolo, yyolo = get_YOLO_xy([cxt, cyt], grid_size, dscale, sizet)
        print "one yolo box is {}".format((i, j, xyolo, yyolo, wt, ht))
        label_vec = np.asarray([1, xyolo, yyolo, wt, ht])
        bboxyolo[i, j, :] = label_vec
    return imgR, bboxyolo


# Convert raw images to rec files
def toRecFile(imgroot, imglist, annotation, sizet, grid_size, dscale, name):

    record = mx.recordio.MXIndexedRecordIO(name+".idx",
                                           name+".rec", 'w')
    for i in range(len(imglist)):
        imgname = imglist[i]
        img = cv2.imread(imgroot+imgname+'.jpg')
        bbox = annotation[imgname]
        print "Now is processing img {}".format(imgname)
        imgR, bboxR = imgResizeBBoxTransform(img, bbox, sizet, grid_size, dscale)
        header = mx.recordio.IRHeader(flag=0, label=bboxR.flatten(), id=0, id2=0)
        s = mx.recordio.pack_img(header, imgR, quality=100, img_fmt='.jpg')
        record.write_idx(i, s)
    print "JPG to rec is Done"
    record.close()

if __name__ == "__main__":
    # transform jpg to rec file
    imgroot = "./DATA/"
    annotation = np.load("./DATA/annotation_list.npy")[()]
    imglist = annotation.keys()
    sizet = 224
    name = "cat"
    toRecFile(imgroot, imglist, annotation, sizet, (7,7,5), 32, name)