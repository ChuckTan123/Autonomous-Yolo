import mxnet as mx

def expit_tensor(x):
    return 1/(1+mx.sym.exp(-x))


# Yolo loss
def YOLO_loss(predict, label):
    """
    predict (params): mx.sym->which is NDarray (tensor), its shape is (batch_size, 7, 7,5 )
    label: same as predict
    """
    # Reshape input to desired shape
    predict = mx.sym.reshape(predict, shape=(-1, 49, 9))
    # shift everything to (0, 1)
    predict_shift = (predict+1)/2
    label = mx.sym.reshape(label, shape=(-1, 49, 9))
    # split the tensor in the order of [prob, x, y, w, h]
    cl, xl, yl, wl, hl, clsl1, clsl2, clsl3, clsl4 = mx.sym.split(label, num_outputs=9, axis=2)
    cp, xp, yp, wp, hp, clsp1, clsp2, clsp3, clsp4 = mx.sym.split(predict_shift, num_outputs=9, axis=2)
    # clsesl = mx.sym.Concat(clsl1, clsl2, clsl3, clsl4, dim=2)
    # clsesp = mx.sym.Concat(clsp1, clsp2, clsp3, clsp4, dim=2)
    # weight different target differently
    lambda_coord = 5
    lambda_obj = 1
    lambda_noobj = 0.2
    mask = cl*lambda_obj+(1-cl)*lambda_noobj

    # linear regression
    lossc = mx.sym.LinearRegressionOutput(label=cl*mask, data=cp*mask)
    lossx = mx.sym.LinearRegressionOutput(label=xl*cl*lambda_coord, data=xp*cl*lambda_coord)
    lossy = mx.sym.LinearRegressionOutput(label=yl*cl*lambda_coord, data=yp*cl*lambda_coord)
    lossw = mx.sym.LinearRegressionOutput(label=mx.sym.sqrt(wl)*cl*lambda_coord, data=mx.sym.sqrt(wp)*cl*lambda_coord)
    lossh = mx.sym.LinearRegressionOutput(label=mx.sym.sqrt(hl)*cl*lambda_coord, data=mx.sym.sqrt(hp)*cl*lambda_coord)
    losscls1 = mx.sym.LinearRegressionOutput(label=clsl1*cl, data=clsp1*cl)
    losscls2 = mx.sym.LinearRegressionOutput(label=clsl2*cl, data=clsp2*cl)
    losscls3 = mx.sym.LinearRegressionOutput(label=clsl3*cl, data=clsp3*cl)
    losscls4 = mx.sym.LinearRegressionOutput(label=clsl4*cl, data=clsp4*cl)
    losscls = losscls1+losscls2+losscls3+losscls4
    # return joint loss
    loss = lossc+lossx+lossy+lossw+lossh+losscls
    return loss


def fcrn_loss_YoloV2(label, pred, anchors, config):
    """Calculate loss specifically for FCRN
    Need MXNET 0.11 or high34 !!!
    Parameters
    ----------
    pred : mx.Symbol
        predicted label (shape is (batch_size, 1024, 7*N_anchors))
    label : mx.Symbol
        ground truth label (shape is (batch_size, 7168))
    Returns
    -------
    float
    """

    # todo: test the function step by step
    # meta data
    sprob = 1
    snoob = 0.5
    scoor = 5
    size_H, size_W, _ = (32,32)  # here we define the size is 32 by 32
    B = 5
    HW = size_H * size_W  # number of grid cells

    # Read anchors, they should be 0<w,h<1
    anchors_w, anchors_h = mx.sym.split(anchors, axis=1, num_outputs=2, name="anchor_split")

    # Read label
    label_reshape = mx.sym.reshape(label, [-1, size_H, size_W, 9])
    # |x_l|<0.5, |y_l|<0.5, 0<w_l & h_l<1, 0<cos_l<1, |sin_l|<1
    prob_l, x_l, y_l, w_l, h_l, cls1, cls2, cls3, cls4 = mx.sym.split(label_reshape, axis=3, num_outputs=9, name="label_split")
    # To compute the upperleft point xy and bottomright point xy, it is based on 32(for example), not the real WH

    upperleft_x_l = x_l - w_l * size_W * 0.5
    upperleft_y_l = y_l - h_l * size_H * 0.5
    bottomright_x_l = x_l + w_l * size_W * 0.5
    bottomright_y_l = y_l + h_l * size_H * 0.5
    upperleft_x_l =  -w_l * size_W * 0.5
    upperleft_y_l = -h_l * size_H * 0.5
    bottomright_x_l = w_l * size_W * 0.5
    bottomright_y_l = h_l * size_H * 0.5
    area_l = (w_l * h_l) * (size_W * size_H)

    # Extrac the coordinate prediction from pred
    pred_reshape = mx.sym.reshape(pred, [-1, size_H, size_W, B, 7])
    prob_p, x_p, y_p, wr_p, hr_p, cls1, cls2, cls3, cls4 = mx.sym.split(pred_reshape, axis=4, num_outputs=9, name="pred_split")
    x_adjust = expit_tensor(x_p) - 0.5  # sigmoid and shift it to -0.5~0.5
    y_adjust = expit_tensor(y_p) - 0.5  # sigmoid and shift it to -0.5~0.5
    w_adjust = mx.sym.sqrt(
        mx.sym.broadcast_mul(mx.sym.exp(wr_p), mx.sym.reshape(anchors_w, shape=[1, 1, B, 1])))
    h_adjust = mx.sym.sqrt(
         mx.sym.broadcast_mul(mx.sym.exp(hr_p), mx.sym.reshape(anchors_h, shape=[1, 1, B, 1])))
    prob_adjust = expit_tensor(prob_p)
    cls1p = expit_tensor(cls1)
    cls2p = expit_tensor(cls2)
    cls3p = expit_tensor(cls3)
    cls4p = expit_tensor(cls4)

    # Get predict upperleft and bottomright
    w_p = w_adjust ** 2
    h_p = h_adjust ** 2
    upperleft_x_p = x_p - w_p * size_W * 0.5  # will be negative, but it doesnt affect anything for computing IOU
    upperleft_y_p = y_p - h_p * size_H * 0.5  # will be negative, but it doesnt affect anything for computing IOU
    bottomright_x_p = x_p + w_p * size_W * 0.5  # will be negative, but it doesnt affect anything for computing IOU
    bottomright_y_p = y_p + h_p * size_H * 0.5  # will be negative, but it doesnt affect anything for computing IOU
    area_pred = (w_p * h_p) * (size_W * size_H)

    # caculate intersection areas
    intersect_upleft_x = mx.sym.broadcast_maximum(upperleft_x_p, mx.sym.expand_dims(upperleft_x_l, axis=-1))
    intersect_upleft_y = mx.sym.broadcast_maximum(upperleft_y_p, mx.sym.expand_dims(upperleft_y_l, axis=-1))
    intersect_botright_x = mx.sym.broadcast_minimum(bottomright_x_p, mx.sym.expand_dims(bottomright_x_l, axis=-1))
    intersect_botright_y = mx.sym.broadcast_minimum(bottomright_y_p, mx.sym.expand_dims(bottomright_y_l, axis=-1))
    intersect_w = mx.sym.maximum(intersect_botright_x - intersect_upleft_x, 0)
    intersect_h = mx.sym.maximum(intersect_botright_y - intersect_upleft_y, 0)
    intersect = intersect_w * intersect_h

    # calculate the best IOU, set 0.0 confidence for worse boxes
    iou = intersect / mx.sym.broadcast_add(area_pred - intersect, mx.sym.expand_dims(area_l, axis=3))
    best_box = mx.sym.broadcast_equal(iou, mx.sym.max_axis(iou, axis=3, keepdims=True)) * 1.0
    prob_anchor_l = mx.sym.broadcast_mul(best_box, mx.sym.expand_dims(prob_l, axis=-1))
    x_anchor_l = mx.sym.broadcast_mul(best_box, mx.sym.expand_dims(x_l, axis=-1), name="x_anchor_l")
    y_anchor_l = mx.sym.broadcast_mul(best_box, mx.sym.expand_dims(y_l, axis=-1), name="y_anchor_l")
    w_anchor_l = mx.sym.broadcast_mul(best_box, mx.sym.expand_dims(w_l, axis=-1), name="w_anchor_l")
    h_anchor_l = mx.sym.broadcast_mul(best_box, mx.sym.expand_dims(h_l, axis=-1), name="h_anchor_l")

    # set up weighted sum
    mask = (prob_anchor_l * 5 + (1 - prob_anchor_l) * .5)  # balance box and non-boxes

    loss_prob = mx.sym.LinearRegressionOutput(data=prob_adjust * mask, label=prob_anchor_l * mask, grad_scale=1,
                                              name="lossprob")
    loss_x = mx.sym.LinearRegressionOutput(data=mx.sym.broadcast_mul(x_adjust, best_box), label=x_anchor_l,
                                         grad_scale=scoor, name="lossx")
    loss_y = mx.sym.LinearRegressionOutput(data=y_adjust, label=y_anchor_l, grad_scale=scoor, name="lossy")
    loss_w = mx.sym.LinearRegressionOutput(data=w_adjust, label=w_anchor_l, grad_scale=scoor, name="lossw")
    loss_h = mx.sym.LinearRegressionOutput(data=h_adjust, label=h_anchor_l, grad_scale=scoor, name="lossh")
    loss_cls1 = mx.sym.LinearRegressionOutput(data=cls1p, label=cls1, grad_scale=scoor, name="losscos")
    loss_cls2 = mx.sym.LinearRegressionOutput(data=cls2p, label=cls2, grad_scale=scoor, name="losssin")
    loss_cls3 = mx.sym.LinearRegressionOutput(data=cls3p, label=cls3, grad_scale=scoor, name="losssin")
    loss_cls4 = mx.sym.LinearRegressionOutput(data=cls4p, label=cls4, grad_scale=scoor, name="losssin")
    loss = loss_prob + loss_x + loss_y + loss_w + loss_h + loss_cls1 + loss_cls2 + loss_cls3 + loss_cls4
    return loss


# Get pretrained imagenet model
def get_resnet_model(model_path, epoch):
    # not necessary to be this name, you can do better
    label = mx.sym.Variable('softmax_label')
    # load symbol and actual weights
    sym, args, aux = mx.model.load_checkpoint(model_path, epoch)
    # extract last bn layer
    sym = sym.get_internals()['bn1_output']
    # append two layers
    sym = mx.sym.Activation(data=sym, act_type="relu", name="relu_final")
    sym = mx.sym.Convolution(data=sym, kernel=(3, 3),
                             num_filter=9, pad=(1, 1),
                             stride=(1, 1), no_bias=True,
                             )
    # get softsign
    sym = sym / (1 + mx.sym.abs(sym))
    logit = mx.sym.transpose(sym, axes=(0, 2, 3, 1), name="logit") # (-1, 7, 7, 5(c,x,y,w,h))
    # apply loss
    loss_ = YOLO_loss(logit, label)
    # mxnet special requirement
    loss = mx.sym.MakeLoss(loss_)
    # multi-output logit should be blocked from generating gradients
    out = mx.sym.Group([loss, mx.sym.BlockGrad(logit)])
    return out
