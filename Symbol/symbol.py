import mxnet as mx


# Yolo loss
def YOLO_loss(predict, label):
    """
    predict (params): mx.sym->which is NDarray (tensor), its shape is (batch_size, 7, 7,5 )
    label: same as predict
    """
    # Reshape input to desired shape
    predict = mx.sym.reshape(predict, shape=(-1, 49, 5))
    # shift everything to (0, 1)
    predict_shift = (predict+1)/2
    label = mx.sym.reshape(label, shape=(-1, 49, 5))
    # split the tensor in the order of [prob, x, y, w, h]
    cl, xl, yl, wl, hl = mx.sym.split(label, num_outputs=5, axis=2)
    cp, xp, yp, wp, hp = mx.sym.split(predict_shift, num_outputs=5, axis=2)

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

    # return joint loss
    loss = lossc+lossx+lossy+lossw+lossh
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
    sym = mx.sym.Activation(data=sym, act_type="relu")
    sym = mx.sym.Convolution(data=sym, kernel=(3, 3),
                             num_filter=5, pad=(1, 1),
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
    out = mx.sym.Group([mx.sym.BlockGrad(logit), loss])
    return out
