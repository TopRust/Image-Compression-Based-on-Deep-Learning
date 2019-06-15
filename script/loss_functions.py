
import chainer.functions as F

def nlogn_loss(prediction, label):

    residual = prediction * 255 - label * 255
    diff_abs = F.absolute(residual) + 1 
    loss = F.mean(diff_abs * F.log2(diff_abs) / 256)

    return loss

def square_loss(prediction, label):
    residual = prediction * 255 - label * 255
    diff_square = F.square(residual)
    loss = F.sqrt(F.mean(diff_square)) / 144
    return loss

def center_square_loss(prediction, label):
    residual = prediction[:, :, 5:-5, 5:-5]* 255 - label * 255
    diff_square = F.square(residual)
    loss = F.sqrt(F.mean(diff_square)) / 144
    return loss

def center_corner_nlogn_loss(prediction, label):

    residual = prediction * 255 - label * 255
    diff_abs = F.absolute(residual) + 1
    diff_abs.array[:,:, 0:4, 0:4] += diff_abs.array[:,:, 0:4, 0:4]
    diff_abs.array[:,:,0:4, -4: ] += diff_abs.array[:,:,0:4, -4: ]
    diff_abs.array[:,:,-4: , 0: 4] += diff_abs.array[:,:,-4: , 0: 4]
    diff_abs.array[:,:,-4: , -4: ] += diff_abs.array[:,:,-4: , -4: ]
    diff_abs.array[:,:,diff_abs.shape[2] // 2 - 4: diff_abs.shape[2] // 2 + 4, diff_abs.shape[3] // 2 - 4: diff_abs.shape[3] // 2 + 4 ] += \
    diff_abs.array[:,:,diff_abs.shape[2] // 2 - 4: diff_abs.shape[2] // 2 + 4, diff_abs.shape[3] // 2 - 4: diff_abs.shape[3] // 2 + 4 ]
    loss = F.mean(diff_abs * F.log2(diff_abs) / 256)

    return loss
