import sys

# video2image------------------------------------
from script import video2image

video_file_name = './data/E01.mkv'
image_format = 'jpg'
train_timeF = 60  
train_offset = 0
valid_timeF = 6000  
valid_offset = 20
test_timeF = 6000  
test_offset = 40
train_directrory =  './data/jpg_train'
valid_directrory = './data/jpg_valid'
test_directrory = './data/jpg_test'

# create_sample_decoder_dataset------------------------------
from script import create_sample_decoder_dataset

size_crop = 40
stride_crop = 20
format_crop = 'png'

# train_network---------------------------------------
from script import train
from script import links
import chainer.functions as F
from script import loss_functions

batch_size = 512
n_process = 9
prefetch = 2

model_name = links.Decon_Network
# model_name = links.Center_Decon_Network
exits_bn = True
activation_function = F.relu
number_filter_list = [8 * i + 3 for i in range(10)]
number_filter_list.append(3)

gpu_id = 0
# lossfun = loss_functions.center_corner_nlogn_loss 
lossfun = loss_functions.square_loss

learning_rate = 0.004
max_epoch = 500
out_dir = '6times_nlogn'
train_epoch = 0

# prediction----------------------------------------
from script import predict

predict_epoch = 29
image_format = 'jpg'
test_sat_dir = 'data/png_test'
image_height = 1080
image_width = 1920

image_stride = 40
offset = 8
crop = 0
# prediction----------------------------------------
from script import offset_predict
# import evalluation

if __name__ == '__main__':

    if sys.argv[1] == 'video2image':
        video2image.run(video_file_name, image_format,
                    train_timeF, train_offset,
                    valid_timeF, valid_offset,
                    test_timeF, test_offset,
                    train_directrory, valid_directrory, test_directrory)
    elif sys.argv[1] == 'create_dataset':
        create_sample_decoder_dataset.run(size_crop, stride_crop, format_crop)
    elif sys.argv[1] == 'train_network':
        train.run(batch_size, n_process, prefetch,
                model_name, exits_bn, activation_function, number_filter_list,
                gpu_id, lossfun, learning_rate, max_epoch, out_dir, train_epoch)
    elif sys.argv[1] == 'prediction':
        predict.run(out_dir, predict_epoch,
                model_name, exits_bn, activation_function, number_filter_list, gpu_id, 
                image_format, test_sat_dir, 
                size_crop, image_stride, image_height, image_width, crop)
    elif sys.argv[1] == 'offset_prediction':
        offset_predict.run(out_dir, predict_epoch,
                model_name, exits_bn, activation_function, number_filter_list, gpu_id, 
                image_format, test_sat_dir, 
                size_crop, image_stride, image_height, image_width, crop, offset)
    # elif sys.argv[1] == 'evalluation':
    #     evalluation.run()
    
    else:
        print('Usage: python ./script/fucntions create_dataset')

