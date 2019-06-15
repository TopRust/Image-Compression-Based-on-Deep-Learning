import cupy as cp

import ctypes
import glob

import os
import re
import time

from multiprocessing import Process
from multiprocessing import Queue
import shutil

from chainer import Variable
from chainer import cuda
from chainer import serializers
from chainer import functions as F
import cv2 as cv
import chainer

from script import links

import numpy as np

   
def test_divide_to_batch(batch_number, sample_height, sample_width, sample_y_limit, sample_x_limit, sample, sample_stride, sample_size, offset):

    sample_batch = []
    sample_zero = np.zeros_like(sample)
    if offset > 0:
        sample_zero[0: sample_zero.shape[0] - offset // 2, 0: sample_zero.shape[1] - offset // 2] += sample[offset // 2: sample_zero.shape[0], offset // 2: sample_zero.shape[1]]
    else:
        sample_zero = sample
    for y in range(0, sample_y_limit, sample_stride):
        for x in range(0, sample_x_limit, sample_stride):

            sample_batch.append(sample_zero[y: y + sample_size, x: x + sample_size])

    print(len(sample_batch))
    sample_batch =  np.array(sample_batch)
    sample_batch = np.transpose(sample_batch, (0, 3, 1, 2))
    sample_batch = np.asarray(sample_batch, dtype=np.float32) / 255

    return sample_batch

def create_offset_minibatch(test_fn, batch_number, sample_height, sample_width, sample_y_limit, sample_x_limit, sample_stride, sample_size, offset):


    image = cv.imread(test_fn)
    sample = cv.resize(image, (sample_width, sample_height))

    sample_batch = test_divide_to_batch(batch_number, sample_height, sample_width, sample_y_limit, sample_x_limit, sample, sample_stride, sample_size, offset)
    image_fn = os.path.basename(test_fn).split(".")[0]
    return sample, sample_batch, image_fn

def batch2image(prediction_batch, image_height, image_width, image_y_limit, image_x_limit, crop, image_size, image_stride, sample_height, sample_width, sample):

    image_batch = np.transpose(prediction_batch, (0, 2, 3, 1)) * 255
    image = np.zeros((image_height, image_width, 3))

    i = 0 
    for y in range(0, image_y_limit, image_stride):
        for x in range(0, image_x_limit, image_stride):

            # image[y: y + image_size, x: x + image_size, 0: 3] += image_batch[i]
            # image[y + crop: y + y_height - crop, x + crop: x + x_width -crop, 0: 3] += image_batch[y // y_stride * ((width - y_height)// x_stride + 1) + x // x_stride][crop: -crop, crop: -crop] * 255
            # image[y + crop: y + crop + image_stride, x + crop: x + crop + image_stride, 0: 3] += image_batch[i][crop: crop + image_stride, crop: crop + image_stride]
            image[y: y + image_size, x: x + image_size, 0: 3] += image_batch[i]
            i += 1
    # for y in range(sample_height):
        # for x in range(sample_width):

            # image[2 * y, 2 * x] = sample[y, x]

    return image

def save_image(sample, image_fn, image, out_dir_compression, out_dir_decompression):

    # residual = image_orignal - image 
    # residual_bool = residual > 0
    # residual_1st = np.where(residual_bool, residual, -residual)
    # residual_2nd = np.where(residual_bool, 1, 0)
    # cv.imwrite('{}/{}_1st.png'.format(out_dir_c, image_fn), residual_1st)
    # cv.imwrite('{}/{}_2nd.png'.format(out_dir_c, image_fn), residual_2nd)
    cv.imwrite('{}/{}_sample.jpg'.format(out_dir_compression, image_fn), sample)
    cv.imwrite('{}/{}.png'.format(out_dir_decompression, image_fn), image)


def run(result_dir, predict_epoch,
        model_name, exits_bn, activation_function, number_filter_list, gpu_id, 
        image_format, test_sat_dir, 
        image_size, image_stride, image_height, image_width, crop, offset):
    if crop > 0:
        out_dir = '{}/center_prediction_{}'.format(result_dir, predict_epoch)
    else:
        out_dir = '{}/prediction_{}'.format(result_dir, predict_epoch)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    out_dir_compression = '{}/compression'.format(out_dir)
    if os.path.exists(out_dir_compression):
        shutil.rmtree(out_dir_compression)
    os.makedirs(out_dir_compression)

    out_dir_decompression = '{}/decompression'.format(out_dir)
    if os.path.exists(out_dir_decompression):
        shutil.rmtree(out_dir_decompression)
    os.makedirs(out_dir_decompression)

    start = time.time()

    saved_model = '{}/model_epoch-{}'.format(result_dir, predict_epoch)
    model = model_name(exits_bn, activation_function, number_filter_list)
    serializers.load_npz(saved_model, model)
    if gpu_id >= 0:
        model.to_gpu()
    test_sat_dir = 'data/{}_test'.format(image_format)
    test_fns = glob.glob('{}/*.{}'.format(test_sat_dir, image_format))

    sample_size = image_size // 2
    sample_stride = image_stride // 2
    sample_height = image_height // 2
    sample_width = image_width // 2

    sample_y_limit = sample_height - sample_size + sample_stride
    sample_x_limit = sample_width - sample_size + sample_stride
    image_y_limit = 2 * sample_y_limit
    image_x_limit = 2 * sample_x_limit
    batch_number = sample_y_limit // sample_stride * (sample_x_limit // sample_stride)

    for test_fn in test_fns:
        # first image
        sample, sample_batch, image_fn = create_offset_minibatch(test_fn,
                                                        batch_number, sample_height, sample_width, sample_y_limit, sample_x_limit,
                                                        sample_stride, sample_size, 0)

        print(test_fn)
        sample_batch = Variable(cp.asarray(sample_batch, dtype=cp.float32))

        with chainer.using_config('train', False):
            prediction_batch = cp.asnumpy(model(sample_batch).data)
        image = batch2image(prediction_batch, image_height, image_width, image_y_limit, image_x_limit, crop, image_size, image_stride, sample_height, sample_width, sample)

        # offset image
        sample_off, sample_batch, image_fn = create_offset_minibatch(test_fn,
                                                        batch_number, sample_height, sample_width, sample_y_limit, sample_x_limit,
                                                        sample_stride, sample_size, offset)

        sample_batch = Variable(cp.asarray(sample_batch, dtype=cp.float32))

        with chainer.using_config('train', False):
            prediction_batch = cp.asnumpy(model(sample_batch).data)
        image_offset = batch2image(prediction_batch, image_height, image_width, image_y_limit, image_x_limit, crop, image_size, image_stride, sample_height, sample_width, sample_off)

        image[offset: image.shape[0], offset: image.shape[1]] = (image[offset: image.shape[0], offset: image.shape[1]] + image_offset[0: image.shape[0] - offset, 0: image.shape[1] - offset]) / 2

        save_image(sample, image_fn + '_offset', image, out_dir_compression, out_dir_decompression)
    print(time.time() - start)

