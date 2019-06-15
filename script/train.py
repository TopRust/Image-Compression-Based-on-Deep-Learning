
import sys

import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

from chainer.training import extensions
from chainer.datasets import TransformDataset

from dataset import get_image

import os
import imp

from script import links

def trans(in_data):

    sample, label = in_data
    sample = np.array(sample, dtype=np.float32) / 255
    label = np.array(label, dtype=np.float32) / 255
    return sample, label


def run(batch_size, n_process, prefetch,
        model_name, exits_bn, activation_function, number_filter_list,
        gpu_id, lossfun, learning_rate, max_epoch, out_dir, epoch):
    train, test = get_image()
    train = TransformDataset(train, trans)
    test = TransformDataset(test, trans)


    train_iter = iterators.MultiprocessIterator(train, batch_size, True, True, n_process, prefetch)
    test_iter = iterators.MultiprocessIterator(test, batch_size, False, False, n_process, prefetch)


    model = model_name(exits_bn, activation_function, number_filter_list)
    
    if gpu_id >= 0:
	    model.to_gpu(gpu_id)

    # Wrap your model by Classifier and include the process of loss calculation within your model.
    # Since we do not specify a loss function here, the default 'softmax_cross_entropy' is used.

    model = links.Loss_Classifier(model, lossfun)
    # selection of your optimizing method
    optimizer = optimizers.MomentumSGD(lr=learning_rate, momentum=0.9)

    # Give the optimizer a reference to the model
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

    # Get an updater that uses the Iterator and Optimizer
    updater = training.updaters.StandardUpdater(train_iter, optimizer, device=gpu_id)

    # Setup a Trainer
    trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='{}'.format(out_dir))

    from chainer.training import extensions

    trainer.extend(extensions.LogReport()) # generate report
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}')) # save updater
    trainer.extend(extensions.snapshot_object(model.predictor, filename='model_epoch-{.updater.epoch}')) # save model
    trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id)) # validation

    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time'])) # show loss and accuracy
    trainer.extend(extensions.ProgressBar()) # show trainning progress
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png')) # loss curve
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png')) # accuracy curve
    trainer.extend(extensions.dump_graph('main/loss'))
    if epoch > 0:
        serializers.load_npz('./{}/snapshot_epoch-{}'.format(out_dir, epoch), trainer)
        trainer.updater.get_optimizer('main').lr = learning_rate
    trainer.run()

