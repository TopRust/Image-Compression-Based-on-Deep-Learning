
import lmdb
import numpy as np

from chainer import dataset
class Dataset(dataset.DatasetMixin):
    

    def __init__(self, image_path, sample_path):
        
        self.env_image = lmdb.open(image_path) #env environment
        self.txn_image = self.env_image.begin(write=False, buffers=False) #txn 
        self.cur_image = self.txn_image.cursor()

        self.env_sample = lmdb.open(sample_path) #env environment
        self.txn_sample = self.env_sample.begin(write=False, buffers=False) #txn 
        self.cur_sample = self.txn_sample.cursor()


    def __len__(self):

        return self.env_image.stat()['entries']

    def get_example(self, i):
        
        i = str(i).encode()
        image = np.fromstring(
            self.cur_image.get(i), dtype=np.uint8).reshape((3, 40, 40))
        sample = np.fromstring(
            self.cur_sample.get(i), dtype=np.uint8).reshape((3, 20, 20))
        return sample, image

def get_image():

    train = Dataset('./data/lmdb/png_train', './data/lmdb/png_sample_train')
    test = Dataset('./data/lmdb/png_test', './data/lmdb/png_sample_test')

    return train, test

