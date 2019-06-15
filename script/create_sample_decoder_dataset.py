import glob
import os
import shutil
import time

import numpy as np

import cv2 as cv
import lmdb

def create_image_sample_patches(image_patch_size, stride, image_dir, image_db_dir, sample_db_dir):

    if os.path.exists(image_db_dir):
        shutil.rmtree(image_db_dir)
    if os.path.exists(sample_db_dir):
        shutil.rmtree(sample_db_dir)

    os.makedirs(image_db_dir)
    os.makedirs(sample_db_dir)
    # db
    image_env = lmdb.Environment(image_db_dir, map_size=1099511627776)
    image_txn = image_env.begin(write=True, buffers=False)

    sample_env = lmdb.Environment(sample_db_dir, map_size=1099511627776)
    sample_txn = sample_env.begin(write=True, buffers=False)

    print('patch size:', image_patch_size, stride)

    # get filenames
    image_fns = np.asarray(sorted(glob.glob('%s/*.png*' % image_dir)))

    index = np.arange(len(image_fns))
    np.random.shuffle(index)
    image_fns = image_fns[index]

    n_all_files = len(image_fns)
    print('n_all_files:', n_all_files)

    n_patches = 0
    for file_i, image_fn in enumerate(image_fns):

        image_im = cv.imread(image_fn, cv.IMREAD_COLOR)

        st = time.time()
        image_patches = []
        sample_patches = []
        row = image_im.shape[0]
        col = image_im.shape[1]

        for y in range(0, row - image_patch_size + stride, stride):
            for x in range(0, col - image_patch_size + stride, stride):
                patch = image_im[row_cur: row_cur + image_patch_size, col_cur: col_cur + image_patch_size, 0: 3]
                sample = cv.resize(patch, (patch.shape[1] // 2, patch.shape[0] // 2))
                image_patches.append(patch)
                sample_patches.append(sample)

        print('divide:{}'.format(time.time() - st))
        image_patches = np.array(image_patches, dtype=np.uint8)
        image_patches = np.transpose(image_patches, (0, 3, 1, 2))
        sample_patches = np.array(sample_patches, dtype=np.uint8)
        sample_patches = np.transpose(sample_patches, (0, 3, 1, 2))

        for patch_i in range(image_patches.shape[0]):
            image_patch = image_patches[patch_i]
            sample_patch = sample_patches[patch_i]
            bn = str(n_patches).encode()
            image_txn.put(bn, image_patch.tobytes())
            sample_txn.put(bn, sample_patch.tobytes())
            n_patches += 1

        print(file_i, '/', n_all_files, 'n_patches:', n_patches)
    
    image_txn.commit()
    image_env.close()
    sample_txn.commit()
    sample_env.close()
    print('patches:\t', n_patches)

def run(size_crop, stride_crop, format_image):

    for type_dataset in ['train', 'valid', 'test']:
        create_image_sample_patches(size_crop, stride_crop,
                    'data/{}_{}'.format(format_image, type_dataset),
                    'data/lmdb/{}_{}'.format(format_image, type_dataset),
                    'data/lmdb/{}_sample_{}'.format(format_image, type_dataset))

