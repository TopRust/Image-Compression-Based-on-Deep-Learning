# from script.create_residual_dataset import create_residual_patches 
from script.create_single_patch_dataset import create_single_patches 

if __name__ == '__main__':

    create_single_patches(72, 24,
                   'data/png_valid',
                   'data/lmdb/png_valid')
    create_single_patches(72, 24,
                   'data/png_test',
                   'data/lmdb/png_test')
    create_single_patches(72, 24,
                   'data/png_train',
                   'data/lmdb/png_train')

# if __name__ == '__main__':
# 
#     create_residual_patches(15, 5,
#                   'data/png_valid',
#                   'data/lmdb/png_valid')
#     create_residual_patches(15, 5,
#                    'data/png_test',
#                   'data/lmdb/png_test')
#     create_residual_patches(15, 5,
###                 'data/png_train',
#                   'data/lmdb/png_train')
