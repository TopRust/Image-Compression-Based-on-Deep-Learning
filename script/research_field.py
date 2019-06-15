import cv2 as cv

image_fn = 'data/png_test/40.png'
image = cv.imread(image_fn)
for (i, j) in [(1000, 800)]:
    
    # print(image[i, j] - image[i, j + 1])
    # print(image[i, j] - image[i + 1, j])
    # print(image[i, j] - image[i + 1, j + 1])
    print(image[i, j])
    print(image[i, j + 1])
    print(image[i + 1, j])
    print(image[i + 1, j + 1])

print('------------------------')
image_fn = 'compression_result_square_loss/prediction_14/decompression/40.png'
image = cv.imread(image_fn)
for (i, j) in [(1000, 800)]:
    
    # print(image[i, j] - image[i, j + 1])
    # print(image[i, j] - image[i + 1, j])
    # print(image[i, j] - image[i + 1, j + 1])
    print(image[i, j])
    print(image[i, j + 1])
    print(image[i + 1, j])
    print(image[i + 1, j + 1])

