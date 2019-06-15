
from script.video2image import video2image
if __name__ == '__main__':

    video_file_name = \
'./data/go_back_couples_01.mkv'
    image_format = \
'png'
    train_timeF = \
60  
    train_offset = \
0
    valid_timeF = \
6000  
    valid_offset = \
20
    test_timeF = \
6000  
    test_offset = \
40
    train_directrory =  \
'./data/png_train'
    valid_directrory = \
'./data/png_valid'
    test_directrory = \
'./data/png_test'
    video2image(video_file_name, image_format,
                train_timeF, train_offset,
                valid_timeF, valid_offset,
                test_timeF, test_offset,
                train_directrory, valid_directrory, test_directrory)
