#You need opencv-python. Please install with the following command.
#pip install opencv-python

import glob
import cv2
import os

def makeVideo(dir, epoch, file_type = 'train'):

    img_array = []

    print(f'Load dir: {dir}')
    print(f'Make type: {file_type}')

    for filename in sorted(glob.glob(dir+"/*.png")):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    name = './video/anime_'+ file_type + '_' + str( epoch ).zfill( 5 ) +'.mp4'

    #Make 30sec video
    fps = int(epoch)/30
    if fps*1000 > 65535:
        fps = round(fps)
    out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

if __name__ == "__main__":

    if not os.path.exists( "./video" ):
        os.mkdir( "./video" )

    dir_name = input('What is the nume of dir? -> ')
    epoch_num = input('What is the number of epoch? -> ')
    file_ty = input('What is the file type? -> ')

    makeVideo(dir_name, epoch_num, file_ty)

    print('finish!')
