# -*- coding: utf-8 -*-

import os
import shutil
import random
import numpy as np
import cv2
import sys


#
def list_files(directory):
    files = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if not item.startswith('.'):  # ファイルでかつ"."で始まらない場合のみ追加する
            files.append(item_path)
    return files

#
def add_noise(img, mu, sigma):
    # 画素数xRGBのノイズを生成
    noise = np.random.normal(mu, sigma, img.shape)

    # ノイズを付加して8bitの範囲にクリップ
    noisy_img = img.astype(np.float64) + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    return noisy_img

#
def process_images(directory, mu, sigma):
    image_files = list_files(directory)
    before_num_images = len(image_files)

    # print(num_images)

    for image_folder in image_files:
        # Make noise
        noise_dir = image_folder+'_noise'
        shutil.copytree(image_folder, noise_dir)

        img = cv2.imread(noise_dir+'/in.png', cv2.IMREAD_GRAYSCALE)
        noisy_img = add_noise(img, mu, sigma)

        cv2.imwrite(noise_dir+'/in.png', noisy_img)

        # Make rotate
        # rotate 90
        r90_dir = image_folder+'_rotate90'
        shutil.copytree(image_folder, r90_dir)
        in_img = cv2.imread(r90_dir+'/in.png', cv2.IMREAD_GRAYSCALE)
        out_img = cv2.imread(r90_dir+'/out.png', cv2.IMREAD_GRAYSCALE)
        r90_in_img = cv2.rotate(in_img, cv2.ROTATE_90_CLOCKWISE)
        r90_out_img = cv2.rotate(out_img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(r90_dir+'/in.png', r90_in_img)
        cv2.imwrite(r90_dir+'/out.png', r90_out_img)

        # rotate 180
        r180_dir = image_folder+'_rotate180'
        shutil.copytree(image_folder, r180_dir)
        in_img = cv2.imread(r180_dir+'/in.png', cv2.IMREAD_GRAYSCALE)
        out_img = cv2.imread(r180_dir+'/out.png', cv2.IMREAD_GRAYSCALE)
        r180_in_img = cv2.rotate(in_img, cv2.ROTATE_180)
        r180_out_img = cv2.rotate(out_img, cv2.ROTATE_180)
        cv2.imwrite(r180_dir+'/in.png', r180_in_img)
        cv2.imwrite(r180_dir+'/out.png', r180_out_img)

        # rotate 270
        r270_dir = image_folder+'_rotate270'
        shutil.copytree(image_folder, r270_dir)
        in_img = cv2.imread(r270_dir+'/in.png', cv2.IMREAD_GRAYSCALE)
        out_img = cv2.imread(r270_dir+'/out.png', cv2.IMREAD_GRAYSCALE)
        r270_in_img = cv2.rotate(in_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        r270_out_img = cv2.rotate(out_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(r270_dir+'/in.png', r270_in_img)
        cv2.imwrite(r270_dir+'/out.png', r270_out_img)

        # Make inversion
        # upside down
        ud_dir = image_folder+'_ud'
        shutil.copytree(image_folder, ud_dir)
        in_img = cv2.imread(ud_dir+'/in.png', cv2.IMREAD_GRAYSCALE)
        out_img = cv2.imread(ud_dir+'/out.png', cv2.IMREAD_GRAYSCALE)
        ud_in_img = cv2.flip(in_img, 0)
        ud_out_img = cv2.flip(out_img, 0)
        cv2.imwrite(ud_dir+'/in.png', ud_in_img)
        cv2.imwrite(ud_dir+'/out.png', ud_out_img)

        # mirror
        mirror_dir = image_folder+'_mirror'
        shutil.copytree(image_folder, mirror_dir)
        in_img = cv2.imread(mirror_dir+'/in.png', cv2.IMREAD_GRAYSCALE)
        out_img = cv2.imread(mirror_dir+'/out.png', cv2.IMREAD_GRAYSCALE)
        mirror_in_img = cv2.flip(in_img, 1)
        mirror_out_img = cv2.flip(out_img, 1)
        cv2.imwrite(mirror_dir+'/in.png', mirror_in_img)
        cv2.imwrite(mirror_dir+'/out.png', mirror_out_img)

    # Result Augumentataion
    image_files = list_files(directory)
    after_num_images = len(image_files)
    print(f'Pair images went {before_num_images} -> {after_num_images}')


if __name__ == "__main__":

    #For Noise
    mu = 0
    sigma = 100

    image_directory = input('Please enter the file to which you want to extend the data -> ')

    aug_directory = 'Aug_'+image_directory+f'_mu{mu}_sig{sigma}'

    if not os.path.exists( aug_directory ):
        shutil.copytree(image_directory, aug_directory)

        process_images(aug_directory+'/', mu, sigma)

        print(f'Create {aug_directory} directory.')

    else:
        print('This augmented file already exists')
