import os
import shutil
import random
import numpy as np
import cv2
import sys

def add_noise(img, mu=0, sigma=100):
    # 画素数xRGBのノイズを生成
    noise = np.random.normal(mu, sigma, img.shape)

    # ノイズを付加して8bitの範囲にクリップ
    noisy_img = img.astype(np.float64) + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    return noisy_img

def list_files(directory):
    files = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if not item.startswith('.'):  # ファイルでかつ"."で始まらない場合のみ追加する
            files.append(item_path)
    return files

def process_images(directory, processing_ratio):
    image_files = list_files(directory)
    num_images = len(image_files)

    # print(num_images)

    num_process = int(num_images * processing_ratio)

    # ランダムに処理する画像を選択
    images_to_process = random.sample(image_files, num_process)

    for image_folder in images_to_process:
        noise_dir = image_folder+'_noise'
        shutil.copytree(image_folder, noise_dir)

        img = cv2.imread(noise_dir+'/in.png', cv2.IMREAD_GRAYSCALE)
        noisy_img = add_noise(img, mu, sigma)

        cv2.imwrite(noise_dir+'/in.png', noisy_img)


if __name__ == "__main__":
    # 使用例
    np.random.seed(0)

    image_directory = 'half'
    processing_ratio = 0.1  # 10%の画像を処理する
    mu = 0
    sigma = 100

    args = sys.argv
    if len( args ) == 4:
        processing_ratio = float(args[ 1 ])
        mu = int(args[ 2 ])
        sigma = int(args[ 3 ])

    #noiseフォルダ作成
    cp_directory = image_directory+f'_nz{int(processing_ratio*100)}%_mu{mu}_sig'+str(sigma)
    if not os.path.exists( cp_directory ):
        shutil.copytree(image_directory, cp_directory)
        image_directory = cp_directory+'/'

        process_images(image_directory, processing_ratio)

        print(f'Create {cp_directory} directory.')

        # cp_dir = list_files(image_directory)
        # print(len(cp_dir))