import numpy as np
import cv2

np.random.seed(0)

def add_noise(img, mu=0, sigma=100):
    # 画素数xRGBのノイズを生成
    noise = np.random.normal(mu, sigma, img.shape)

    # ノイズを付加して8bitの範囲にクリップ
    noisy_img = img.astype(np.float64) + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    return noisy_img

mu = 0
sigma = 250
img = cv2.imread('half/akehara003/in.png', cv2.IMREAD_GRAYSCALE)
noisy_img = add_noise(img, mu, sigma)

cv2.imwrite(f'noisy_img_mu{mu}_sigma{sigma}.png', noisy_img)
