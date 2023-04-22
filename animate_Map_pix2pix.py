import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F
from tqdm import tqdm
import statistics

from PIL import Image
import glob

import os
from os import listdir

import time

from UNetGenerator import UNetGenerator
from Discriminator import MyDiscriminator, Discriminator
from UNet_dataset import PairImges

def labelshow( img, name, label ):
    plt.clf()
    img = torchvision.utils.make_grid( img, nrow=4, padding=16 )
    img = img / 2 + 0.5
    npimg = img.detach().numpy()
    # print( plt.rcParams.keys() )
    plt.figure( figsize = (12, 12) )
    plt.rcParams[ "text.color" ] = 'white'
    plt.rcParams[ "font.size" ] = 20
    # plt.rcParams["font.family"]='Helvetica'
    # plt.rcParams["font.family"]='Impact'
    # plt.rcParams["font.family"]='Tahoma'
    # plt.rcParams["font.family"]='Arial'
    # plt.rcParams["font.family"]='Comic Sans MS'
    # plt.rcParams["font.family"]='Osaka'
    #plt.rcParams["font.family"]='DejaVu Serif'
    plt.rcParams["font.family"]='Ubuntu Mono'
    # plt.font( family = 'Osaka' )
    # plt.text(920, 1090, label) # fontsize = 16
    # plt.text( 860, 1085, "color=white", label )
    plt.text( 860, 1085, label )
    # plt.text( 820, 1085, label )
    plt.axis("off")
    plt.imshow( np.transpose( npimg, (1, 2, 0) ) )
    plt.savefig( name, bbox_inches='tight',pad_inches=0 )
    # plt.show()

def train():
    # 時間計測開始
    time_sta = time.time()

    #input size
    inSize = 256

    # モデル
    device = torch.device( 'cuda:0' if torch.cuda.is_available() else 'cpu' )
    torch.backends.cudnn.benchmark = True

    nEpochs = 1
    args = sys.argv
    if len( args ) == 2:
        nEpochs = int(args[ 1 ] )

    print( ' nEpochs = ', nEpochs )
    print( ' device = ', device )

    # Make directory
    # frame/
    #  ├ Map_train_p2p_(nEpochs)/
    #  | ├ snap00001.png
    #  |       :
    #  └ Map_test_p2p_(nEpochs)/
    #    ├ snap00001.png
    #          :

    frame_train_dir = "Map_train_p2p_" + str( nEpochs ).zfill( 5 )
    if not os.path.exists( "./frame/"+frame_train_dir ):
        os.mkdir( "./frame/"+frame_train_dir )

    frame_test_dir = "Map_test_p2p_" + str( nEpochs ).zfill( 5 )
    if not os.path.exists( "./frame/"+frame_test_dir ):
        os.mkdir( "./frame/"+frame_test_dir )

    # Set Models
    model_G, model_D = UNetGenerator(), MyDiscriminator()
    model_G, model_D = nn.DataParallel(model_G), nn.DataParallel(model_D)
    model_G, model_D = model_G.to(device), model_D.to(device)

    params_G = torch.optim.Adam(model_G.parameters(),
                lr=0.0002, betas=(0.5, 0.999))
    params_D = torch.optim.Adam(model_D.parameters(),
                lr=0.0002, betas=(0.5, 0.999))

    # ロスを計算するためのラベル変数 (PatchGAN)
    ones = torch.ones(32, 1, 4, 4).to(device)
    zeros = torch.zeros(32, 1, 4, 4).to(device)

    # ロスを計算するためのラベル変数 (DCGAN)
    # ones = torch.ones(32).to(device)
    # zeros = torch.zeros(32).to(device)

    # 損失関数
    bce_loss = nn.BCEWithLogitsLoss()
    mae_loss = nn.L1Loss()

    # エラー推移
    result = {}
    result["log_loss_G_sum"] = []
    result["log_loss_G_bce"] = []
    result["log_loss_G_mae"] = []
    result["log_loss_D"] = []

    # 訓練
    transform = transforms.Compose( [transforms.ToTensor(),
                                     transforms.Normalize( (0.5,), (0.5,) ) ] )
    #dataset
    dataset_dir = "./half"
    testset_dir = "./test"
    print(f"dataset_dir: {dataset_dir}")
    print(f"testset_dir: {testset_dir}")

    train_dataset = PairImges(dataset_dir, transform=transform)
    test_dataset = PairImges(testset_dir, transform=transform)

    print( 'size of train_dataset = ', len(train_dataset) )
    print( 'size of test_dataset = ', len(test_dataset) )

    indices = np.arange( len( train_dataset ) )
    frameset = torch.utils.data.Subset( train_dataset, indices[train_size:train_size+16] )

    batch_size = 32

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True )
    frameloader = DataLoader(frameset, batch_size=16, shuffle=False )
    testloader = DataLoader(test_dataset, batch_size=16, shuffle=False )

    print( 'Number of iteration for each epoch = ', len( trainloader ) )

    nBatches = len( trainloader )
    print( 'in train' )

    log_loss_G_sum, log_loss_G_bce, log_loss_G_mae, log_loss_D = [], [], [], []
    for i in range(nEpochs):

        for counter, (ori_img, ans_img) in enumerate( trainloader ):

            print( counter, ' / ', nBatches )

            batch_len = len(ans_img)
            # print( ' batch_len = ', batch_len )
            # print( ' nBatches = ', nBatches )
            ans_img, ori_img = ans_img.to(device), ori_img.to(device)

            # Gの訓練
            # 偽画像を作成
            fake_img = model_G(ori_img)

            # 偽画像を一時保存
            fake_img_tensor = fake_img.detach()


            # 偽画像を本物と騙せるようにロスを計算
            LAMBD = 100.0 # BCEとMAEの係数
            cat_img = torch.cat([fake_img, ori_img], dim=1)
            out = model_D(cat_img)
            # ones_listの最初からbatch_len番目までを取得
            in_ones = ones[:batch_len]
            loss_G_bce = bce_loss(out, in_ones)
            loss_G_mae = LAMBD * mae_loss(fake_img, ans_img)
            loss_G_sum = loss_G_bce + loss_G_mae

            log_loss_G_bce.append(loss_G_bce.item())
            log_loss_G_mae.append(loss_G_mae.item())
            log_loss_G_sum.append(loss_G_sum.item())

            # 微分計算・重み更新
            params_D.zero_grad()
            params_G.zero_grad()
            loss_G_sum.backward()
            params_G.step()

            # Discriminatoの訓練
            # 本物のカラー画像を本物と識別できるようにロスを計算
            real_out = model_D(torch.cat([ans_img, ori_img], dim=1))
            loss_D_real = bce_loss(real_out, ones[:batch_len])

            # 偽の画像の偽と識別できるようにロスを計算
            fake_out = model_D(torch.cat([fake_img_tensor, ori_img], dim=1))
            loss_D_fake = bce_loss(fake_out, zeros[:batch_len])

            # 実画像と偽画像のロスを合計
            loss_D = loss_D_real + loss_D_fake
            log_loss_D.append(loss_D.item())

            # 微分計算・重み更新
            params_D.zero_grad()
            params_G.zero_grad()
            loss_D.backward()
            params_D.step()

        # output.append((fake_img.cpu(), ans_img.cpu()))

        result["log_loss_G_sum"].append(statistics.mean(log_loss_G_sum))
        result["log_loss_G_bce"].append(statistics.mean(log_loss_G_bce))
        result["log_loss_G_mae"].append(statistics.mean(log_loss_G_mae))
        result["log_loss_D"].append(statistics.mean(log_loss_D))
        print(f"epoch={i} " +
              f"log_loss_G_sum = {result['log_loss_G_sum'][-1]} " +
              f"({result['log_loss_G_bce'][-1]}, {result['log_loss_G_mae'][-1]}) " +
              f"log_loss_D = {result['log_loss_D'][-1]}")


        # animation frame recording
        with torch.no_grad():
            for fid, ( frameI, frameO ) in enumerate( frameloader ):

                # save learning results
                frameI = frameI.to( device )

                validated = model_G( frameI )
                print( ' animation frame = ', i+1 )
                snapname = f"./frame/"+frame_train_dir+"/"+"snap" + str( i+1 ).zfill( 5 ) + ".png"
                labelname = "epoch=" + str( i+1 ).zfill( 5 )
                labelshow( validated.detach().reshape( -1, 1, inSize, inSize ).cpu(),
                           snapname, labelname )

            for fid, ( frameI, frameO ) in enumerate( testloader ):

                # save learning results
                frameI = frameI.to( device )

                validated = model_G( frameI )
                print( ' animation frame = ', i+1 )
                snapname = f"./frame/"+frame_test_dir+"/"+"snap" + str( i+1 ).zfill( 5 ) + ".png"
                labelname = "epoch=" + str( i+1 ).zfill( 5 )
                labelshow( validated.detach().reshape( -1, 1, inSize, inSize ).cpu(),
                           snapname, labelname )


    # 時間計測終了
    time_end = time.time()
    # 経過時間（秒）
    tim = time_end - time_sta

    print(f"Execution time: {tim}")

    with open("./log/epoch"+ str( nEpochs ).zfill( 5 ) +"/ExecutionTime.txt", "w") as f:
        f.write(f"Execution time: {tim}")

    print( 'finished' )

    ####### Save log #######
    # log_file_name = "logs_" + str( nEpochs ).zfill( 5 )
    # if not os.path.exists("./"+log_file_name):
    #    os.mkdir("./"+log_file_name)

    #outputの保存
    # print( 'output = ', output )
    # print( 'output size = ', len( output ) )
    # filename_output = "Map_output_pix2pix_" + str( nEpochs ).zfill( 5 ) + ".pth"
    # print( 'saving ', filename_output )
    # torch.save( output, f"./"+log_file_name+"/"+filename_output )

    # if not os.path.exists("./"+log_file_name+"/losses"):
    #    os.mkdir("./"+log_file_name+"/losses")

    #loss_G_sumの保存
    # filename_loss_G_sum = "Map_loss_G_sum_pix2pix_" + str( nEpochs ).zfill( 5 ) + ".pth"
    # print( 'saving ', filename_loss_G_sum )
    # torch.save( log_loss_G_sum, f"./"+log_file_name+f"/losses/"+filename_loss_G_sum )

    # #loss_G_bceの保存
    # filename_loss_G_bce = "Map_loss_G_bce_pix2pix_" + str( nEpochs ).zfill( 5 ) + ".pth"
    # print( 'saving ', filename_loss_G_bce )
    # torch.save( log_loss_G_bce, f"./"+log_file_name+f"/losses/"+filename_loss_G_bce )
    #
    # #loss_G_maeの保存
    # filename_loss_G_mae = "Map_loss_G_mae_pix2pix_" + str( nEpochs ).zfill( 5 ) + ".pth"
    # print( 'saving ', filename_loss_G_mae )
    # torch.save( log_loss_G_mae, f"./"+log_file_name+f"/losses/"+filename_loss_G_mae )

    #loss_Dの保存
    # filename_loss_D = "Map_loss_D_pix2pix_" + str( nEpochs ).zfill( 5 ) + ".pth"
    # print( 'saving ', filename_loss_D )
    # torch.save( log_loss_D, f"./"+log_file_name+f"/losses/"+filename_loss_D )

    # if not os.path.exists("./"+log_file_name+"/models"):
    #        os.mkdir("./"+log_file_name+"/models")

    #model_Gの保存
    # filename_model_G = "Map_model_G_pix2pix_" + str( nEpochs ).zfill( 5 ) + ".pth"
    # print( 'saving ', filename_model_G )
    # torch.save( model_G.state_dict(), f"./"+log_file_name+f"/models/"+filename_model_G )

    #model_Dの保存
    # filename_model_D = "Map_model_D_pix2pix_" + str( nEpochs ).zfill( 5 ) + ".pth"
    # print( 'saving ', filename_model_D )
    # torch.save( model_D.state_dict(), f"./"+log_file_name+f"/models/"+filename_model_D )

if __name__ == "__main__":
    #logファイルとframeファイルの確認
    if not os.path.exists( "./log"):
        print("Make directory ./log")
        os.mkdir( "./log")

    if not os.path.exists( "./frame"):
        print("Make directory ./frame")
        os.mkdir( "./frame")

    train()
