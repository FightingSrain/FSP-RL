import copy
import time
import math
import numpy as np
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from config import config
import State as State
from pixelwise_a3c import PixelWiseA3C_InnerState
from utils import init_net, savevaltocsv, patin_val
from mini_batch_loader import MiniBatchLoader
# from Net.unet_gru import Actor
from FCN import Actor
from tqdm import tqdm
import os
from Myloss import pixel_color_rate

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

TRAINING_DATA_PATH = "train.txt"
TESTING_DATA_PATH = "train.txt"
VAL_DATA_PATH = "val.txt"
IMAGE_DIR_PATH = "..//"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.manual_seed(1234)
np.random.seed(1234)


def main():
    model = init_net(Actor().to(device), 'kaiming', gpu_ids=[])
    # model.load_state_dict(torch.load("torch_initweight/sig50_gray.pth"))
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    i_index = 0

    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH,
        TESTING_DATA_PATH,
        VAL_DATA_PATH,
        IMAGE_DIR_PATH,
        config.corp_size)

    current_state = State.State((config.BATCH_SIZE, 3, config.img_size, config.img_size), config.MOVE_RANGE)
    agent = PixelWiseA3C_InnerState(model, optimizer, config.BATCH_SIZE, config.EPISODE_LEN, config.GAMMA)

    # train dataset
    train_data_size = MiniBatchLoader.count_paths(TRAINING_DATA_PATH)
    indices = np.random.permutation(train_data_size)

    # val dataset
    val_data_size = MiniBatchLoader.count_paths(VAL_DATA_PATH)
    indices_val = np.random.permutation(val_data_size)

    r_val = indices_val
    raw_val = mini_batch_loader.load_val_data(r_val)
    len_val = len(raw_val)
    ValData = []
    pre_pnsr = -math.inf


    for n_epi in tqdm(range(0, 100000), ncols=70, initial=0):

        r = indices[i_index: i_index + config.BATCH_SIZE]
        raw_x = mini_batch_loader.load_training_data(r)
        h_t1 = np.zeros([config.BATCH_SIZE, model.nf, config.img_size, config.img_size], dtype=np.float32)
        h_t2 = np.zeros([config.BATCH_SIZE, model.nf, config.img_size, config.img_size], dtype=np.float32)

        label = copy.deepcopy(raw_x)
        # 高斯噪声
        raw_n = np.random.normal(0, config.sigma, label.shape).astype(label.dtype) / 255.
        ori_ins_noisy = np.clip(label + raw_n, a_min=0., a_max=1.)
        # 泊松噪声
        # noise_level = 50
        # noise_level = 30
        # noise_level = 10
        # noisy = torch.poisson(noise_level * torch.tensor(label)) / noise_level
        # ori_ins_noisy = np.clip(noisy.numpy(), a_min=0., a_max=1.)

        if n_epi % 10 == 0:
            image = np.asanyarray(label[10].transpose(1, 2, 0) * 255, dtype=np.uint8)
            image = np.squeeze(image)
            cv2.imshow("label", image)
            cv2.waitKey(1)

        current_state.reset(ori_ins_noisy.copy())
        reward = np.zeros((config.BATCH_SIZE * 3, 1, config.img_size, config.img_size))
        sum_reward = 0


        for t in range(config.EPISODE_LEN):

            if n_epi % 10 == 0:
                image = np.asanyarray(current_state.image[10].transpose(1, 2, 0) * 255, dtype=np.uint8)
                image = np.squeeze(image)
                cv2.imshow("temp", image)
                cv2.waitKey(1)

            previous_image = np.clip(copy.deepcopy(current_state.image), a_min=0., a_max=1.)

            action, action_par, action_prob, h_t1, h_t2, tst_act = agent.act_and_train(current_state.tensor,
                                                                                       h_t1, h_t2, reward)

            current_state.step(action, action_par)

            if n_epi % 150 == 0:
                paint_amap(tst_act[10*3])

            reward = 255 * (np.square(label - previous_image) -
                            np.square(label - current_state.image))

            reward = reward.reshape(reward.shape[0]*3, 1, reward.shape[2], reward.shape[3])
            sum_reward += np.mean(reward) * np.power(config.GAMMA, t)

        agent.stop_episode_and_train(current_state.tensor, reward, True)

        torch.cuda.empty_cache()

        if n_epi % 100 == 0 and n_epi != 0:
            temp_psnr, temp_ssim = agent.val(agent, State, raw_val, config.EPISODE_LEN)
            if temp_psnr > pre_pnsr:
                pre_pnsr = temp_psnr
                for f in os.listdir("./GaussianModel_{}/".format(config.sigma)):
                    if os.path.splitext(f)[1] == ".pth":
                        os.remove("./GaussianModel_{}/{}".format(config.sigma, f))
                torch.save(model.state_dict(),
                       "./GaussianModel_{}/{}_{:.4f}_{:.4f}.pth".
                           format(config.sigma, n_epi, temp_psnr, temp_ssim))
                print("save model")
            ValData.append([n_epi, temp_psnr, temp_ssim])
            # savevaltocsv(ValData, "val.csv", config.sigma)  # 保存验证集数据
            patin_val(np.asarray(ValData)[:, 1])

        if i_index + config.BATCH_SIZE >= train_data_size:
            i_index = 0
            indices = np.random.permutation(train_data_size)
        else:
            i_index += config.BATCH_SIZE

        if i_index + 2 * config.BATCH_SIZE >= train_data_size:
            i_index = train_data_size - config.BATCH_SIZE

        print("train total reward {a}".format(a=sum_reward * 255))

def paint_amap(acmap):
    image = np.asanyarray(acmap.squeeze(), dtype=np.uint8)
    plt.imshow(image, vmin=1, vmax=9)
    plt.colorbar()
    plt.pause(1)
    # plt.show()
    plt.close('all')

if __name__ == '__main__':
    main()
