import numpy as np
import skimage.measure
import scipy
import cv2
import math
import matplotlib.pyplot as plt
from torch.nn import init
import torch
import pandas as pd



def init_weights(net, init_type='xavier', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='xavier', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net

def savevaltocsv(records, filename, sigma):
    col_name = ['n_epi', 'val_psnr', 'val_ssim']
    # records = [['001', '小明', 18]]
    # 先转为DataFrame格式
    df = pd.DataFrame(columns=col_name, data=records)
    # index=False表示存储csv时没有默认的id索引
    # 如果文件不存在，则创建文件，如果文件存在，则追加内容
    df.to_csv("./Val_PSNR_and_SSIM_{}/{}".format(sigma, filename), encoding='utf-8', index=False)
def patin_val(val_PSNR):
    plt.plot(val_PSNR)
    plt.grid()
    # plt.show()
    plt.pause(2)
    plt.close('all')
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for m1, m2 in zip(target.modules(), source.modules()):
        m1._buffers = m2._buffers.copy()
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

def gauss_filter(img):
    return cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=3.5)
    # return cv2.cv2.medianBlur(img, ksize=5)

def vis_bgrimg(img, time=1):
    image = np.asanyarray(img[0, 0:3, :, :].transpose(1, 2, 0) * 255, dtype=np.uint8)
    image = np.squeeze(image)
    cv2.imshow("rerrs", image)
    cv2.waitKey(time)

def vis_labimg(img, time=1):
    srs = img[:, 0:3, :, :].copy()
    # srs[:, 1, :, :] = (srs[:, 1, :, :] - 0.5) * 2
    # srs[:, 2, :, :] = (srs[:, 2, :, :] - 0.5) * 2

    srs[:, 0, :, :] *= 100
    srs[:, 1, :, :] *= 127
    srs[:, 2, :, :] *= 127
    image = np.asanyarray(lab2bgr(srs)[0, 0:3, :, :].transpose(1, 2, 0) * 255, dtype=np.uint8)
    image = np.squeeze(image)
    # plt.imshow(image)
    #
    # plt.ion()
    # plt.pause(1)
    # plt.close()
    cv2.imshow("rerr", image)
    cv2.waitKey(time)

def bgr2lab(src):
    b, c, h, w = src.shape
    src_t = np.transpose(src, (0, 2, 3, 1))
    dst = np.zeros(src_t.shape, src_t.dtype)
    for i in range(0, b):
        dst[i] = cv2.cvtColor(src_t[i], cv2.COLOR_BGR2Lab)
    lab_c = np.transpose(dst, (0, 3, 1, 2))
    lab_c[:, 0, :, :] /= 100.
    lab_c[:, 1, :, :] /= 127.
    lab_c[:, 2, :, :] /= 127.
    return lab_c

def bgr2gray(src):
    b, c, h, w = src.shape
    src_t = np.transpose(src, (0, 2, 3, 1))
    dst = np.zeros((b, h, w, 1))
    for i in range(0, b):
        dst[i] = np.expand_dims(cv2.cvtColor(src_t[i], cv2.COLOR_BGR2GRAY), 2)
    lab_c = np.transpose(dst, (0, 3, 1, 2))

    return lab_c

def bgr2labs(src):
    b, c, h, w = src.shape
    src_t = np.transpose(src, (0, 2, 3, 1))
    dst = np.zeros(src_t.shape, src_t.dtype)
    for i in range(0, b):
        dst[i] = cv2.cvtColor(src_t[i], cv2.COLOR_BGR2Lab)
    lab_c = np.transpose(dst, (0, 3, 1, 2))
    #lab_c[:, 0, :, :] /= 100.
    #lab_c[:, 1, :, :] /= 127.
    #lab_c[:, 2, :, :] /= 127.
    return lab_c
def bgr2hsv(src):
    b, c, h, w = src.shape
    src_t = np.transpose(src, (0, 2, 3, 1))
    dst = np.zeros(src_t.shape, src_t.dtype)
    for i in range(0, b):
        dst[i] = cv2.cvtColor(src_t[i], cv2.COLOR_BGR2HSV)
    return np.transpose(dst, (0, 3, 1, 2))


def lab2bgr(src):
    b, c, h, w = src.shape
    src_t = np.transpose(src, (0,2,3,1))
    dst = np.zeros(src_t.shape, src_t.dtype)
    for i in range(0,b):
        dst[i] = cv2.cvtColor(src_t[i], cv2.COLOR_Lab2BGR)
    return np.transpose(dst, (0,3,1,2))
def fft_shift(x):
    fft = scipy.fftpack.fft2(x)
    fft = scipy.fftpack.fftshift(fft)
    return fft


def shift_ifft(fft):
    fft = scipy.fftpack.ifftshift(fft)
    x = scipy.fftpack.ifft2(fft)
    return x


def Downsample(x, mask):
    fft = scipy.fftpack.fft2(x)
    fft_good = scipy.fftpack.fftshift(fft)
    fft_bad = fft_good * mask
    fft = scipy.fftpack.ifftshift(fft_bad)
    x = scipy.fftpack.ifft2(fft)
#    x = np.abs(x)
    x = np.real(x)
    return x, fft_good, fft_bad


def SSIM(x_good, x_bad):
    assert len(x_good.shape) == 2
    ssim_res = skimage.measure.compare_ssim(x_good, x_bad)
    return ssim_res


def PSNR(x_good, x_bad):
    assert len(x_good.shape) == 2
    psnr_res = skimage.measure.compare_psnr(x_good, x_bad)
    return psnr_res


def NMSE(x_good, x_bad):
    assert len(x_good.shape) == 2
    nmse_a_0_1 = np.sum((x_good - x_bad) ** 2)
    nmse_b_0_1 = np.sum(x_good ** 2)
    # this is DAGAN implementation, which is wrong
    nmse_a_0_1, nmse_b_0_1 = np.sqrt(nmse_a_0_1), np.sqrt(nmse_b_0_1)
    nmse_0_1 = nmse_a_0_1 / nmse_b_0_1
    return nmse_0_1


def computePSNR(o_, p_, i_):
    return PSNR(o_, p_), PSNR(o_, i_)


def computeSSIM(o_, p_, i_):
    return SSIM(o_, p_), SSIM(o_, i_)


def computeNMSE(o_, p_, i_):
    return NMSE(o_, p_), NMSE(o_, i_)


def DC(x_good, x_rec, mask):
    fft_good = fft_shift(x_good)
    fft_rec = fft_shift(x_rec)
    fft = fft_good * mask + fft_rec * (1 - mask)
    x = shift_ifft(fft)
    x = np.real(x)
    #x = np.abs(x)
    return x


def adjust_learning_rate(optimizer, iters, base_lr, policy_parameter, policy='step', multiple=[1]):
    '''
    source: https://github.com/last-one/Pytorch_Realtime_Multi-Person_Pose_Estimation/blob/master/utils.py
    '''
    if policy == 'fixed':
        lr = base_lr
    elif policy == 'step':
        lr = base_lr * (policy_parameter['gamma'] ** (iters // policy_parameter['step_size']))
    elif policy == 'exp':
        lr = base_lr * (policy_parameter['gamma'] ** iters)
    elif policy == 'inv':
        lr = base_lr * ((1 + policy_parameter['gamma'] * iters) ** (-policy_parameter['power']))
    elif policy == 'multistep':
        lr = base_lr
        for stepvalue in policy_parameter['stepvalue']:
            if iters >= stepvalue:
                lr *= policy_parameter['gamma']
            else:
                break
    elif policy == 'poly':
        lr = base_lr * ((1 - iters * 1.0 / policy_parameter['max_iter']) ** policy_parameter['power'])
    elif policy == 'sigmoid':
        lr = base_lr * (1.0 / (1 + math.exp(-policy_parameter['gamma'] * (iters - policy_parameter['stepsize']))))
    elif policy == 'multistep-poly':
        lr = base_lr
        stepstart = 0
        stepend = policy_parameter['max_iter']
        for stepvalue in policy_parameter['stepvalue']:
            if iters >= stepvalue:
                lr *= policy_parameter['gamma']
                stepstart = stepvalue
            else:
                stepend = stepvalue
                break
        lr = max(lr * policy_parameter['gamma'], lr * (1 - (iters - stepstart) * 1.0 / (stepend - stepstart)) ** policy_parameter['power'])

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr * multiple[i]
    return lr

if __name__ == "__main__":
    pass
