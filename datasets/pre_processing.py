



import scipy.signal as sig
import pywt
import numpy as np
from scipy.io import loadmat, savemat
import os


def ecg_preprocessing(data, wfun='db6', levels=9, type=2):
    # data is the row data
    # 第一步：去除基线漂移---此噪声一般小于0.5HZ， 去除一般噪声，经验值50HZ（注意这两个值可以调整）
    filter_sig = bandpass_filter(
        data, low_cutoff=0.5, high_cutoff=50, sampling_frequency=500, filter_order=1)
    # 第二步：小波降噪
    levels = min(levels, pywt.dwt_max_level(data.shape[0], pywt.Wavelet(wfun)))
    if type == 1:
        # 论文：ECG beat classification using PCA, LDA, ICA and discrete wavelet transform
        # 用db6小波对信号进行9级小波分解，去除D1，D2，A9分量，使用剩下的分量进行重构，得到滤波后的信号。
        # 这种降噪方式理论上就是带通滤波器，所以测试精度可以使用type=1或者type=0加上带通进行测试
        coeffs = pywt.wavedec(filter_sig, wfun, level=levels)
        coeffs[-1] = np.zeros(len(coeffs[-1]))
        #coeffs[-2] = np.zeros(len(coeffs[-2]))
        coeffs[0] = np.zeros(len(coeffs[0]))
        processed_sig = pywt.waverec(coeffs, wfun)
    elif type == 2:
        processed_sig = filter_sig
    else:
        # 论文：Application of deep convolutional neural network for automated detection of myocardial infarction using ECG signals
        # 估计噪声，利用Donoho的估计公式
        coef = pywt.wavedec(filter_sig, 'sym8', level=2)
        Sigma = np.median(np.abs(coef[-1] - np.median(coef[-1]))) / 0.6745
        # 小波降噪
        coeffs = pywt.wavedec(filter_sig, wfun, level=levels)
        thresh = Sigma * np.sqrt(2 * np.log(filter_sig.shape[0]))
        for i in range(len(coeffs)):
            coeffs[i] = pywt.threshold(
                coeffs[i], thresh, 'soft')  # 还可以用'hard'，
        processed_sig = pywt.waverec(coeffs, wfun)

    return processed_sig


def bandpass_filter(data, low_cutoff, high_cutoff, sampling_frequency, filter_order):

    nyquist_frequency = sampling_frequency / 2
    low = low_cutoff / nyquist_frequency
    high = high_cutoff / nyquist_frequency
    b, a = sig.butter(filter_order, [low, high], btype="band")
    filter_sig = sig.lfilter(b, a, data)
    return filter_sig


data_path = '../TRAIN/'
save_path = './Output/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

dirs = os.listdir(data_path)
for filename in dirs:
    print('Processing: ' + filename)
    m = loadmat(os.path.join(data_path, filename))
    data = m['data']
    for i in range(data.shape[0]):
        data[i, :] = ecg_preprocessing(
            data[i, :], wfun='db6', levels=9, type=1)
    # (name, extension) = os.path.splitext(filename)
    savemat(os.path.join(save_path, filename), {'data': data})
