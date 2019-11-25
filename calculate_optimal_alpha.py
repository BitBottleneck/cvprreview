# -*-coding:utf-8-*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.linear_model import Lasso
import struct
import os
import math
import heapq
import sys


# The layer names of ResNet-50
names = ['conv0.txt', 'conv1_0_2.txt', 'conv1_1_1.txt', 'conv1_1_2.txt', 'conv1_2_1.txt', 'conv1_2_2.txt',
         'conv1_3_1.txt', 'conv1_3_2.txt', 'conv1_4_1.txt', 'conv1_4_2.txt', 'conv1_5_1.txt', 'conv1_5_2.txt',
         'conv1_6_1.txt', 'conv1_6_2.txt', 'conv1_7_1.txt', 'conv1_7_2.txt', 'conv2_0_1.txt', 'conv2_0_2.txt',
         'conv2_1_1.txt', 'conv2_1_2.txt', 'conv2_2_1.txt', 'conv2_2_2.txt', 'conv2_3_1.txt', 'conv2_3_2.txt',
         'conv2_4_1.txt', 'conv2_4_2.txt', 'conv2_5_1.txt', 'conv2_5_2.txt', 'conv2_6_1.txt', 'conv2_6_2.txt',
         'conv2_7_1.txt', 'conv2_7_2.txt', 'conv3_0_1.txt', 'conv3_0_2.txt', 'conv3_1_1.txt', 'conv3_1_2.txt',
         'conv3_2_1.txt', 'conv3_2_2.txt', 'conv3_3_1.txt', 'conv3_3_2.txt', 'conv3_4_1.txt', 'conv3_4_2.txt',
         'conv3_5_1.txt', 'conv3_5_2.txt', 'conv3_6_1.txt', 'conv3_6_2.txt', 'conv3_7_1.txt', 'conv3_7_2.txt',
         ]

# Hyperparameter Setting
layer_num = 48     # the number of layers
max_range = 20     # the maximal search range of \lambda
qt = 8             # the number of bits of baseline quantization
maximum = 2.**qt - 1
threshold_of_psnr_loss = 20      # the threshold of PSNR loss
threshold_of_psnr = 59. - threshold_of_psnr_loss   # 59 is the PSNR when \lambda is very small, such as 1e-10.

alpha_array = np.zeros([layer_num, 8], dtype=np.float)
for t in range(0, layer_num):
    content = np.loadtxt('/home/zxc/Liu/Bit-Bottleneck-ResNet/feature_map/8_bit/'+'%s'%names[t])
    data = np.array(content, dtype=np.float).reshape(-1, 1)
    data_abs = np.abs(data)
    data_abs[data_abs > maximum] = maximum
    data_round = np.round(data_abs)
    data_int = np.array(data_round, dtype=np.uint8)
    # Covert the data to bits array
    data_bit = np.unpackbits(data_int, axis=1)

    # Compress feature of neural network according to the threshold of PSNR
    for i in np.arange(0.001, max_range, 0.1):
        model = Lasso(alpha=i, positive=True, max_iter=1000, random_state=0)  # 调节alpha可以实现对拟合的程度
        model.fit(data_bit, data_abs)
        coef = np.array(model.coef_)
        # Calculate the PSNR
        PSNR = 0
        alpha = np.array(model.coef_).reshape((8, 1))
        data_recov = np.dot(data_bit, alpha)
        MSE = np.mean((data_recov / 1.0 - data_abs / 1.0) ** 2)
        PSNR = 10 * math.log10(maximum ** 2 / MSE)

        if PSNR <= threshold_of_psnr or i >= max_range-0.1:
            # if the minimal \lambda Below the threshold of PSNR, cycle will break
            if i == 0.001 or i >= max_range-0.1:
                break
            for i_p1 in np.arange(i - 0.1, i+0.1, 0.01):
                model = Lasso(alpha=i_p1, positive=True, max_iter=1000, random_state=0)
                model.fit(data_bit, data_abs)
                coef = np.array(model.coef_)
                # Calculate the PSNR
                PSNR = 0
                alpha = np.array(model.coef_).reshape((8, 1))
                data_recov = np.dot(data_bit, alpha)
                MSE = np.mean((data_recov / 1.0 - data_abs / 1.0) ** 2)
                PSNR = 10 * math.log10(maximum ** 2 / MSE)

                if PSNR <= threshold_of_psnr:
                    # Third level search
                    for i_p2 in np.arange(i_p1 - 0.01, i_p1+0.01, 0.001):
                        model = Lasso(alpha=i_p2, positive=True, max_iter=1000, random_state=0)
                        model.fit(data_bit, data_abs)
                        coef = np.array(model.coef_)
                        # Calculate the PSNR
                        PSNR = 0
                        alpha = np.array(model.coef_).reshape((8, 1))
                        data_recov = np.dot(data_bit, alpha)
                        MSE = np.mean((data_recov / 1.0 - data_abs / 1.0) ** 2)
                        PSNR = 10 * math.log10(maximum ** 2 / MSE)
                        if PSNR <= threshold_of_psnr:
                            break
                if PSNR <= threshold_of_psnr:
                    break
        if PSNR <= threshold_of_psnr or i >= max_range - 0.1:
            break
    # --------------------------------------------------------------------------------------------
    if i == 0.001:                             # When the minimal search range is large
        i_b = 0.001
        print('The minimal alpha is large.')
    elif i >= max_range-0.1:                   # When the maximal search range is small
        i_b = max_range
        print('The max range is small.')
    else:
        i_b = i_p2
        print('i_b = ', i_b)

    # Calculate the number of bit and \alpha correspond to the threshold of PSNR.
    lambda_final = i_b-0.001 if i_b-0.001 > 1e-5 else 1e-3
    model = Lasso(alpha=lambda_final, positive=True, max_iter=1000, random_state=0)
    model.fit(data_bit, data_abs)
    coef = np.array(model.coef_)
    print('---------------------------------------')
    print(names[t])
    print('---------------------------------------')
    print('Before tuning:')
    print('The lambda :', lambda_final)
    print('The alpha  :\n', np.reshape(coef, (8, 1)))
    # Calculate the PSNR
    PSNR = 0
    alpha = np.array(model.coef_).reshape((8, 1))
    data_recov = np.dot(data_bit, alpha)
    MSE = np.mean((data_recov / 1.0 - data_abs / 1.0) ** 2)
    PSNR = 10 * math.log10(maximum ** 2 / MSE)
    print('The PSNR  :', PSNR)
    # Count the number of bits after compression.
    bit_num_zero = 0
    for m in range(0, len(coef)):
        if coef[m] != 0.:
            bit_num_zero = bit_num_zero + 1
    print('The number of bits :', bit_num_zero)

    # ---------------------------------------------------------------------------------
    # Adjust the value of \lambda to achieve a higher PNSR under the same bits.
    # First level search of adjustment.
    i_b1_threshold = i_b - 10 if i_b - 10 >= 0 else 0
    for i_b1 in np.arange(i_b - 0.001, i_b1_threshold, -0.1):
        # if the minimal \lambda already below the threshold of PSNR or
        # the maximal \lambda can't below the threshold of PSNR.
        if i_b == 0.001 or i_b == max_range:
            break
        # if the feature couldn't be compressed under the threshold of PSNR
        if bit_num_zero == qt:
            break
        model = Lasso(alpha=i_b1, positive=True, max_iter=1000, random_state=0)
        model.fit(data_bit, data_abs)
        coef = np.array(model.coef_)
        bit_num = 0
        for m_1 in range(0, len(coef)):
            if coef[m_1] != 0.:
                bit_num = bit_num + 1

        if bit_num > bit_num_zero or i_b1 <= 0.1:  # i_b1 <= 0.1 prevent the ib_1 from becoming zero.
            # Second level search of adjustment.
            i_2_threshold = i_b1 - 0.1 if i_b1 - 0.1 >= 0 else 0
            for i_2 in np.arange(i_b1 + 0.1, i_2_threshold, -0.01):
                model = Lasso(alpha=i_2, positive=True, max_iter=1000, random_state=0)
                model.fit(data_bit, data_abs)
                coef = np.array(model.coef_)
                bit_num = 0
                for m_2 in range(0, len(coef)):
                    if coef[m_2] != 0.:
                        bit_num = bit_num + 1

                if bit_num > bit_num_zero or i_2 <= 0.01:
                    # Third level search of adjustment.
                    i_3_threshold = i_2 - 0.01 if i_2 - 0.01 >= 0 else 0
                    for i_3 in np.arange(i_2 + 0.01, i_3_threshold, -0.001):
                        model = Lasso(alpha=i_3, positive=True, max_iter=1000, random_state=0)
                        model.fit(data_bit, data_abs)
                        coef = np.array(model.coef_)
                        bit_num = 0
                        for m_3 in range(0, len(coef)):
                            if coef[m_3] != 0.:
                                bit_num = bit_num + 1
                        if bit_num > bit_num_zero or i_3 <= 0.001 + 1e-5:  # Adding 0.00001 prevents the noise
                            break
                if bit_num > bit_num_zero or i_2 <= 0.01:
                    break
        if bit_num > bit_num_zero or i_b1 <= 0.1:
            break
    # The process of search is over.
    # Get the best alpha
    print('---------------------------------------')
    print('After tuning:')
    # when reaching the smallest PSNR at the beginning or the bit number can't be compreseed
    if i == 0.001 or bit_num_zero == qt:
        best_alpha = 1e-5
        print(names[t] + ' alpha = ', best_alpha)
        print('The minimal alpha is large.')
    # When the minimal search range isn't small enough.
    elif i_3 <= 0.001 + 1e-5:
        best_alpha = 0.001
        print('alpha = ', best_alpha)
    # When the maximal search range isn't large enough.
    elif i_b == max_range:
        best_alpha = max_range
    else:
        best_alpha = i_3 + 0.001
        print('alpha = ', best_alpha)

    # Obtain the final vector \alpha
    model = Lasso(alpha=best_alpha, positive=True, max_iter=1000, random_state=0)  # 调节alpha可以实现对拟合的程度
    model.fit(data_bit, data_abs)
    alpha = np.array(model.coef_).reshape((8, 1))
    print('The final vector alpha: \n', alpha)
    alpha_array[t] = alpha.reshape((1, 8))
    # Count the final PSNR
    data_recov = np.dot(data_bit, alpha)
    MSE = np.mean((data_recov / 1.0 - data_abs / 1.0) ** 2)
    PSNR = 10 * math.log10(maximum ** 2 / MSE)
    print('The PSNR :', PSNR)
    print('\n')
    print('\n')
# Save the alpha of each layer as a .npz file.
np.savez("./alpha_file/alpha_array.npz", alpha_array)
print("it's finished!")

