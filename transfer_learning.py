# -*-coding:utf-8-*-

import tensorflow as tf
import numpy as np
from tensorflow.python import pywrap_tensorflow
import os

# The save address of the weight of baseline quantization.
FILE_PATH_old = "/home/zxc/Liu/Bit-Bottleneck-ResNet/logs_Bit_Bottleneck/old/model.ckpt"
# The new address used to save the transfer weight for new model.
FILE_PATH_new = "/home/zxc/Liu/Bit-Bottleneck-ResNet/logs_Bit_Bottleneck/model.ckpt"
# The save address of the weight of new model which is inserted Bit Bottleneck layers.
OUTPUT_FILE = "/home/zxc/Bit-Bottleneck-ResNet/logs_Bit_Bottleneck/new/"

old_data = []
old_name = []

new_data = []
new_name = []

# Read the baseline quantization weights.
for var_name_old, _ in tf.contrib.framework.list_variables(FILE_PATH_old):
    var_old = tf.contrib.framework.load_variable(FILE_PATH_old, var_name_old)
    old_data.append(var_old)
    old_name.append(var_name_old)

# Read the weights of new model.
for var_name_new, _ in tf.contrib.framework.list_variables(FILE_PATH_new):
    var_new = tf.contrib.framework.load_variable(FILE_PATH_new, var_name_new)
    new_data.append(var_new)
    new_name.append(var_name_new)


transform_variable_list = []
# If the name of variable is same , then use the old value to replace the new value.
for i in range(0, len(new_name)):
    for j in range(0, len(old_name)):
        if new_name[i] == old_name[j]:
            new_data[i] = old_data[j]
            print(new_name[i])
    rename = new_name[i]
    redata = new_data[i]
    # the variable of Variable_1 and Variable are int32 type， Others are float32 type
    if rename.find('Variable_1') != -1 or rename.find('Variable') != -1:
        renamed_var = tf.Variable(redata, name=rename, dtype=tf.int32)
    else:
        renamed_var = tf.Variable(redata, name=rename, dtype=tf.float32)
    transform_variable_list.append(renamed_var)


def save(saver, sess, logdir):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, write_meta_graph=False)
    print('The weights have been converted to {}.'.format(checkpoint_path))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(var_list=transform_variable_list, write_version=1)
    save(saver, sess, OUTPUT_FILE)
print("It's finished!")






