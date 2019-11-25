'''
This is the resnet structure
'''
import numpy as np
from hyper_parameters import *
import tensorflow as tf

BN_EPSILON = 0.001
MOVING_AVERAGES_FACTOR = 0.9
is_main_training_tower = False


# Import the vector alpha which was saved as a .npy file
alpha_array = np.load("./alpha_file/psnr_loss_17db.npy")
# The range of baseline quantization
maximum = 2**8 - 1
bit_num = 8
rounding = False
print_feature = False


def _save_mat(name, tensor_x):
    print(tensor_x.shape)
    f = open(name.decode('utf-8')+'.txt', 'w')
    for i in range(64):
        for j in range(tensor_x.shape[3]):
            v_2d = tensor_x[i, :, :, j]
            w = v_2d.shape[0]
            h = v_2d.shape[1]
            for Ii in range(w):
                for Ji in range(h):
                    strNum = str(v_2d[Ii, Ji])
                    f.write(strNum)
                    pass
                    pass
                    f.write('\n')
    f.close()
    return tensor_x


def _convert_x_1(name, tensor_x, i):
    _save_mat(name, tensor_x)
    v_2d = tensor_x[0, :, :,i]
    return v_2d


def convert_x_1(name, tensor_x):
    '''
    This is a function used to print feature into a .txt file.
    :param name: the name of feature
    :param tensor_x: the feature
    :return: .txt file
    '''
    for i in range(1):
        tensor_1 = tf.py_func(_convert_x_1, [name, tensor_x, i], tf.float32)  # 调用
        conv1out = tf.reshape(tensor_1, shape=[1, tf.shape(tensor_1)[0], tf.shape(tensor_1)[1], 1])  # 2d->4d
        tf.summary.image(name, conv1out, max_outputs=64)
    return


def DoReFa_quantization(x, name):
    '''
    This is DoReFa quantization method.
    :param x: the feature among layers
    :param name: the name of feature
    :return: the quantized feature
    '''
    rank = x.get_shape().ndims
    assert rank is not None

    maxx = tf.reduce_max(tf.abs(x), list(range(1, rank)), keep_dims=True)
    x = x / maxx
    x_normal = x*0.5 + tf.random_uniform(tf.shape(x), minval=-0.5 /maximum, maxval=0.5 / maximum)
    back_round = x_normal
    infer_round = tf.round(x_normal * maximum) / maximum
    y_round = back_round + tf.stop_gradient(infer_round - back_round)

    y_round = y_round + 0.5
    y_round = tf.clip_by_value(y_round, 0.0, 1.0)
    y_round = y_round - 0.5

    output = y_round * maxx * 2

    return output


def bit_bottleneck_layer(x, name, rounding=False, print_feature=False):
    '''
    This is the Bit Bottleneck layer
    :param x: the feature
    :param name:  the name of feature
    :param rounding:  if ture, Bit Bottleneck only quantize the feature without compression, or it will compress feature
    :param print_feature: if ture, it will print the feature in the inference process.
    :return: compressed feature
    '''
    if rounding:
        rank = x.get_shape().ndims
        assert rank is not None

        maxx = tf.reduce_max(tf.abs(x), list(range(1, rank)), keep_dims=True)
        x = x / maxx
        x_normal = x * 0.5 + tf.random_uniform(tf.shape(x), minval=-0.5 / maximum, maxval=0.5 / maximum)

        if print_feature:
            x_print = x_normal * maximum
            convert_x_1(name, x_print)

        back_round = x_normal
        infer_round = tf.round(x_normal * maximum) / maximum
        y_round = back_round + tf.stop_gradient(infer_round - back_round)

        y_round = y_round + 0.5
        y_round = tf.clip_by_value(y_round, 0.0, 1.0)
        y_round = y_round - 0.5

        output = y_round * maxx * 2

    else:
        origin_beta = np.ones(shape=(bit_num, 1), dtype=np.float32)
        # Import different vector \alpha according to the names fo layers
        if name == 'conv0':
            alpha = alpha_array[0].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv1_0_2':
            alpha = alpha_array[1].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv1_1_1':
            alpha = alpha_array[2].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv1_1_2':
            alpha = alpha_array[3].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv1_2_1':
            alpha = alpha_array[4].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv1_2_2':
            alpha = alpha_array[5].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv1_3_1':
            alpha = alpha_array[6].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv1_3_2':
            alpha = alpha_array[7].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv1_4_1':
            alpha = alpha_array[8].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv1_4_2':
            alpha = alpha_array[9].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv1_5_1':
            alpha = alpha_array[10].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv1_5_2':
            alpha = alpha_array[11].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv1_6_1':
            alpha = alpha_array[12].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv1_6_2':
            alpha = alpha_array[13].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv1_7_1':
            alpha = alpha_array[14].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv1_7_2':
            alpha = alpha_array[15].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv2_0_1':
            alpha = alpha_array[16].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv2_0_2':
            alpha = alpha_array[17].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv2_1_1':
            alpha = alpha_array[18].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv2_1_2':
            alpha = alpha_array[19].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv2_2_1':
            alpha = alpha_array[20].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv2_2_2':
            alpha = alpha_array[21].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv2_3_1':
            alpha = alpha_array[22].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv2_3_2':
            alpha = alpha_array[23].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv2_4_1':
            alpha = alpha_array[24].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv2_4_2':
            alpha = alpha_array[25].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv2_5_1':
            alpha = alpha_array[26].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv2_5_2':
            alpha = alpha_array[27].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv2_6_1':
            alpha = alpha_array[28].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv2_6_2':
            alpha = alpha_array[29].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv2_7_1':
            alpha = alpha_array[30].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv2_7_2':
            alpha = alpha_array[31].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv3_0_1':
            alpha = alpha_array[32].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv3_0_2':
            alpha = alpha_array[33].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv3_1_1':
            alpha = alpha_array[34].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv3_1_2':
            alpha = alpha_array[35].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv3_2_1':
            alpha = alpha_array[36].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv3_2_2':
            alpha = alpha_array[37].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv3_3_1':
            alpha = alpha_array[38].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv3_3_2':
            alpha = alpha_array[39].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv3_4_1':
            alpha = alpha_array[40].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv3_4_2':
            alpha = alpha_array[41].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv3_5_1':
            alpha = alpha_array[42].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv3_5_2':
            alpha = alpha_array[43].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv3_6_1':
            alpha = alpha_array[44].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv3_6_2':
            alpha = alpha_array[45].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv3_7_1':
            alpha = alpha_array[46].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        elif name == 'conv3_7_2':
            alpha = alpha_array[47].reshape(bit_num, 1)
            init_beta = tf.multiply(origin_beta, alpha)
        else:
            print('There is something wrong !')

        init_beta = tf.reshape(init_beta, shape=[bit_num, 1])
        with tf.variable_scope('bit_bottle'):
            beta = tf.Variable(init_beta, name='bit_beta', trainable=True)

        beta_back = tf.reshape(tf.constant(np.ones(shape=(bit_num, 1), dtype=np.float32)), shape=(bit_num, 1))
        rank = x.get_shape().ndims
        assert rank is not None
        # DoReFa quantization
        maxx = tf.reduce_max(tf.abs(x), list(range(1, rank)), keep_dims=True)
        x = x / maxx
        x_normal = x * 0.5 + tf.random_uniform(tf.shape(x), minval=-0.5 / maximum, maxval=0.5 / maximum)
        round_back = x_normal * maximum
        round_infer = tf.round(x_normal * maximum)
        y = round_back + tf.stop_gradient(round_infer - round_back)

        y_sign = tf.sign(y)
        y_shape = y.shape
        y = tf.multiply(y, y_sign)
        y = tf.reshape(y, [-1])
        # Obtain the bits array of feature
        fdiv_back_0 = tf.div(y, 2.)
        fdiv_forward_0 = tf.floordiv(y, 2.)
        y_fdiv2 = fdiv_back_0 + tf.stop_gradient(fdiv_forward_0 - fdiv_back_0)
        xbit0 = y + tf.stop_gradient(tf.subtract(y, tf.multiply(y_fdiv2, 2.)) - y)

        fdiv_back_1 = tf.div(y_fdiv2, 2.)
        fdiv_forward_1 = tf.floordiv(y_fdiv2, 2.)
        y_fdiv4 = fdiv_back_1 + tf.stop_gradient(fdiv_forward_1 - fdiv_back_1)
        xbit1 = y + tf.stop_gradient(tf.subtract(y_fdiv2, tf.multiply(y_fdiv4, 2.)) - y)

        fdiv_back_2 = tf.div(y_fdiv4, 2.)
        fdiv_forward_2 = tf.floordiv(y_fdiv4, 2.)
        y_fdiv8 = fdiv_back_2 + tf.stop_gradient(fdiv_forward_2 - fdiv_back_2)
        xbit2 = y + tf.stop_gradient(tf.subtract(y_fdiv4, tf.multiply(y_fdiv8, 2.)) - y)

        fdiv_back_3 = tf.div(y_fdiv8, 2.)
        fdiv_forward_3 = tf.floordiv(y_fdiv8, 2.)
        y_fdiv16 = fdiv_back_3 + tf.stop_gradient(fdiv_forward_3 - fdiv_back_3)
        xbit3 = y + tf.stop_gradient(tf.subtract(y_fdiv8, tf.multiply(y_fdiv16, 2.)) - y)

        fdiv_back_4 = tf.div(y_fdiv16, 2.)
        fdiv_forward_4 = tf.floordiv(y_fdiv16, 2.)
        y_fdiv32 = fdiv_back_4 + tf.stop_gradient(fdiv_forward_4 - fdiv_back_4)
        xbit4 = y + tf.stop_gradient(tf.subtract(y_fdiv16, tf.multiply(y_fdiv32, 2.)) - y)

        fdiv_back_5 = tf.div(y_fdiv32, 2.)
        fdiv_forward_5 = tf.floordiv(y_fdiv32, 2.)
        y_fdiv64 = fdiv_back_5 + tf.stop_gradient(fdiv_forward_5 - fdiv_back_5)
        xbit5 = y + tf.stop_gradient(tf.subtract(y_fdiv32, tf.multiply(y_fdiv64, 2.)) - y)

        fdiv_back_6 = tf.div(y_fdiv64, 2.)
        fdiv_forward_6 = tf.floordiv(y_fdiv64, 2.)
        y_fdiv128 = fdiv_back_6 + tf.stop_gradient(fdiv_forward_6 - fdiv_back_6)
        xbit6 = y + tf.stop_gradient(tf.subtract(y_fdiv64, tf.multiply(y_fdiv128, 2.)) - y)

        fdiv_back_7 = tf.div(y_fdiv128, 2.)
        fdiv_forward_7 = tf.floordiv(y_fdiv128, 2.)
        y_fdiv256 = fdiv_back_7 + tf.stop_gradient(fdiv_forward_7 - fdiv_back_7)
        xbit7 = y + tf.stop_gradient(tf.subtract(y_fdiv128, tf.multiply(y_fdiv256, 2.)) - y)

        y_stack = tf.stack([xbit7, xbit6, xbit5, xbit4, xbit3, xbit2, xbit1, xbit0], axis=1)
        # The bits arrays multiply the vector \alpha
        y_recov = tf.matmul(y_stack, beta)

        y_recov_back = tf.div(tf.matmul(y_stack, beta_back), tf.cast(bit_num, tf.float32))
        y_output = y_recov_back + tf.stop_gradient(y_recov - y_recov_back)

        y_output = tf.reshape(y_output, shape=y_shape)
        y_output = tf.multiply(y_output, y_sign)

        output = y_output / maximum + 0.5
        output = tf.clip_by_value(output, 0.0, 1.0)
        output = output - 0.5
        output = 2 * maxx * output

    return output


def activation_summary(x):
    '''
    :param x: A Tensor
    :return: Add histogram summary and scalar summary of the sparsity of the tensor
    '''
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''
    
    ## TODO: to allow different weight decay to fully connected layer and conv layer
    regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)

    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                      regularizer=regularizer)
    return new_variables


def output_layer(input_layer, num_labels):
    '''
    :param input_layer: 2D tensor
    :param num_labels: int. How many output labels in total? (10 for cifar10 and 100 for cifar100)
    :return: output layer Y = WX + B
    '''
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())

    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h


def batch_normalization_layer(input_layer, dimension):
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable('beta', dimension, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return bn_layer


def conv_bn_relu_layer(input_layer, filter_shape, stride, name):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''

    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel)

    bit_bottle = bit_bottleneck_layer(bn_layer, name, rounding, print_feature)

    output = tf.nn.relu(bit_bottle)

    return output


def bn_relu_conv_layer(input_layer, filter_shape, stride, name):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    '''

    in_channel = input_layer.get_shape().as_list()[-1]
    bn_layer = batch_normalization_layer(input_layer, in_channel)
    bit_bottle = bit_bottleneck_layer(bn_layer, name, rounding, print_feature)
    filter = create_variables(name='conv', shape=filter_shape)
    conv = tf.nn.conv2d(bit_bottle, filter, strides=[1, stride, stride, 1], padding='SAME')

    output = tf.nn.relu(conv)

    return output


def residual_block(input_layer, output_channel, first_block=False, name='conv'):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_in_block'):
        if first_block:
            filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
            conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride, name+'_%d'%1)

    with tf.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1, name+'_%d'%2)

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                     input_channel // 2]])
    else:
        padded_input = input_layer

    output = conv2 + padded_input
    return output


def inference(input_tensor_batch, n, reuse):
    '''
    The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
    :param input_tensor_batch: 4D tensor
    :param n: num_residual_blocks
    :param reuse: To build train graph, reuse=False. To build validation graph and share weights
    with train graph, resue=True
    :return: last layer in the network. Not softmax-ed
    '''
    layers = []
    with tf.variable_scope('conv0', reuse=reuse):
        conv0 = conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, 16], 1, 'conv0')
        activation_summary(conv0)
        layers.append(conv0)

    for i in range(n):
        with tf.variable_scope('conv1_%d' %i, reuse=reuse):
            if i == 0:
                conv1 = residual_block(layers[-1], 16, first_block=True, name='conv1_%d' %i)

            else:
                conv1 = residual_block(layers[-1], 16, name='conv1_%d' %i)
            activation_summary(conv1)
            layers.append(conv1)

    for i in range(n):
        with tf.variable_scope('conv2_%d' %i, reuse=reuse):
            conv2 = residual_block(layers[-1], 32, name='conv2_%d' %i)
            activation_summary(conv2)
            layers.append(conv2)

    for i in range(n):
        with tf.variable_scope('conv3_%d' %i, reuse=reuse):
            conv3 = residual_block(layers[-1], 64, name='conv3_%d' %i)
            layers.append(conv3)
        assert conv3.get_shape().as_list()[1:] == [8, 8, 64]

    with tf.variable_scope('fc', reuse=reuse):
        in_channel = layers[-1].get_shape().as_list()[-1]
        bn_layer = batch_normalization_layer(layers[-1], in_channel)
        relu_layer = tf.nn.relu(bn_layer)
        global_pool = tf.reduce_mean(relu_layer, [1, 2])

        assert global_pool.get_shape().as_list()[-1:] == [64]
        output = output_layer(global_pool, 10)
        layers.append(output)

    return layers[-1]


def test_graph(train_dir='logs'):
    '''
    Run this function to look at the graph structure on tensorboard. A fast way!
    :param train_dir:
    '''
    input_tensor = tf.constant(np.ones([128, 32, 32, 3]), dtype=tf.float32)
    result = inference(input_tensor, 2, reuse=False)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)
