# -*- coding=utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import slim


def conv_lstm(input, batch_size_ph, cell_output, final_output, cell_kernel, weight_decay):
    '''

    :param input: 输入的tensor, 格式是[batch_size, step_size, h, w, channel]
    :param batch_size_ph: 输入的placeholder, 代表的是batch size用于初始化zero_state
    :param cell_output: ConvLSTMCell的输出
    :param cell_kernel: cell 里面卷积的kernel size
    :param final_output 3个cell的输出结合后再使用卷积的输出
    :return:
    '''
    batch_size, step_size, height, width, channel = input.get_shape().as_list()
    p_input_list = tf.split(input, step_size,
                            1)  # creates a list of leghth time_steps and one elemnt has the shape of (?, 400, 400, 1, 10)
    p_input_list = [tf.squeeze(p_input_, 1) for p_input_ in
                    p_input_list]  # remove the third dimention now one list elemnt has the shape of (?, 400, 400, 10)
    cell = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2,  # ConvLSTMCell definition
                                       input_shape=[height, width, channel],
                                       output_channels=cell_output,
                                       kernel_shape=cell_kernel,
                                       skip_connection=False)
    state = cell.zero_state(batch_size_ph, dtype=tf.float32)
    outputs = []
    with tf.variable_scope("ConvLSTM") as scope:
        for i, p_input_ in enumerate(p_input_list):
            print('i is ', i)
            if i > 0:
                scope.reuse_variables()
                print('set reuse')
            # ConvCell takes Tensor with size [batch_size, height, width, channel].
            t_output, state = cell(p_input_, state)
            outputs.append(t_output)
    outputs = tf.concat(outputs, axis=-1)
    print(outputs)
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        padding='SAME',
                        biases_initializer=tf.zeros_initializer()):
        _, _, _, output_channel = outputs.get_shape().as_list()
        outputs = slim.conv2d(outputs, output_channel, [3, 3], stride=1)
        outputs = slim.conv2d(outputs, final_output, [1, 1], stride=1)
    print(outputs)
    return outputs



if __name__ == '__main__':
    input_placeholder = tf.placeholder(shape=[None, 3, 512, 512, 32], dtype=tf.float32)
    batch_size_placeholder = tf.placeholder(dtype=tf.int32, shape=[], name='the_number_of_batch_size')
    conv_lstm(input_placeholder, batch_size_placeholder, 16, 32, [1, 1])
