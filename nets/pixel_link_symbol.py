# -*- coding=utf-8 -*-
import tensorflow as tf
from nets.conv_lstm import conv_lstm
FLAGS = tf.app.flags.FLAGS
slim = tf.contrib.slim


MODEL_TYPE_vgg16 = 'vgg16'
MODEL_TYPE_vgg16_no_dilation = 'vgg16_no_dilation'

FUSE_TYPE_cascade_conv1x1_upsample_sum = 'cascade_conv1x1_upsample_sum'
FUSE_TYPE_cascade_conv1x1_128_upsamle_sum_conv1x1_2 = \
                            'cascade_conv1x1_128_upsamle_sum_conv1x1_2'
FUSE_TYPE_cascade_conv1x1_128_upsamle_concat_conv1x1_2 = \
                            'cascade_conv1x1_128_upsamle_concat_conv1x1_2'


class PixelLinkNet_multiphase_multislice_clstm(object):
    def __init__(self, nc_inputs, art_inputs, pv_inputs, is_training, batch_size_ph):
        self.nc_inputs = nc_inputs
        self.batch_size_ph = batch_size_ph
        self.art_inputs = art_inputs
        self.pv_inputs = pv_inputs
        self.is_training = is_training
        self._build_network()
        self._fuse_feat_layers()
        self._logits_to_scores()

    def _build_network(self):
        import config

        self.nets = []
        self.end_points_s = []
        phase2input={
            0: self.nc_inputs,
            1: self.art_inputs,
            2: self.pv_inputs
        }
        for phase_idx in range(3):
            if config.model_type == MODEL_TYPE_vgg16:
                from nets import vgg
                with slim.arg_scope([slim.conv2d],
                                    activation_fn=tf.nn.relu,
                                    reuse=(phase_idx != 0),
                                    weights_regularizer=slim.l2_regularizer(config.weight_decay),
                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                    biases_initializer=tf.zeros_initializer()):
                    with slim.arg_scope([slim.max_pool2d], padding='SAME'):
                        with slim.arg_scope([slim.conv2d],
                                            reuse=(phase_idx != 0),
                                            padding='SAME') as sc:
                            if phase_idx == 0:
                                self.arg_scope = sc
                            net, end_points = vgg.basenet(
                                inputs=phase2input[phase_idx])

            elif config.model_type == MODEL_TYPE_vgg16_no_dilation:
                from nets import vgg
                with slim.arg_scope([slim.conv2d],
                                    activation_fn=tf.nn.relu,
                                    reuse=(phase_idx != 0),
                                    weights_regularizer=slim.l2_regularizer(config.weight_decay),
                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                    biases_initializer=tf.zeros_initializer()):
                    with slim.arg_scope([slim.max_pool2d], padding='SAME'):
                        with slim.arg_scope([slim.conv2d],
                                            reuse=(phase_idx != 0),
                                            padding='SAME') as sc:
                            if phase_idx == 0:
                                self.arg_scope = sc
                            net, end_points = vgg.basenet(
                                inputs=phase2input[phase_idx], dilation=False)
            else:
                raise ValueError('model_type not supported:%s' % (config.model_type))
            self.nets.append(net)
            self.end_points_s.append(end_points)
        with tf.variable_scope('fuse_multi_phase'):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                weights_regularizer=slim.l2_regularizer(config.weight_decay),
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                padding='SAME',
                                biases_initializer=tf.zeros_initializer()):
                # self.net = tf.concat(self.nets, axis=-1)
                # _, _, _, channel = self.net.get_shape().as_list()
                # output_channel = channel // 3
                # self.net = slim.conv2d(self.net, output_channel, [1, 1], stride=1, scope='merged_net')
                self.end_points = {}
                for name in config.feat_layers:
                    cur_input = []
                    final_output = None
                    for i in range(3):
                        if i == 0:
                            _, _, _, final_output = self.end_points_s[i][name].get_shape().as_list()
                        cur_input.append(tf.expand_dims(self.end_points_s[i][name], axis=1))
                    cur_input = tf.concat(cur_input, axis=1)
                    with tf.variable_scope('clstm_%s' % name):
                        self.end_points[name] = conv_lstm(cur_input, self.batch_size_ph, final_output // 2, final_output,
                                                          [1, 1], config.weight_decay)

    def _score_layer(self, input_layer, num_classes, scope):
        import config
        with slim.arg_scope(self.arg_scope):
            logits = slim.conv2d(input_layer, num_classes, [1, 1],
                                 stride=1,
                                 activation_fn=None,
                                 scope='score_from_%s' % scope,
                                 normalizer_fn=None)
            try:
                use_dropout = config.dropout_ratio > 0
            except:
                use_dropout = False

            if use_dropout:
                if self.is_training:
                    dropout_ratio = config.dropout_ratio
                else:
                    dropout_ratio = 0
                keep_prob = 1.0 - dropout_ratio
                tf.logging.info('Using Dropout, with keep_prob = %f' % (keep_prob))
                logits = tf.nn.dropout(logits, keep_prob)
            return logits

    def _upscore_layer(self, layer, target_layer):
        #             target_shape = target_layer.shape[1:-1] # NHWC
        target_shape = tf.shape(target_layer)[1:-1]
        upscored = tf.image.resize_images(layer, target_shape)
        return upscored

    def _fuse_by_cascade_conv1x1_128_upsamle_sum_conv1x1_2(self, scope):
        """
        The feature fuse fashion of
            'Deep Direct Regression for Multi-Oriented Scene Text Detection'

        Instead of fusion of scores, feature map from 1x1, 128 conv are fused,
        and the scores are predicted on it.
        """
        base_map = self._fuse_by_cascade_conv1x1_upsample_sum(num_classes=128,
                                                              scope='feature_fuse')
        return base_map

    def _fuse_by_cascade_conv1x1_128_upsamle_concat_conv1x1_2(self, scope, num_classes=32):
        import config
        num_layers = len(config.feat_layers)

        with tf.variable_scope(scope):
            smaller_score_map = None
            for idx in range(0, len(config.feat_layers))[::-1]:  # [4, 3, 2, 1, 0]
                current_layer_name = config.feat_layers[idx]
                current_layer = self.end_points[current_layer_name]
                current_score_map = self._score_layer(current_layer,
                                                      num_classes, current_layer_name)
                if smaller_score_map is None:
                    smaller_score_map = current_score_map
                else:
                    upscore_map = self._upscore_layer(smaller_score_map, current_score_map)
                    smaller_score_map = tf.concat([current_score_map, upscore_map], axis=0)

        return smaller_score_map

    def _fuse_by_cascade_conv1x1_upsample_sum(self, num_classes, scope):
        """
        The feature fuse fashion of FCN for semantic segmentation:
        Suppose there are several feature maps with decreasing sizes ,
        and we are going to get a single score map from them.

        Every feature map contributes to the final score map:
            predict score on all the feature maps using 1x1 conv, with
            depth equal to num_classes

        The score map is upsampled and added in a cascade way:
            start from the smallest score map, upsmale it to the size
            of the next score map with a larger size, and add them
            to get a fused score map. Upsample this fused score map and
            add it to the next sibling larger score map. The final
            score map is got when all score maps are fused together
        """
        import config
        num_layers = len(config.feat_layers)

        with tf.variable_scope(scope):
            smaller_score_map = None
            for idx in range(0, len(config.feat_layers))[::-1]:  # [4, 3, 2, 1, 0]
                current_layer_name = config.feat_layers[idx]
                current_layer = self.end_points[current_layer_name]
                current_score_map = self._score_layer(current_layer,
                                                      num_classes, current_layer_name)
                if smaller_score_map is None:
                    smaller_score_map = current_score_map
                else:
                    upscore_map = self._upscore_layer(smaller_score_map, current_score_map)
                    smaller_score_map = current_score_map + upscore_map

        return smaller_score_map

    def _fuse_feat_layers(self):
        import config
        if config.feat_fuse_type == FUSE_TYPE_cascade_conv1x1_upsample_sum:
            self.pixel_cls_logits = self._fuse_by_cascade_conv1x1_upsample_sum(
                config.num_classes, scope='pixel_cls')

            self.pixel_link_logits = self._fuse_by_cascade_conv1x1_upsample_sum(
                config.num_neighbours * 2, scope='pixel_link')

        elif config.feat_fuse_type == FUSE_TYPE_cascade_conv1x1_128_upsamle_sum_conv1x1_2:
            base_map = self._fuse_by_cascade_conv1x1_128_upsamle_sum_conv1x1_2(
                scope='fuse_feature')

            self.pixel_cls_logits = self._score_layer(base_map,
                                                      config.num_classes, scope='pixel_cls')

            self.pixel_link_logits = self._score_layer(base_map,
                                                       config.num_neighbours * 2, scope='pixel_link')
        elif config.feat_fuse_type == FUSE_TYPE_cascade_conv1x1_128_upsamle_concat_conv1x1_2:
            base_map = self._fuse_by_cascade_conv1x1_128_upsamle_concat_conv1x1_2(
                scope='fuse_feature')
        else:
            raise ValueError('feat_fuse_type not supported:%s' % (config.feat_fuse_type))

    def _flat_pixel_cls_values(self, values):
        shape = values.shape.as_list()
        values = tf.reshape(values, shape=[shape[0], -1, shape[-1]])
        return values

    def _logits_to_scores(self):
        self.pixel_cls_scores = tf.nn.softmax(self.pixel_cls_logits)
        # 将[N, W, H, C] -> [N, W*H, C]的格式
        self.pixel_cls_logits_flatten = \
            self._flat_pixel_cls_values(self.pixel_cls_logits)
        self.pixel_cls_scores_flatten = \
            self._flat_pixel_cls_values(self.pixel_cls_scores)

        import config
        #         shape = self.pixel_link_logits.shape.as_list()
        shape = tf.shape(self.pixel_link_logits)
        self.pixel_link_logits = tf.reshape(self.pixel_link_logits,
                                            [shape[0], shape[1], shape[2], config.num_neighbours, 2])

        self.pixel_link_scores = tf.nn.softmax(self.pixel_link_logits)

        self.pixel_pos_scores = self.pixel_cls_scores[:, :, :, 1]
        self.link_pos_scores = self.pixel_link_scores[:, :, :, :, 1]

    def build_loss(self, pixel_cls_labels, pixel_cls_weights,
                   pixel_link_labels, pixel_link_weights,
                   do_summary=True
                   ):
        """
        The loss consists of two parts: pixel_cls_loss + link_cls_loss,
            and link_cls_loss is calculated only on positive pixels
        """
        import config
        count_warning = tf.get_local_variable(
            name='count_warning', initializer=tf.constant(0.0))
        batch_size = config.batch_size_per_gpu
        ignore_label = config.ignore_label
        background_label = config.background_label
        text_label = config.text_label
        pixel_link_neg_loss_weight_lambda = config.pixel_link_neg_loss_weight_lambda
        pixel_cls_loss_weight_lambda = config.pixel_cls_loss_weight_lambda
        pixel_link_loss_weight = config.pixel_link_loss_weight

        def OHNM_single_image(scores, n_pos, neg_mask):
            """Online Hard Negative Mining.
                scores: the scores of being predicted as negative cls
                n_pos: the number of positive samples
                neg_mask: mask of negative samples
                Return:
                    the mask of selected negative samples.
                    if n_pos == 0, top 10000 negative samples will be selected.
            """

            def has_pos():
                return n_pos * config.max_neg_pos_ratio

            def no_pos():
                return tf.constant(10000, dtype=tf.int32)

            n_neg = tf.cond(n_pos > 0, has_pos, no_pos)
            max_neg_entries = tf.reduce_sum(tf.cast(neg_mask, tf.int32))

            n_neg = tf.minimum(n_neg, max_neg_entries)
            n_neg = tf.cast(n_neg, tf.int32)

            def has_neg():
                neg_conf = tf.boolean_mask(scores, neg_mask)
                vals, _ = tf.nn.top_k(-neg_conf, k=n_neg)
                threshold = vals[-1]  # a negtive value
                selected_neg_mask = tf.logical_and(neg_mask, scores <= -threshold)
                return selected_neg_mask

            def no_neg():
                selected_neg_mask = tf.zeros_like(neg_mask)
                return selected_neg_mask

            selected_neg_mask = tf.cond(n_neg > 0, has_neg, no_neg)
            return tf.cast(selected_neg_mask, tf.int32)

        def OHNM_batch(neg_conf, pos_mask, neg_mask):
            selected_neg_mask = []
            for image_idx in range(batch_size):
                image_neg_conf = neg_conf[image_idx, :]
                image_neg_mask = neg_mask[image_idx, :]
                image_pos_mask = pos_mask[image_idx, :]
                n_pos = tf.reduce_sum(tf.cast(image_pos_mask, tf.int32))
                selected_neg_mask.append(OHNM_single_image(image_neg_conf, n_pos, image_neg_mask))

            selected_neg_mask = tf.stack(selected_neg_mask)
            return selected_neg_mask

        # OHNM on pixel classification task
        pixel_cls_labels_flatten = tf.reshape(pixel_cls_labels, [batch_size, -1])
        pos_pixel_weights_flatten = tf.reshape(pixel_cls_weights, [batch_size, -1])

        pos_mask = tf.equal(pixel_cls_labels_flatten, text_label)
        neg_mask = tf.equal(pixel_cls_labels_flatten, background_label)

        n_pos = tf.reduce_sum(tf.cast(pos_mask, dtype=tf.float32))

        with tf.name_scope('pixel_cls_loss'):
            def no_pos():
                return tf.constant(.0);

            def has_pos():
                pixel_cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.pixel_cls_logits_flatten,
                    labels=tf.cast(pos_mask, dtype=tf.int32))

                pixel_neg_scores = self.pixel_cls_scores_flatten[:, :, 0]
                selected_neg_pixel_mask = OHNM_batch(pixel_neg_scores, pos_mask, neg_mask)

                pixel_cls_weights = pos_pixel_weights_flatten + \
                                    tf.cast(selected_neg_pixel_mask, tf.float32)
                n_neg = tf.cast(tf.reduce_sum(selected_neg_pixel_mask), tf.float32)
                loss = tf.reduce_sum(pixel_cls_loss * pixel_cls_weights) / (n_neg + n_pos)
                return loss

            #             pixel_cls_loss = tf.cond(n_pos > 0, has_pos, no_pos)
            pixel_cls_loss = has_pos()
            tf.add_to_collection(tf.GraphKeys.LOSSES, pixel_cls_loss * pixel_cls_loss_weight_lambda)

        with tf.name_scope('pixel_link_loss'):
            def no_pos():
                return tf.constant(.0), tf.constant(.0);

            def has_pos():
                pixel_link_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.pixel_link_logits,
                    labels=pixel_link_labels)

                def get_loss(label):
                    link_mask = tf.equal(pixel_link_labels, label)
                    link_weights = pixel_link_weights * tf.cast(link_mask, tf.float32)
                    n_links = tf.reduce_sum(link_weights)
                    loss = tf.reduce_sum(pixel_link_loss * link_weights) / n_links
                    return loss

                neg_loss = get_loss(0)
                pos_loss = get_loss(1)
                return neg_loss, pos_loss

            pixel_neg_link_loss, pixel_pos_link_loss = \
                tf.cond(n_pos > 0, has_pos, no_pos)

            pixel_link_loss = pixel_pos_link_loss + \
                              pixel_neg_link_loss * pixel_link_neg_loss_weight_lambda

            tf.add_to_collection(tf.GraphKeys.LOSSES,
                                 pixel_link_loss_weight * pixel_link_loss)

        if do_summary:
            tf.summary.scalar('pixel_cls_loss', pixel_cls_loss)
            tf.summary.scalar('pixel_pos_link_loss', pixel_pos_link_loss)
            tf.summary.scalar('pixel_neg_link_loss', pixel_neg_link_loss)

class PixelLinkNet_multiphase_multislice(object):
    def __init__(self, nc_inputs, art_inputs, pv_inputs, is_training):
        self.nc_inputs = nc_inputs
        self.art_inputs = art_inputs
        self.pv_inputs = pv_inputs
        self.is_training = is_training
        self._build_network()
        self._fuse_feat_layers()
        self._logits_to_scores()

    def _build_network(self):
        import config

        self.nets = []
        self.end_points_s = []
        phase2input={
            0: self.nc_inputs,
            1: self.art_inputs,
            2: self.pv_inputs
        }
        for phase_idx in range(3):
            if config.model_type == MODEL_TYPE_vgg16:
                from nets import vgg
                with slim.arg_scope([slim.conv2d],
                                    activation_fn=tf.nn.relu,
                                    reuse=(phase_idx != 0),
                                    weights_regularizer=slim.l2_regularizer(config.weight_decay),
                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                    biases_initializer=tf.zeros_initializer()):
                    with slim.arg_scope([slim.max_pool2d], padding='SAME'):
                        with slim.arg_scope([slim.conv2d],
                                            reuse=(phase_idx != 0),
                                            padding='SAME') as sc:
                            if phase_idx == 0:
                                self.arg_scope = sc
                            net, end_points = vgg.basenet(
                                inputs=phase2input[phase_idx])

            elif config.model_type == MODEL_TYPE_vgg16_no_dilation:
                from nets import vgg
                with slim.arg_scope([slim.conv2d],
                                    activation_fn=tf.nn.relu,
                                    reuse=(phase_idx != 0),
                                    weights_regularizer=slim.l2_regularizer(config.weight_decay),
                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                    biases_initializer=tf.zeros_initializer()):
                    with slim.arg_scope([slim.max_pool2d], padding='SAME'):
                        with slim.arg_scope([slim.conv2d],
                                            reuse=(phase_idx != 0),
                                            padding='SAME') as sc:
                            if phase_idx == 0:
                                self.arg_scope = sc
                            net, end_points = vgg.basenet(
                                inputs=phase2input[phase_idx], dilation=False)
            else:
                raise ValueError('model_type not supported:%s' % (config.model_type))
            self.nets.append(net)
            self.end_points_s.append(end_points)
        with tf.variable_scope('fuse_multi_phase'):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                weights_regularizer=slim.l2_regularizer(config.weight_decay),
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                padding='SAME',
                                biases_initializer=tf.zeros_initializer()):
                # self.net = tf.concat(self.nets, axis=-1)
                # _, _, _, channel = self.net.get_shape().as_list()
                # output_channel = channel // 3
                # self.net = slim.conv2d(self.net, output_channel, [1, 1], stride=1, scope='merged_net')
                self.end_points = {}
                for name in config.feat_layers:
                    cur_input = []
                    for i in range(3):
                        cur_input.append(self.end_points_s[i][name])
                    cur_input = tf.concat(cur_input, axis=-1)
                    _, _, _, channel = cur_input.get_shape().as_list()
                    output_channel = channel // 3
                    self.end_points[name] = slim.conv2d(
                        cur_input, output_channel, [1, 1], stride=1, scope='merged_' + name
                    )

    def _score_layer(self, input_layer, num_classes, scope):
        import config
        with slim.arg_scope(self.arg_scope):
            logits = slim.conv2d(input_layer, num_classes, [1, 1],
                                 stride=1,
                                 activation_fn=None,
                                 scope='score_from_%s' % scope,
                                 normalizer_fn=None)
            try:
                use_dropout = config.dropout_ratio > 0
            except:
                use_dropout = False

            if use_dropout:
                if self.is_training:
                    dropout_ratio = config.dropout_ratio
                else:
                    dropout_ratio = 0
                keep_prob = 1.0 - dropout_ratio
                tf.logging.info('Using Dropout, with keep_prob = %f' % (keep_prob))
                logits = tf.nn.dropout(logits, keep_prob)
            return logits

    def _upscore_layer(self, layer, target_layer):
        #             target_shape = target_layer.shape[1:-1] # NHWC
        target_shape = tf.shape(target_layer)[1:-1]
        upscored = tf.image.resize_images(layer, target_shape)
        return upscored

    def _fuse_by_cascade_conv1x1_128_upsamle_sum_conv1x1_2(self, scope):
        """
        The feature fuse fashion of
            'Deep Direct Regression for Multi-Oriented Scene Text Detection'

        Instead of fusion of scores, feature map from 1x1, 128 conv are fused,
        and the scores are predicted on it.
        """
        base_map = self._fuse_by_cascade_conv1x1_upsample_sum(num_classes=128,
                                                              scope='feature_fuse')
        return base_map

    def _fuse_by_cascade_conv1x1_128_upsamle_concat_conv1x1_2(self, scope, num_classes=32):
        import config
        num_layers = len(config.feat_layers)

        with tf.variable_scope(scope):
            smaller_score_map = None
            for idx in range(0, len(config.feat_layers))[::-1]:  # [4, 3, 2, 1, 0]
                current_layer_name = config.feat_layers[idx]
                current_layer = self.end_points[current_layer_name]
                current_score_map = self._score_layer(current_layer,
                                                      num_classes, current_layer_name)
                if smaller_score_map is None:
                    smaller_score_map = current_score_map
                else:
                    upscore_map = self._upscore_layer(smaller_score_map, current_score_map)
                    smaller_score_map = tf.concat([current_score_map, upscore_map], axis=0)

        return smaller_score_map

    def _fuse_by_cascade_conv1x1_upsample_sum(self, num_classes, scope):
        """
        The feature fuse fashion of FCN for semantic segmentation:
        Suppose there are several feature maps with decreasing sizes ,
        and we are going to get a single score map from them.

        Every feature map contributes to the final score map:
            predict score on all the feature maps using 1x1 conv, with
            depth equal to num_classes

        The score map is upsampled and added in a cascade way:
            start from the smallest score map, upsmale it to the size
            of the next score map with a larger size, and add them
            to get a fused score map. Upsample this fused score map and
            add it to the next sibling larger score map. The final
            score map is got when all score maps are fused together
        """
        import config
        num_layers = len(config.feat_layers)

        with tf.variable_scope(scope):
            smaller_score_map = None
            for idx in range(0, len(config.feat_layers))[::-1]:  # [4, 3, 2, 1, 0]
                current_layer_name = config.feat_layers[idx]
                current_layer = self.end_points[current_layer_name]
                current_score_map = self._score_layer(current_layer,
                                                      num_classes, current_layer_name)
                if smaller_score_map is None:
                    smaller_score_map = current_score_map
                else:
                    upscore_map = self._upscore_layer(smaller_score_map, current_score_map)
                    smaller_score_map = current_score_map + upscore_map

        return smaller_score_map

    def _fuse_feat_layers(self):
        import config
        if config.feat_fuse_type == FUSE_TYPE_cascade_conv1x1_upsample_sum:
            self.pixel_cls_logits = self._fuse_by_cascade_conv1x1_upsample_sum(
                config.num_classes, scope='pixel_cls')

            self.pixel_link_logits = self._fuse_by_cascade_conv1x1_upsample_sum(
                config.num_neighbours * 2, scope='pixel_link')

        elif config.feat_fuse_type == FUSE_TYPE_cascade_conv1x1_128_upsamle_sum_conv1x1_2:
            base_map = self._fuse_by_cascade_conv1x1_128_upsamle_sum_conv1x1_2(
                scope='fuse_feature')

            self.pixel_cls_logits = self._score_layer(base_map,
                                                      config.num_classes, scope='pixel_cls')

            self.pixel_link_logits = self._score_layer(base_map,
                                                       config.num_neighbours * 2, scope='pixel_link')
        elif config.feat_fuse_type == FUSE_TYPE_cascade_conv1x1_128_upsamle_concat_conv1x1_2:
            base_map = self._fuse_by_cascade_conv1x1_128_upsamle_concat_conv1x1_2(
                scope='fuse_feature')
        else:
            raise ValueError('feat_fuse_type not supported:%s' % (config.feat_fuse_type))

    def _flat_pixel_cls_values(self, values):
        shape = values.shape.as_list()
        values = tf.reshape(values, shape=[shape[0], -1, shape[-1]])
        return values

    def _logits_to_scores(self):
        self.pixel_cls_scores = tf.nn.softmax(self.pixel_cls_logits)
        # 将[N, W, H, C] -> [N, W*H, C]的格式
        self.pixel_cls_logits_flatten = \
            self._flat_pixel_cls_values(self.pixel_cls_logits)
        self.pixel_cls_scores_flatten = \
            self._flat_pixel_cls_values(self.pixel_cls_scores)

        import config
        #         shape = self.pixel_link_logits.shape.as_list()
        shape = tf.shape(self.pixel_link_logits)
        self.pixel_link_logits = tf.reshape(self.pixel_link_logits,
                                            [shape[0], shape[1], shape[2], config.num_neighbours, 2])

        self.pixel_link_scores = tf.nn.softmax(self.pixel_link_logits)

        self.pixel_pos_scores = self.pixel_cls_scores[:, :, :, 1]
        self.link_pos_scores = self.pixel_link_scores[:, :, :, :, 1]

    def build_loss(self, pixel_cls_labels, pixel_cls_weights,
                   pixel_link_labels, pixel_link_weights,
                   do_summary=True
                   ):
        """
        The loss consists of two parts: pixel_cls_loss + link_cls_loss,
            and link_cls_loss is calculated only on positive pixels
        """
        import config
        count_warning = tf.get_local_variable(
            name='count_warning', initializer=tf.constant(0.0))
        batch_size = config.batch_size_per_gpu
        ignore_label = config.ignore_label
        background_label = config.background_label
        text_label = config.text_label
        pixel_link_neg_loss_weight_lambda = config.pixel_link_neg_loss_weight_lambda
        pixel_cls_loss_weight_lambda = config.pixel_cls_loss_weight_lambda
        pixel_link_loss_weight = config.pixel_link_loss_weight

        def OHNM_single_image(scores, n_pos, neg_mask):
            """Online Hard Negative Mining.
                scores: the scores of being predicted as negative cls
                n_pos: the number of positive samples
                neg_mask: mask of negative samples
                Return:
                    the mask of selected negative samples.
                    if n_pos == 0, top 10000 negative samples will be selected.
            """

            def has_pos():
                return n_pos * config.max_neg_pos_ratio

            def no_pos():
                return tf.constant(10000, dtype=tf.int32)

            n_neg = tf.cond(n_pos > 0, has_pos, no_pos)
            max_neg_entries = tf.reduce_sum(tf.cast(neg_mask, tf.int32))

            n_neg = tf.minimum(n_neg, max_neg_entries)
            n_neg = tf.cast(n_neg, tf.int32)

            def has_neg():
                neg_conf = tf.boolean_mask(scores, neg_mask)
                vals, _ = tf.nn.top_k(-neg_conf, k=n_neg)
                threshold = vals[-1]  # a negtive value
                selected_neg_mask = tf.logical_and(neg_mask, scores <= -threshold)
                return selected_neg_mask

            def no_neg():
                selected_neg_mask = tf.zeros_like(neg_mask)
                return selected_neg_mask

            selected_neg_mask = tf.cond(n_neg > 0, has_neg, no_neg)
            return tf.cast(selected_neg_mask, tf.int32)

        def OHNM_batch(neg_conf, pos_mask, neg_mask):
            selected_neg_mask = []
            for image_idx in range(batch_size):
                image_neg_conf = neg_conf[image_idx, :]
                image_neg_mask = neg_mask[image_idx, :]
                image_pos_mask = pos_mask[image_idx, :]
                n_pos = tf.reduce_sum(tf.cast(image_pos_mask, tf.int32))
                selected_neg_mask.append(OHNM_single_image(image_neg_conf, n_pos, image_neg_mask))

            selected_neg_mask = tf.stack(selected_neg_mask)
            return selected_neg_mask

        # OHNM on pixel classification task
        pixel_cls_labels_flatten = tf.reshape(pixel_cls_labels, [batch_size, -1])
        pos_pixel_weights_flatten = tf.reshape(pixel_cls_weights, [batch_size, -1])

        pos_mask = tf.equal(pixel_cls_labels_flatten, text_label)
        neg_mask = tf.equal(pixel_cls_labels_flatten, background_label)

        n_pos = tf.reduce_sum(tf.cast(pos_mask, dtype=tf.float32))

        with tf.name_scope('pixel_cls_loss'):
            def no_pos():
                return tf.constant(.0);

            def has_pos():
                pixel_cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.pixel_cls_logits_flatten,
                    labels=tf.cast(pos_mask, dtype=tf.int32))

                pixel_neg_scores = self.pixel_cls_scores_flatten[:, :, 0]
                selected_neg_pixel_mask = OHNM_batch(pixel_neg_scores, pos_mask, neg_mask)

                pixel_cls_weights = pos_pixel_weights_flatten + \
                                    tf.cast(selected_neg_pixel_mask, tf.float32)
                n_neg = tf.cast(tf.reduce_sum(selected_neg_pixel_mask), tf.float32)
                loss = tf.reduce_sum(pixel_cls_loss * pixel_cls_weights) / (n_neg + n_pos)
                return loss

            #             pixel_cls_loss = tf.cond(n_pos > 0, has_pos, no_pos)
            pixel_cls_loss = has_pos()
            tf.add_to_collection(tf.GraphKeys.LOSSES, pixel_cls_loss * pixel_cls_loss_weight_lambda)

        with tf.name_scope('pixel_link_loss'):
            def no_pos():
                return tf.constant(.0), tf.constant(.0);

            def has_pos():
                pixel_link_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.pixel_link_logits,
                    labels=pixel_link_labels)

                def get_loss(label):
                    link_mask = tf.equal(pixel_link_labels, label)
                    link_weights = pixel_link_weights * tf.cast(link_mask, tf.float32)
                    n_links = tf.reduce_sum(link_weights)
                    loss = tf.reduce_sum(pixel_link_loss * link_weights) / n_links
                    return loss

                neg_loss = get_loss(0)
                pos_loss = get_loss(1)
                return neg_loss, pos_loss

            pixel_neg_link_loss, pixel_pos_link_loss = \
                tf.cond(n_pos > 0, has_pos, no_pos)

            pixel_link_loss = pixel_pos_link_loss + \
                              pixel_neg_link_loss * pixel_link_neg_loss_weight_lambda

            tf.add_to_collection(tf.GraphKeys.LOSSES,
                                 pixel_link_loss_weight * pixel_link_loss)

        if do_summary:
            tf.summary.scalar('pixel_cls_loss', pixel_cls_loss)
            tf.summary.scalar('pixel_pos_link_loss', pixel_pos_link_loss)
            tf.summary.scalar('pixel_neg_link_loss', pixel_neg_link_loss)

class PixelLinkNet(object):
    def __init__(self, inputs, is_training):
        self.inputs = inputs
        self.is_training = is_training
        self._build_network()
        self._fuse_feat_layers()
        self._logits_to_scores()
        
    def _build_network(self):
        import config
        if config.model_type == MODEL_TYPE_vgg16:
            from nets import vgg
            with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(config.weight_decay),
                        weights_initializer= tf.contrib.layers.xavier_initializer(),
                        biases_initializer = tf.zeros_initializer()):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    padding='SAME') as sc:
                    self.arg_scope = sc
                    self.net, self.end_points = vgg.basenet(
                              inputs =  self.inputs)
                    
        elif config.model_type == MODEL_TYPE_vgg16_no_dilation:
            from nets import vgg
            with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(config.weight_decay),
                        weights_initializer= tf.contrib.layers.xavier_initializer(),
                        biases_initializer = tf.zeros_initializer()):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    padding='SAME') as sc:
                    self.arg_scope = sc
                    self.net, self.end_points = vgg.basenet(
                              inputs =  self.inputs, dilation = False)
        else:
            raise ValueError('model_type not supported:%s'%(config.model_type))
        
    def _score_layer(self, input_layer, num_classes, scope):
        import config
        with slim.arg_scope(self.arg_scope):
            logits = slim.conv2d(input_layer, num_classes, [1, 1], 
                 stride=1,
                 activation_fn=None, 
                 scope='score_from_%s'%scope,
                 normalizer_fn=None)
            try:
                use_dropout = config.dropout_ratio > 0
            except:
                use_dropout = False
                
            if use_dropout:
                if self.is_training:
                    dropout_ratio = config.dropout_ratio
                else:
                    dropout_ratio = 0
                keep_prob = 1.0 - dropout_ratio
                tf.logging.info('Using Dropout, with keep_prob = %f'%(keep_prob))
                logits = tf.nn.dropout(logits, keep_prob)
            return logits
        
    def _upscore_layer(self, layer, target_layer):   
#             target_shape = target_layer.shape[1:-1] # NHWC
            target_shape = tf.shape(target_layer)[1:-1]
            upscored = tf.image.resize_images(layer, target_shape)
            return upscored        
    def _fuse_by_cascade_conv1x1_128_upsamle_sum_conv1x1_2(self, scope):
        """
        The feature fuse fashion of 
            'Deep Direct Regression for Multi-Oriented Scene Text Detection'
        
        Instead of fusion of scores, feature map from 1x1, 128 conv are fused,
        and the scores are predicted on it.
        """
        base_map = self._fuse_by_cascade_conv1x1_upsample_sum(num_classes = 128, 
                                                              scope = 'feature_fuse')
        return base_map
    
    def _fuse_by_cascade_conv1x1_128_upsamle_concat_conv1x1_2(self, scope, num_classes = 32):
        import config
        num_layers = len(config.feat_layers)
        
        with tf.variable_scope(scope):
            smaller_score_map = None
            for idx in range(0, len(config.feat_layers))[::-1]: #[4, 3, 2, 1, 0]
                current_layer_name = config.feat_layers[idx]
                current_layer = self.end_points[current_layer_name]
                current_score_map = self._score_layer(current_layer, 
                                      num_classes, current_layer_name)
                if smaller_score_map is None:
                    smaller_score_map = current_score_map
                else:
                    upscore_map = self._upscore_layer(smaller_score_map, current_score_map)
                    smaller_score_map = tf.concat([current_score_map, upscore_map], axis = 0)
            
        return smaller_score_map
    
        
    def _fuse_by_cascade_conv1x1_upsample_sum(self, num_classes, scope):
        """
        The feature fuse fashion of FCN for semantic segmentation:
        Suppose there are several feature maps with decreasing sizes , 
        and we are going to get a single score map from them.
        
        Every feature map contributes to the final score map:
            predict score on all the feature maps using 1x1 conv, with 
            depth equal to num_classes
            
        The score map is upsampled and added in a cascade way:
            start from the smallest score map, upsmale it to the size
            of the next score map with a larger size, and add them 
            to get a fused score map. Upsample this fused score map and
            add it to the next sibling larger score map. The final 
            score map is got when all score maps are fused together 
        """
        import config
        num_layers = len(config.feat_layers)
        
        with tf.variable_scope(scope):
            smaller_score_map = None
            for idx in range(0, len(config.feat_layers))[::-1]: #[4, 3, 2, 1, 0]
                current_layer_name = config.feat_layers[idx]
                current_layer = self.end_points[current_layer_name]
                current_score_map = self._score_layer(current_layer, 
                                      num_classes, current_layer_name)
                if smaller_score_map is None:
                    smaller_score_map = current_score_map
                else:
                    upscore_map = self._upscore_layer(smaller_score_map, current_score_map)
                    smaller_score_map = current_score_map + upscore_map
            
        return smaller_score_map
            
    def _fuse_feat_layers(self):
        import config
        if config.feat_fuse_type == FUSE_TYPE_cascade_conv1x1_upsample_sum:
            self.pixel_cls_logits = self._fuse_by_cascade_conv1x1_upsample_sum(
                config.num_classes, scope = 'pixel_cls')
            
            self.pixel_link_logits = self._fuse_by_cascade_conv1x1_upsample_sum(
                config.num_neighbours * 2, scope = 'pixel_link')
            
        elif config.feat_fuse_type == FUSE_TYPE_cascade_conv1x1_128_upsamle_sum_conv1x1_2:
            base_map = self._fuse_by_cascade_conv1x1_128_upsamle_sum_conv1x1_2(
                                    scope = 'fuse_feature')
            
            self.pixel_cls_logits = self._score_layer(base_map,
                  config.num_classes, scope = 'pixel_cls')
            
            self.pixel_link_logits = self._score_layer(base_map,
                   config.num_neighbours  * 2, scope = 'pixel_link')
        elif config.feat_fuse_type == FUSE_TYPE_cascade_conv1x1_128_upsamle_concat_conv1x1_2:
            base_map = self._fuse_by_cascade_conv1x1_128_upsamle_concat_conv1x1_2(
                                    scope = 'fuse_feature')
        else:
            raise ValueError('feat_fuse_type not supported:%s'%(config.feat_fuse_type))
        
    def _flat_pixel_cls_values(self, values):
        shape = values.shape.as_list()
        values = tf.reshape(values, shape = [shape[0], -1, shape[-1]])
        return values
    
        
    def _logits_to_scores(self):
        self.pixel_cls_scores = tf.nn.softmax(self.pixel_cls_logits)
        # 将[N, W, H, C] -> [N, W*H, C]的格式
        self.pixel_cls_logits_flatten = \
            self._flat_pixel_cls_values(self.pixel_cls_logits)
        self.pixel_cls_scores_flatten = \
            self._flat_pixel_cls_values(self.pixel_cls_scores)
            
        import config
#         shape = self.pixel_link_logits.shape.as_list()
        shape = tf.shape(self.pixel_link_logits)
        self.pixel_link_logits = tf.reshape(self.pixel_link_logits, 
                                [shape[0], shape[1], shape[2], config.num_neighbours, 2])
            
        self.pixel_link_scores = tf.nn.softmax(self.pixel_link_logits)
        
        self.pixel_pos_scores = self.pixel_cls_scores[:, :, :, 1]
        self.link_pos_scores = self.pixel_link_scores[:, :, :, :, 1]
        
    def build_loss(self, pixel_cls_labels, pixel_cls_weights, 
                        pixel_link_labels, pixel_link_weights,
                        do_summary = True
                        ):      
        """
        The loss consists of two parts: pixel_cls_loss + link_cls_loss, 
            and link_cls_loss is calculated only on positive pixels
        """
        import config
        count_warning = tf.get_local_variable(
            name = 'count_warning', initializer = tf.constant(0.0))
        batch_size = config.batch_size_per_gpu
        ignore_label = config.ignore_label
        background_label = config.background_label
        text_label = config.text_label
        pixel_link_neg_loss_weight_lambda = config.pixel_link_neg_loss_weight_lambda
        pixel_cls_loss_weight_lambda = config.pixel_cls_loss_weight_lambda
        pixel_link_loss_weight = config.pixel_link_loss_weight
        
        def OHNM_single_image(scores, n_pos, neg_mask):
            """Online Hard Negative Mining.
                scores: the scores of being predicted as negative cls
                n_pos: the number of positive samples 
                neg_mask: mask of negative samples
                Return:
                    the mask of selected negative samples.
                    if n_pos == 0, top 10000 negative samples will be selected.
            """
            def has_pos():
                return n_pos * config.max_neg_pos_ratio
            def no_pos():
                return tf.constant(10000, dtype = tf.int32)
            
            n_neg = tf.cond(n_pos > 0, has_pos, no_pos)
            max_neg_entries = tf.reduce_sum(tf.cast(neg_mask, tf.int32))
                
            n_neg = tf.minimum(n_neg, max_neg_entries)
            n_neg = tf.cast(n_neg, tf.int32)
            def has_neg():
                neg_conf = tf.boolean_mask(scores, neg_mask)
                vals, _ = tf.nn.top_k(-neg_conf, k=n_neg)
                threshold = vals[-1]# a negtive value
                selected_neg_mask = tf.logical_and(neg_mask, scores <= -threshold)
                return selected_neg_mask
            def no_neg():
                selected_neg_mask = tf.zeros_like(neg_mask)
                return selected_neg_mask
            
            selected_neg_mask = tf.cond(n_neg > 0, has_neg, no_neg)
            return tf.cast(selected_neg_mask, tf.int32)
        
        def OHNM_batch(neg_conf, pos_mask, neg_mask):
            selected_neg_mask = []
            for image_idx in range(batch_size):
                image_neg_conf = neg_conf[image_idx, :]
                image_neg_mask = neg_mask[image_idx, :]
                image_pos_mask = pos_mask[image_idx, :]
                n_pos = tf.reduce_sum(tf.cast(image_pos_mask, tf.int32))
                selected_neg_mask.append(OHNM_single_image(image_neg_conf, n_pos, image_neg_mask))
                
            selected_neg_mask = tf.stack(selected_neg_mask)
            return selected_neg_mask
        
        # OHNM on pixel classification task
        pixel_cls_labels_flatten = tf.reshape(pixel_cls_labels, [batch_size, -1])
        pos_pixel_weights_flatten = tf.reshape(pixel_cls_weights, [batch_size, -1])
        
        pos_mask = tf.equal(pixel_cls_labels_flatten, text_label)
        neg_mask = tf.equal(pixel_cls_labels_flatten, background_label)

        n_pos = tf.reduce_sum(tf.cast(pos_mask, dtype = tf.float32))

        with tf.name_scope('pixel_cls_loss'):            
            def no_pos():
                return tf.constant(.0);
            def has_pos():
                pixel_cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits = self.pixel_cls_logits_flatten, 
                    labels = tf.cast(pos_mask, dtype = tf.int32))
                
                pixel_neg_scores = self.pixel_cls_scores_flatten[:, :, 0]
                selected_neg_pixel_mask = OHNM_batch(pixel_neg_scores, pos_mask, neg_mask)
                
                pixel_cls_weights = pos_pixel_weights_flatten + \
                            tf.cast(selected_neg_pixel_mask, tf.float32)
                n_neg = tf.cast(tf.reduce_sum(selected_neg_pixel_mask), tf.float32)
                loss = tf.reduce_sum(pixel_cls_loss * pixel_cls_weights) / (n_neg + n_pos)
                return loss
            
#             pixel_cls_loss = tf.cond(n_pos > 0, has_pos, no_pos)
            pixel_cls_loss = has_pos()
            tf.add_to_collection(tf.GraphKeys.LOSSES, pixel_cls_loss * pixel_cls_loss_weight_lambda)
            
        
        with tf.name_scope('pixel_link_loss'):
            def no_pos():
                return tf.constant(.0), tf.constant(.0);
            
            def has_pos():
                pixel_link_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits = self.pixel_link_logits, 
                    labels = pixel_link_labels)
                
                def get_loss(label):
                    link_mask = tf.equal(pixel_link_labels, label)
                    link_weights = pixel_link_weights * tf.cast(link_mask, tf.float32)
                    n_links = tf.reduce_sum(link_weights)
                    loss = tf.reduce_sum(pixel_link_loss * link_weights) / n_links
                    return loss
                
                neg_loss = get_loss(0)
                pos_loss = get_loss(1)
                return neg_loss, pos_loss
            
            pixel_neg_link_loss, pixel_pos_link_loss = \
                        tf.cond(n_pos > 0, has_pos, no_pos)
            
            pixel_link_loss = pixel_pos_link_loss + \
                    pixel_neg_link_loss * pixel_link_neg_loss_weight_lambda
                    
            tf.add_to_collection(tf.GraphKeys.LOSSES, 
                                 pixel_link_loss_weight * pixel_link_loss)
            
        if do_summary:
            tf.summary.scalar('pixel_cls_loss', pixel_cls_loss)
            tf.summary.scalar('pixel_pos_link_loss', pixel_pos_link_loss)
            tf.summary.scalar('pixel_neg_link_loss', pixel_neg_link_loss)
