import tensorflow as tf


def compute_dice_loss(gt_tensor, pred_tensor, epsilon=1e-6, axis=[1, 2], seg_num_classes=5):
    softmaxed = tf.nn.softmax(pred_tensor)

    cond = tf.less(softmaxed, 0.5)
    output = tf.where(cond, tf.zeros(tf.shape(softmaxed)), tf.ones(tf.shape(softmaxed)))

    target = gt_tensor  # tf.one_hot(labels, depth=2)

    # Make sure inferred shapes are equal during graph construction.
    # Do a static check as shapes won't change dynamically.
    print(output.get_shape().as_list(), target.get_shape().as_list())
    assert output.get_shape().as_list() == target.get_shape().as_list()
    dice_op = None
    dice_loss_tensor = None
    dice_tensors = []
    for idx in range(1, seg_num_classes):
        with tf.name_scope('dice_%d' % idx):
            cur_output = output[:, :, :, idx]
            cur_target = target[:, :, :, idx]
            cur_output = tf.cast(cur_output, tf.float32)
            cur_target = tf.cast(cur_target, tf.float32)
            inse = tf.reduce_sum(cur_output * cur_target, axis=axis)
            l = tf.reduce_sum(cur_output, axis=axis)
            r = tf.reduce_sum(cur_target, axis=axis)
            dice = (2. * inse + epsilon) / (l + r + epsilon)
            dice = tf.reduce_mean(dice)
            dice_tensors.append(dice)
            dice_loss = (1 - dice) + (1 / dice) ** 0.3
            if dice_op is None:
                dice_op = dice

                dice_loss_tensor = dice_loss
            else:
                dice_op += dice
                dice_loss_tensor += dice_loss
    dice_op /= (seg_num_classes - 1.0)
    dice_loss_tensor /= (seg_num_classes - 1.0)
    return dice_loss_tensor, dice_op, dice_tensors


def compute_seg_loss(gt_tensor, pred_tensor, num_classes, epsilon=1e-6, axis=[1, 2]):
    gt_tensor = tf.cast(gt_tensor, tf.int32)
    class_weights = tf.constant([0.1, 10.0, 50.0, 10.0, 50.0, 20.0])
    weights = tf.gather(class_weights, gt_tensor)
    cross_entropy_loss_tensor = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=pred_tensor,
                                                                                      weights=weights,
                                                                                      loss_collection=None,
                                                                                      labels=gt_tensor))

    # dice_loss_tensor, dice_tensor, dice_tensors = compute_dice_loss(tf.one_hot(gt_tensor, depth=num_classes + 1),
    #                                                                 pred_tensor, epsilon, axis, num_classes + 1)

    # seg_loss_tensor = dice_loss_tensor + cross_entropy_loss_tensor

    seg_loss_tensor = cross_entropy_loss_tensor
    # seg_loss_tensor = cross_entropy_loss_tensor
    # return seg_loss_tensor, dice_tensor, dice_tensors
    return seg_loss_tensor