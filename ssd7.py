
from __future__ import division

import keras
import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, ELU, Reshape, Concatenate, Activation, \
    GlobalAveragePooling2D
from keras.regularizers import l2
import keras.backend as K

from SSD.keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from SSD.keras_layers.keras_layer_DecodeDetections import DecodeDetections
from SSD.keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
import tensorflow as tf

def Conv_Block(kernel, kernel_size, pad, stride, input_layer):

    temp = input_layer
    if pad is not 'same':
        temp = keras.layers.ZeroPadding2D(padding=((0, pad), (0, pad)))(input_layer)

    layer = Conv2D(kernel, (kernel_size, kernel_size), padding='valid', strides=stride)(temp)
    layer_bn = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(layer)
    layer_ac = keras.layers.Activation('relu')(layer_bn)
    return layer_ac


def Dense_Block(num_layer, input_layer, bottleneck_width, growth_rate = 32):

    layer = input_layer
    x = (int)(growth_rate/2)
    x = x * bottleneck_width
    for i in range(0, num_layer):

        dense_block_1_1 = Conv_Block(x, 1,'same', 1, layer)
        dense_block_1_2 = Conv_Block((int)(x/2), 3, 2, 1, dense_block_1_1)

        dense_block_2_1 = Conv_Block(x, 1,'same', 1, layer)
        dense_block_2_2 = Conv_Block((int)(x/2), 3, 2, 1, dense_block_2_1)
        dense_block_2_3 = Conv_Block((int)(x/2), 3, 2, 1, dense_block_2_2)
        dense_filter_con = keras.layers.concatenate([layer, dense_block_1_2, dense_block_2_3], axis=3)

        layer = dense_filter_con

    return layer


def build_model(image_size,
                n_classes,
                mode='training',
                l2_regularization=0.0,
                min_scale=0.1,
                max_scale=0.9,
                scales=None,
                aspect_ratios_global=[0.5, 1.0, 2.0],
                aspect_ratios_per_layer=None,
                two_boxes_for_ar1=True,
                steps=None,
                offsets=None,
                clip_boxes=False,
                variances=[1.0, 1.0, 1.0, 1.0],
                coords='centroids',
                normalize_coords=False,
                subtract_mean=None,
                divide_by_stddev=None,
                swap_channels=False,
                confidence_thresh=0.01,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400,
                return_predictor_sizes=False):


    n_predictor_layers = 4 # The number of predictor conv layers in the network
    n_classes += 1 # Account for the background class.
    l2_reg = l2_regularization # Make the internal name shorter.
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    ############################################################################
    # Get a few exceptions out of the way.
    ############################################################################

    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError("`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers+1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(n_predictor_layers+1, len(scales)))
    else: # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers+1)

    if len(variances) != 4: # We need one variance value for each of the four box coordinates
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")

    ############################################################################
    # Compute the anchor box parameters.
    ############################################################################

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1) # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))
    else: # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    ############################################################################
    # Define functions for the Lambda layers below.
    ############################################################################

    def identity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    def input_channel_swap(tensor):
        if len(swap_channels) == 3:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]]], axis=-1)
        elif len(swap_channels) == 4:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]], tensor[...,swap_channels[3]]], axis=-1)

    ############################################################################
    # Build the network.
    ############################################################################

    x = Input(shape=(img_height, img_width, img_channels))

    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(x)
    if not (subtract_mean is None):
        x1 = Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels), name='input_mean_normalization')(x1)
    if not (divide_by_stddev is None):
        x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels), name='input_stddev_normalization')(x1)
    if swap_channels:
        x1 = Lambda(input_channel_swap, output_shape=(img_height, img_width, img_channels), name='input_channel_swap')(x1)

    # conv1 = Conv2D(32, (5, 5), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1')(x1)
    # conv1 = BatchNormalization(axis=3, momentum=0.99, name='bn1')(conv1) # Tensorflow uses filter format [filter_height, filter_width, in_channels, out_channels], hence axis = 3
    # conv1 = ELU(name='elu1')(conv1)
    # pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)
    #
    # conv2 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv2')(pool1)
    # conv2 = BatchNormalization(axis=3, momentum=0.99, name='bn2')(conv2)
    # conv2 = ELU(name='elu2')(conv2)
    # pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)
    #
    # conv3 = Conv2D(64, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3')(pool2)
    # conv3 = BatchNormalization(axis=3, momentum=0.99, name='bn3')(conv3)
    # conv3 = ELU(name='elu3')(conv3)
    # pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3)
    #
    # conv4 = Conv2D(64, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4')(pool3)
    # conv4 = BatchNormalization(axis=3, momentum=0.99, name='bn4')(conv4)
    # conv4 = ELU(name='elu4')(conv4)
    # pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(conv4)
    #
    # conv5 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5')(pool4)
    # conv5 = BatchNormalization(axis=3, momentum=0.99, name='bn5')(conv5)
    # conv5 = ELU(name='elu5')(conv5)
    # pool5 = MaxPooling2D(pool_size=(2, 2), name='pool5')(conv5)
    #
    # conv6 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6')(pool5)
    # conv6 = BatchNormalization(axis=3, momentum=0.99, name='bn6')(conv6)
    # conv6 = ELU(name='elu6')(conv6)
    # pool6 = MaxPooling2D(pool_size=(2, 2), name='pool6')(conv6)
    #
    # conv7 = Conv2D(32, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7')(pool6)
    # conv7 = BatchNormalization(axis=3, momentum=0.99, name='bn7')(conv7)
    # conv7 = ELU(name='elu7')(conv7)


    #**********************************************************

    dense_layer = [3, 4, 8, 6]
    bottleneck_width = [1, 2, 4, 4]

    # Stage - 0
    # STEM BLOCK - based on Inception v4

    # input - 224*224*3

    block = Conv_Block(32, 3, 1, 2, x1)
    block_1_1 = Conv_Block(16, 1, 'same', 1, block)
    block_1_2 = Conv_Block(32, 3, 1, 2, block_1_1)
    block_2_1 = MaxPooling2D((2, 2), strides=2, padding='same')(block)
    filter_con = keras.layers.concatenate([block_1_2, block_2_1], axis=3)
    output_stem = Conv_Block(32, 1, 'same', 1, filter_con)

    # output - 56*56*32

    # Stage - 1
    # input - 56*56*32
    # DENSE LAYER - originally total 3

    s1_dense_block = Dense_Block(dense_layer[0], output_stem, bottleneck_width[0])

    # TRANSITION LAYER

    s1_trans_1 = Conv2D(128, (1, 1), padding='same', activation='relu', strides=1)(s1_dense_block)
    s1_trans_2 = MaxPooling2D((2, 2), strides=2, padding='same')(s1_trans_1)
    output_stage_1 = s1_trans_2

    # output - 28*28*128

    # Stage - 2
    # input - 28*28*128
    # DENSE LAYER - originally total 4

    s2_dense_block = Dense_Block(dense_layer[1], output_stage_1, bottleneck_width[1])

    # TRANSITION LAYER

    s2_trans_1 = Conv2D(256, (1, 1), padding='same', activation='relu', strides=1)(s2_dense_block)
    s2_trans_2 = MaxPooling2D((2, 2), strides=2, padding='same')(s2_trans_1)
    output_stage_2 = s2_trans_2

    # output - 14*14*256

    # Stage - 3
    # input - 14*14*256
    # DENSE LAYER - originally total 8

    s3_dense_block = Dense_Block(dense_layer[2], output_stage_2, bottleneck_width[2])

    # TRANSITION LAYER

    s3_trans_1 = Conv2D(512, (1, 1), padding='same', activation='relu', strides=1)(s3_dense_block)
    s3_trans_2 = MaxPooling2D((2, 2), strides=2, padding='same')(s3_trans_1)
    output_stage_3 = s3_trans_2

    # output - 7*7*512

    # Stage - 4
    # input - 7*7*512
    # DENSE LAYER - originally total 6

    s4_dense_block = Dense_Block(dense_layer[3], output_stage_3, bottleneck_width[3])

    # TRANSITION LAYER

    s4_trans_1 = Conv2D(704, (1, 1), padding='same', activation='relu', strides=1)(s4_dense_block)
    output_stage_4 = s4_trans_1

    # output - 7*7*704

    # Stage - 5
    # input - 7*7*704

    output_stage_5 = GlobalAveragePooling2D()(output_stage_4)

    # output - 1*1*704

    # CLASSIFICATION LAYER - SOFTMAX

    output = keras.layers.Dense(10, activation='softmax')(output_stage_5)



    #***********************************************************



    # The next part is to add the convolutional predictor layers on top of the base network
    # that we defined above. Note that I use the term "base network" differently than the paper does.
    # To me, the base network is everything that is not convolutional predictor layers or anchor
    # box layers. In this case we'll have four predictor layers, but of course you could
    # easily rewrite this into an arbitrarily deep base network and add an arbitrary number of
    # predictor layers on top of the base network by simply following the pattern shown here.

    # Build the convolutional predictor layers on top of conv layers 4, 5, 6, and 7.
    # We build two predictor layers on top of each of these layers: One for class prediction (classification), one for box coordinate prediction (localization)
    # We precidt `n_classes` confidence values for each box, hence the `classes` predictors have depth `n_boxes * n_classes`
    # We predict 4 box coordinates for each box, hence the `boxes` predictors have depth `n_boxes * 4`
    # Output shape of `classes`: `(batch, height, width, n_boxes * n_classes)`
    classes4 = Conv2D(n_boxes[0] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes4')(output_stage_1)
    classes5 = Conv2D(n_boxes[1] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes5')(output_stage_2)
    classes6 = Conv2D(n_boxes[2] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes6')(output_stage_3)
    classes7 = Conv2D(n_boxes[3] * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes7')(output_stage_4)
    # Output shape of `boxes`: `(batch, height, width, n_boxes * 4)`
    boxes4 = Conv2D(n_boxes[0] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes4')(output_stage_1)
    boxes5 = Conv2D(n_boxes[1] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes5')(output_stage_2)
    boxes6 = Conv2D(n_boxes[2] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes6')(output_stage_3)
    boxes7 = Conv2D(n_boxes[3] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes7')(output_stage_4)

    # Generate the anchor boxes
    # Output shape of `anchors`: `(batch, height, width, n_boxes, 8)`
    anchors4 = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios[0],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0],
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors4')(boxes4)
    anchors5 = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios[1],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1],
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors5')(boxes5)
    anchors6 = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios[2],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2], this_offsets=offsets[2],
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors6')(boxes6)
    anchors7 = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4], aspect_ratios=aspect_ratios[3],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3], this_offsets=offsets[3],
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors7')(boxes7)

    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    classes4_reshaped = Reshape((-1, n_classes), name='classes4_reshape')(classes4)
    classes5_reshaped = Reshape((-1, n_classes), name='classes5_reshape')(classes5)
    classes6_reshaped = Reshape((-1, n_classes), name='classes6_reshape')(classes6)
    classes7_reshaped = Reshape((-1, n_classes), name='classes7_reshape')(classes7)
    # Reshape the box coordinate predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    boxes4_reshaped = Reshape((-1, 4), name='boxes4_reshape')(boxes4)
    boxes5_reshaped = Reshape((-1, 4), name='boxes5_reshape')(boxes5)
    boxes6_reshaped = Reshape((-1, 4), name='boxes6_reshape')(boxes6)
    boxes7_reshaped = Reshape((-1, 4), name='boxes7_reshape')(boxes7)
    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    anchors4_reshaped = Reshape((-1, 8), name='anchors4_reshape')(anchors4)
    anchors5_reshaped = Reshape((-1, 8), name='anchors5_reshape')(anchors5)
    anchors6_reshaped = Reshape((-1, 8), name='anchors6_reshape')(anchors6)
    anchors7_reshaped = Reshape((-1, 8), name='anchors7_reshape')(anchors7)

    # Concatenate the predictions from the different layers and the assosciated anchor box tensors
    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1
    # Output shape of `classes_concat`: (batch, n_boxes_total, n_classes)
    classes_concat = Concatenate(axis=1, name='classes_concat')([classes4_reshaped,
                                                                 classes5_reshaped,
                                                                 classes6_reshaped,
                                                                 classes7_reshaped])

    # Output shape of `boxes_concat`: (batch, n_boxes_total, 4)
    boxes_concat = Concatenate(axis=1, name='boxes_concat')([boxes4_reshaped,
                                                             boxes5_reshaped,
                                                             boxes6_reshaped,
                                                             boxes7_reshaped])

    # Output shape of `anchors_concat`: (batch, n_boxes_total, 8)
    anchors_concat = Concatenate(axis=1, name='anchors_concat')([anchors4_reshaped,
                                                                 anchors5_reshaped,
                                                                 anchors6_reshaped,
                                                                 anchors7_reshaped])

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    classes_softmax = Activation('softmax', name='classes_softmax')(classes_concat)

    # Concatenate the class and box coordinate predictions and the anchors to one large predictions tensor
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([classes_softmax, boxes_concat, anchors_concat])

    if mode == 'training':
        model = Model(inputs=x, outputs=predictions)
    elif mode == 'inference':
        decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
                                               iou_threshold=iou_threshold,
                                               top_k=top_k,
                                               nms_max_output_size=nms_max_output_size,
                                               coords=coords,
                                               normalize_coords=normalize_coords,
                                               img_height=img_height,
                                               img_width=img_width,
                                               name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    elif mode == 'inference_fast':
        decoded_predictions = DecodeDetectionsFast(confidence_thresh=confidence_thresh,
                                                   iou_threshold=iou_threshold,
                                                   top_k=top_k,
                                                   nms_max_output_size=nms_max_output_size,
                                                   coords=coords,
                                                   normalize_coords=normalize_coords,
                                                   img_height=img_height,
                                                   img_width=img_width,
                                                   name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    else:
        raise ValueError("`mode` must be one of 'training', 'inference' or 'inference_fast', but received '{}'.".format(mode))

    if return_predictor_sizes:
        # The spatial dimensions are the same for the `classes` and `boxes` predictor layers.
        predictor_sizes = np.array([classes4._keras_shape[1:3],
                                    classes5._keras_shape[1:3],
                                    classes6._keras_shape[1:3],
                                    classes7._keras_shape[1:3]])
        return model, predictor_sizes
    else:
        return model


