import tensorflow as tf
# import core.utils as utils
# from core.config import cfg
from core.nn_skeleton import ModelSkeleton
import json

BATCH_NORM_MOMENTUM = 0.997
BATCH_NORM_EPSILON = 1e-3
BATCH_NORM_EPSILON_SEPARABLE = 1e-5
DEPTH = 116
OUT_STRIDE = 16
DROPOUT_KEEP_PROB = 0.9
NUMCALSS = 19
BN_FUSION = True

def get_stride(layer_stride, current_stride, rate):
    """
    Function to adjust the output_stride using atrous rate
    获得空洞卷积步长，只有几层是2，进行下采样的时候
    """
    if current_stride == OUT_STRIDE:
        layer_stride = 1
        rate *= layer_stride
    else:
        rate = 1
        current_stride *= layer_stride
    return layer_stride, current_stride, rate


class shuffle_net_v2(object):
    def __init__(self, input_data, trainable):
        self.bn_momentum = BATCH_NORM_MOMENTUM
        self.bn_epsolom = BATCH_NORM_EPSILON
        self.input_data = input_data
        self.trainable = trainable
        self.numclass = NUMCALSS
        self.scale = [1024, 2048]

        self.nnlib = ModelSkeleton(trainable, self.bn_epsolom)

    def shuffle_net(self):
        self.bn_epsolom = BATCH_NORM_EPSILON
        self.nnlib.BATCH_NORM_EPSILON = self.bn_epsolom
        current_stride = 1
        layer_stride = 2
        rate = 1
        end_points = {}

        # 设置此模块中是否使用浮点量化
        self.nnlib.quant = False

        # 第一层卷积与池化
        check = False
        layer_stride, current_stride, rate = get_stride(
            layer_stride, current_stride, rate)
        net = self.nnlib.conv_bn_relu_layer(self.input_data, 'Conv1', (3, 3, 3, 24), self.trainable, bn_fusion=BN_FUSION, downsample=True, check=check)
        tf.add_to_collection("conv1", net)
        check = False
        layer_stride, current_stride, rate = get_stride(
            layer_stride, current_stride, rate)
        net = tf.nn.max_pool(net, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name='Maxpool')
        end_points['Stage1'] = net

        # stage2-4
        layers = [
            {'num_units': 4, 'out_channels': DEPTH,
             'scope': 'Stage2', 'stride': 2},
            {'num_units': 8, 'out_channels': None,
             'scope': 'Stage3', 'stride': 2},
            {'num_units': 4, 'out_channels': None,
             'scope': 'Stage4', 'stride': 2},
        ]
        for i in range(3):
            layer = layers[i]
            layer_rate = rate
            layer_stride, current_stride, rate = get_stride(
                layer['stride'], current_stride, rate
            )
            # check = True if i == 0 else False
            with tf.variable_scope(layer['scope']):
                with tf.variable_scope('1'):
                    x, y = self.nnlib.basic_unit_with_downsampling(
                        net, out_channels=layer['out_channels'], stride=layer_stride, rate=(layer_rate, layer_rate), trainable=self.trainable, bn_fusion=BN_FUSION, check=check)

                for j in range(2, layer['num_units'] + 1):
                    with tf.variable_scope('%d' % j):
                        x, y = self.nnlib.concat_shuffle_split(x, y)
                        x = self.nnlib.basic_unit(x, (rate, rate), self.trainable, BN_FUSION)
                x = tf.concat([x, y], axis=3)

            net = x
        return net, end_points

    def dense_predition_cell(self, input_tensor):
        self.bn_epsolom = BATCH_NORM_EPSILON_SEPARABLE
        self.nnlib.BATCH_NORM_EPSILON = self.bn_epsolom
        # 设置此模块中是否使用浮点量化
        self.nnlib.quant = False

        with tf.gfile.Open('dense_prediction_cell_branch5_top1_cityscapes.json', 'r') as f:
            dense_prediction_cell_config = json.load(f)
        branch_logits = []
        with tf.variable_scope("DPC"):
            for i, current_config in enumerate(dense_prediction_cell_config):
                if current_config["input"] < 0:
                    operation_input = input_tensor
                else:
                    operation_input = branch_logits[current_config["input"]]
                in_channels = operation_input.get_shape().as_list()[3]
                scope = "branch%d" % i
                rate = current_config["rate"]
                with tf.variable_scope(scope):
                    x = self.nnlib.separable_conv2d(operation_input, "d", (3, 3, in_channels, 1), (1, 1, 1, 1), self.trainable, rate=rate, bn_fusion=BN_FUSION)
                    x = tf.nn.relu(x)
                    x = self.nnlib.conv_bn_relu_layer(x, "p", (1, 1, in_channels, 256), self.trainable, bn_fusion=BN_FUSION)
                    branch_logits.append(x)
            concat_logits = tf.concat(branch_logits, 3)
            # concat_logits = tf.nn.dropout(concat_logits, DROPOUT_KEEP_PROB)

        concat_logits = self.nnlib.conv_bn_relu_layer(concat_logits, "Conv2", (1, 1, 256*5, 256), self.trainable, bn_fusion=BN_FUSION)
        return concat_logits

    def prediction(self, input_tensor):
        # 设置此模块中是否使用浮点量化
        self.nnlib.quant = False

        in_channels = input_tensor.shape[3]
        predict = self.nnlib.conv_layer(input_tensor, "Conv3", [1, 1, in_channels, self.numclass], self.trainable)
        predict = tf.image.resize_bilinear(predict, self.scale, True)
        return predict

    def naive(self, input_tensor):
        # 设置此模块中是否使用浮点量化
        self.nnlib.quant = False

        pool_height = tf.shape(input_tensor)[1]
        pool_width = tf.shape(input_tensor)[2]
        in_channels = input_tensor.shape[3]
        image_feature = tf.reduce_mean(
            input_tensor, axis=[1, 2], keep_dims=True)
        resize_height = pool_height
        resize_width = pool_width
        image_feature = self.nnlib.conv_bn_relu_layer(image_feature, "image_pooling", (1, 1, in_channels, 256), self.trainable)
        image_feature = tf.image.resize_bilinear(image_feature, [resize_height, resize_width], True)
        image_feature = tf.cast(image_feature, dtype=tf.float32)

        aspp0 = self.nnlib.conv_bn_relu_layer(input_tensor, "aspp0", (1, 1, in_channels, 256), self.trainable)

        concat_logits = tf.concat([image_feature, aspp0], 3)
        concat_logits = self.nnlib.conv_bn_relu_layer(concat_logits, "concat_projection", (1, 1, 512, 256), self.trainable)

        return  concat_logits













