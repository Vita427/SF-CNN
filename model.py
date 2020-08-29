import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import tensorflow.contrib.slim as slim


def conv_layer(input, filter, kernel, active = None, stride=1, padding='valid', layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=True, filters=filter, kernel_size=kernel,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                   strides=stride, padding=padding)
        if active == 'sigmoid':
            network = Sigmoid(network)
        elif active=='relu':
            network = Relu(network)
        else:
            pass
        return network

def dconv_layer(input, filter, kernel, active = None, depth_multiplier=1, stride=1, padding='valid', layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.separable_conv2d(inputs=input,depth_multiplier=depth_multiplier,
                                             depthwise_initializer=tf.contrib.layers.xavier_initializer(),
                                             pointwise_initializer=tf.contrib.layers.xavier_initializer(),
                                             use_bias=True, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        if active == 'sigmoid':
            network = Sigmoid(network)
        elif active=='relu':
            network = Relu(network)
        else:
            pass
        return network

def deconv_layer(input, filter, active = None,stride=1, padding='VALID', layer_name="conv"):
    with tf.name_scope(layer_name):
        # network = tf.nn.depthwise_conv2d_native(input, filter, strides, padding, name=None)
        print(filter)
        w1 = tf.Variable(tf.random_normal(filter))

        b1 = tf.Variable(tf.random_normal(filter[3]))

        network = tf.nn.depthwise_conv2d_native(input=input, filter=w1, strides=[1, stride, stride, 1], padding=padding, name=None)
        network = tf.nn.bias_add(network,b1)
        if active == 'sigmoid':
            network = Sigmoid(network)
        elif active=='relu':
            network = Relu(network)
        else:
            pass
        return network

def atrous_layer(input, filter, active = None, rate=1, padding='VALID', layer_name="atrous"):
    with tf.name_scope(layer_name):
        # network = tf.nn.depthwise_conv2d_native(input, filter, strides, padding, name=None)
        print("ffffffffffffffffff",filter)
        w1 = tf.Variable(tf.random_normal(filter))
        # print("shape of w1", w1.shape)
        b1 = tf.Variable(tf.random_normal([filter[3]]))

        network = tf.nn.atrous_conv2d(value=input, filters=w1, rate=rate, padding=padding, name=None )
        network = tf.nn.bias_add(network,b1)
        if active == 'sigmoid':
            network = Sigmoid(network)
        elif active=='relu':
            network = Relu(network)
        else:
            pass
        return network



def Fully_connected(x, units=100, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=True, units=units)

def Relu(x):
    return tf.nn.relu(x)

def Sigmoid(x):
    return tf.nn.sigmoid(x)

def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')

def Max_pooling(x, pool_size=[3,3], stride=2, padding='VALID') :
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Avg_pooling(x, pool_size=[3,3], stride=1, padding='SAME') :
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Dropout(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)


def Squeeze_excitation_layer(input_x, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name) :
        squeeze = Global_Average_Pooling(input_x)
        excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
        excitation = Relu(excitation)
        excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
        excitation = Sigmoid(excitation)
        excitation = tf.reshape(excitation, [-1,1,1,out_dim])
        scale = input_x * excitation
        return scale

K = 5
class Model():
    def __init__(self, x1, x2, y_, training, n_classes,rate):
        self.training = training
        self.rate = rate
        self.sim = False
        self.model = self.network(x1,n_classes)
        self.y_ = y_
        self.o1 = None
        self.o2 = None

        with tf.variable_scope('siamese') as scope:
            _, self.o1 = self.network(x1,n_classes)
            scope.reuse_variables()
            _, self.o2 = self.network(x2,n_classes)
        self.loss = self.loss_with_spring()


    def loss_with_spring(self):
        margin = 5.0   # alpha
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")
        self.o1 = tf.reduce_sum(tf.reshape(self.o1, [-1, K, 128]), 1) / K
        self.o2 = tf.reduce_sum(tf.reshape(self.o2, [-1, K, 128]), 1) / K
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        print("loss shape",eucd2.shape)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
        neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss

    def network(self, x, n_classes):
        net = slim.conv2d(x, 32, [6, 6], padding="VALID", activation_fn=tf.nn.sigmoid, scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        net = slim.conv2d(net, 64, [3, 3], padding="VALID", activation_fn=tf.nn.sigmoid, scope='conv2')

        net = slim.conv2d(net, 128, [3, 3], padding="VALID", activation_fn=tf.nn.sigmoid, scope='conv3')

        net = slim.flatten(net, scope='flat')

        no = net

        net = slim.dropout(net, keep_prob=self.rate, scope='drop1')
        net = slim.fully_connected(net, num_outputs=int(n_classes), activation_fn=None, scope='fc1')
        return net, no


