import csv

import numpy as np
import tensorflow as tf


def load_placeholders(cfg):
    """
    1. Placeholder for image.
    2. Placeholder for the question. 
    3. Placeholder for the answer.
    """

    ## 1. 
    input_image = tf.placeholder(tf.float32, shape=(None, 448, 448, 3), name="input_image")

    ## 2. 
    question_placeholder = tf.placeholder(tf.float32, shape=(None, cfg.question_max_words, cfg.embed_size),
                                          name="question_placeholder")

    ## 3. 
    answer_placeholder = tf.placeholder(tf.float32, shape=(None, cfg.answer_max_words, cfg.embed_size),
                                        name="answer_placeholder")

    option1_placeholder = tf.placeholder(tf.float32, shape=(None, cfg.answer_max_words, cfg.embed_size), name="option1")
    option2_placeholder = tf.placeholder(tf.float32, shape=(None, cfg.answer_max_words, cfg.embed_size), name="option2")
    option3_placeholder = tf.placeholder(tf.float32, shape=(None, cfg.answer_max_words, cfg.embed_size), name="option3")
    labels_placeholder = tf.placeholder(tf.float32, shape=(None, 4), name="labels")

    return input_image, question_placeholder, answer_placeholder, option1_placeholder, option2_placeholder, option3_placeholder, labels_placeholder


def load_glove(cfg):
    print(cfg.glove_path)
    W2VEC_LEN = cfg.embed_size
    reader = csv.reader(open(cfg.glove_path.split('../')[1]), delimiter=' ', quoting=csv.QUOTE_NONE)
    W2VEC = {line[0]: np.array(list(map(float, line[1:]))) for line in reader}
    return W2VEC


def load_weights(weight_file, sess, is_Train=True):
    if is_Train:
        print("The weights are trainable")
    else:
        print("The weights are not trainable")

    print("weight_file is: ", weight_file.split('../')[1])
    weights_loaded = np.load(weight_file.split('../')[1])

    keys = sorted(weights_loaded.keys())

    weights = {
        'conv1_1_W': tf.Variable(tf.zeros([3, 3, 3, 64], dtype=tf.float32), name='conv1_1_W', trainable=is_Train),
        'conv1_2_W': tf.Variable(tf.zeros([3, 3, 64, 64], dtype=tf.float32), name='conv1_2_W', trainable=is_Train),

        'conv2_1_W': tf.Variable(tf.zeros([3, 3, 64, 128], dtype=tf.float32), name='conv2_1_W', trainable=is_Train),
        'conv2_2_W': tf.Variable(tf.zeros([3, 3, 128, 128], dtype=tf.float32), name='conv2_2_W', trainable=is_Train),

        'conv3_1_W': tf.Variable(tf.zeros([3, 3, 128, 256], dtype=tf.float32), name='conv3_1_W', trainable=is_Train),
        'conv3_2_W': tf.Variable(tf.zeros([3, 3, 256, 256], dtype=tf.float32), name='conv3_2_W', trainable=is_Train),
        'conv3_3_W': tf.Variable(tf.zeros([3, 3, 256, 256], dtype=tf.float32), name='conv3_3_W', trainable=is_Train),

        'conv4_1_W': tf.Variable(tf.zeros([3, 3, 256, 512], dtype=tf.float32), name='conv4_1_W', trainable=is_Train),
        'conv4_2_W': tf.Variable(tf.zeros([3, 3, 512, 512], dtype=tf.float32), name='conv4_2_W', trainable=is_Train),
        'conv4_3_W': tf.Variable(tf.zeros([3, 3, 512, 512], dtype=tf.float32), name='conv4_3_W', trainable=is_Train),

        'conv5_1_W': tf.Variable(tf.zeros([3, 3, 512, 512], dtype=tf.float32), name='conv5_1_W', trainable=is_Train),
        'conv5_2_W': tf.Variable(tf.zeros([3, 3, 512, 512], dtype=tf.float32), name='conv5_2_W', trainable=is_Train),
        'conv5_3_W': tf.Variable(tf.zeros([3, 3, 512, 512], dtype=tf.float32), name='conv5_3_W', trainable=is_Train),

        'fc6_W': tf.Variable(tf.zeros([25088, 4096], dtype=tf.float32), name='fc6_W', trainable=is_Train),
        'fc7_W': tf.Variable(tf.zeros([4096, 4096], dtype=tf.float32), name='fc7_W', trainable=is_Train),
        'fc8_W': tf.Variable(tf.zeros([4096, 1000], dtype=tf.float32), name='fc8_W', trainable=is_Train),

        'conv1_1_b': tf.Variable(tf.zeros([64, ], dtype=tf.float32), name='conv1_1_b', trainable=is_Train),
        'conv1_2_b': tf.Variable(tf.zeros([64, ], dtype=tf.float32), name='conv1_2_b', trainable=is_Train),

        'conv2_1_b': tf.Variable(tf.zeros([128, ], dtype=tf.float32), name='conv1_2_b', trainable=is_Train),
        'conv2_2_b': tf.Variable(tf.zeros([128, ], dtype=tf.float32), name='conv1_2_b', trainable=is_Train),

        'conv3_1_b': tf.Variable(tf.zeros([256, ], dtype=tf.float32), name='conv1_2_b', trainable=is_Train),
        'conv3_2_b': tf.Variable(tf.zeros([256, ], dtype=tf.float32), name='conv1_2_b', trainable=is_Train),
        'conv3_3_b': tf.Variable(tf.zeros([256, ], dtype=tf.float32), name='conv1_2_b', trainable=is_Train),

        'conv4_1_b': tf.Variable(tf.zeros([512, ], dtype=tf.float32), name='conv1_2_b', trainable=is_Train),
        'conv4_2_b': tf.Variable(tf.zeros([512, ], dtype=tf.float32), name='conv1_2_b', trainable=is_Train),
        'conv4_3_b': tf.Variable(tf.zeros([512, ], dtype=tf.float32), name='conv1_2_b', trainable=is_Train),

        'conv5_1_b': tf.Variable(tf.zeros([512, ], dtype=tf.float32), name='conv1_2_b', trainable=is_Train),
        'conv5_2_b': tf.Variable(tf.zeros([512, ], dtype=tf.float32), name='conv1_2_b', trainable=is_Train),
        'conv5_3_b': tf.Variable(tf.zeros([512, ], dtype=tf.float32), name='conv1_2_b', trainable=is_Train),

        'fc6_b': tf.Variable(tf.zeros([4096, ], dtype=tf.float32), name='fc6_b', trainable=is_Train),
        'fc7_b': tf.Variable(tf.zeros([4096, ], dtype=tf.float32), name='fc7_b', trainable=is_Train),
        'fc8_b': tf.Variable(tf.zeros([1000, ], dtype=tf.float32), name='fc8_b', trainable=is_Train)
    }

    # @TODO: i is never used remove this
    # @TODO: there's something fishy about this for loop gut feeling it can be improved
    for i, k in enumerate(keys):
        sess.run(weights[k].assign(weights_loaded[k]))

    return keys, weights


class encodeImage(object):
    """
    load_weights: Loads the weights for the pre trained models
    conv2d: Convolution operation
    conv2d_transpose: Reverse convolution operation. Deconvolution. 
    maxpool2d: Maxpooling operation
    """

    def __init__(self, config):
        self.cfg = config

    def load_weights(self, weight_file, sess, is_Train=False):
        keys, weights = load_weights(self.cfg.weights_path, sess, is_Train)
        self.weights = weights
        self.keys = keys

    def conv2d(self, x, W, b, strides=1):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def conv2d_transpose(self, x, W, b, output_sh, strides=1):
        x = tf.nn.conv2d_transpose(x, W, output_shape=output_sh, strides=[1, strides, strides, 1], padding='VALID')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(self, x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')

    def forward_pass(self, x):
        """
        conv5_3 : (?, 8, 8, 512)
        Considering only the final convolution layer. - Flattened later for classification. 
        
        """
        conv1_1 = self.conv2d(x, self.weights['conv1_1_W'], self.weights['conv1_1_b'], 1)
        conv1_2 = self.conv2d(conv1_1, self.weights['conv1_2_W'], self.weights['conv1_2_b'], 1)
        conv1_2 = self.maxpool2d(conv1_2, 2)

        conv2_1 = self.conv2d(conv1_2, self.weights['conv2_1_W'], self.weights['conv2_1_b'], 1)
        conv2_2 = self.conv2d(conv2_1, self.weights['conv2_2_W'], self.weights['conv2_2_b'], 1)
        conv2_2 = self.maxpool2d(conv2_1, 2)

        conv3_1 = self.conv2d(conv2_2, self.weights['conv3_1_W'], self.weights['conv3_1_b'], 1)
        conv3_2 = self.conv2d(conv3_1, self.weights['conv3_2_W'], self.weights['conv3_2_b'], 1)
        conv3_3 = self.conv2d(conv3_2, self.weights['conv3_3_W'], self.weights['conv3_3_b'], 1)
        conv3_3 = self.maxpool2d(conv3_3, 2)

        conv4_1 = self.conv2d(conv3_3, self.weights['conv4_1_W'], self.weights['conv4_1_b'], 1)
        conv4_2 = self.conv2d(conv4_1, self.weights['conv4_2_W'], self.weights['conv4_2_b'], 1)
        conv4_3 = self.conv2d(conv4_2, self.weights['conv4_3_W'], self.weights['conv4_3_b'], 1)
        conv4_3 = self.maxpool2d(conv4_3, 2)

        conv5_1 = self.conv2d(conv4_3, self.weights['conv5_1_W'], self.weights['conv5_1_b'], 1)
        conv5_2 = self.conv2d(conv5_1, self.weights['conv5_2_W'], self.weights['conv5_2_b'], 1)
        conv5_3 = self.conv2d(conv5_2, self.weights['conv5_3_W'], self.weights['conv5_3_b'], 1)
        conv5_3 = self.maxpool2d(conv5_3, 2)

        return conv5_3


class encodeText(object):
    """
    encodeText: to encode a question
    """

    # @TODO: 100 and then 50, typo maybe
    def __init__(self, config, state_size=512, embed_size=100):
        self.state_size = 512
        self.embed_size = 50
        self.cfg = config

    def encode(self, inputs, encoder_input=None, dropout=1.0):
        """
        inputs: (?, T, D) - Dimension to give as input to the tf.nn.dynamic_rnn
            - T: Maximum sequence length
            - D: Embedding size
        """
        lstm_fw_cell = tf.contrib.rnn.GRUCell(self.state_size)
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell)

        lstm_fw_initial_state = encoder_input

        # initial_state = lstm_fw_cell.
        output_fw, final_state_fw = tf.nn.dynamic_rnn(lstm_fw_cell, \
                                                      inputs, \
                                                      initial_state=lstm_fw_initial_state,
                                                      dtype=tf.float32 \
                                                      )

        return output_fw, final_state_fw

    def declareAttentionNetworkVariables(self):
        # subscripts v: word token; h: output of lstm cell; r: attention of lstm cell;

        # learnable weights and bias for 'i' gate
        W_vi = tf.Variable(tf.truncated_normal([self.embed_size, self.cfg.batch_size], dtype=tf.float32),
                           name='W_{vi}', trainable=is_Train);
        W_hi = tf.Variable(tf.truncated_normal([self.cfg.batch_size, self.cfg.batch_size], dtype=tf.float32),
                           name='W_{hi}', trainable=is_Train);
        W_ri = tf.Variable(tf.truncated_normal([self.cfg.batch_size, self.cfg.batch_size], dtype=tf.float32),
                           name='W_{ri}', trainable=is_Train);
        b_i = tf.Variable(tf.truncated_normal([self.cfg.batch_size, self.state_size], dtype=tf.float32), name='b_{i}',
                          trainable=is_Train);

        # learnable weights and bias for 'f' gate
        W_vf = tf.Variable(tf.truncated_normal([self.embed_size, self.cfg.batch_size], dtype=tf.float32),
                           name='W_{vf}', trainable=is_Train);
        W_hf = tf.Variable(tf.truncated_normal([self.cfg.batch_size, self.cfg.batch_size], dtype=tf.float32),
                           name='W_{hf}', trainable=is_Train);
        W_rf = tf.Variable(tf.truncated_normal([self.cfg.batch_size, self.cfg.batch_size], dtype=tf.float32),
                           name='W_{rf}', trainable=is_Train);
        b_f = tf.Variable(tf.truncated_normal([self.cfg.batch_size, self.state_size], dtype=tf.float32), name='b_{f}',
                          trainable=is_Train);

        # learnable weights and bias for 'o' gate
        W_vo = tf.Variable(tf.truncated_normal([self.embed_size, self.cfg.batch_size], dtype=tf.float32),
                           name='W_{vo}', trainable=is_Train);
        W_ho = tf.Variable(tf.truncated_normal([self.cfg.batch_size, self.cfg.batch_size], dtype=tf.float32),
                           name='W_{ho}', trainable=is_Train);
        W_ro = tf.Variable(tf.truncated_normal([self.cfg.batch_size, self.cfg.batch_size], dtype=tf.float32),
                           name='W_{ro}', trainable=is_Train);
        b_o = tf.Variable(tf.truncated_normal([self.cfg.batch_size, self.state_size], dtype=tf.float32), name='b_{o}',
                          trainable=is_Train);

        # learnable weights and bias for 'g' gate
        W_vg = tf.Variable(tf.truncated_normal([self.embed_size, self.cfg.batch_size], dtype=tf.float32),
                           name='W_{vg}', trainable=is_Train);
        W_hg = tf.Variable(tf.truncated_normal([self.cfg.batch_size, self.cfg.batch_size], dtype=tf.float32),
                           name='W_{hg}', trainable=is_Train);
        W_rg = tf.Variable(tf.truncated_normal([self.cfg.batch_size, self.cfg.batch_size], dtype=tf.float32),
                           name='W_{rg}', trainable=is_Train);
        b_g = tf.Variable(tf.truncated_normal([self.cfg.batch_size, self.state_size], dtype=tf.float32), name='b_{g}',
                          trainable=is_Train);

        # i am not sure of dimension of following:
        W_a = tf.Variable(tf.truncated_normal([None, None], dtype=tf.float32),
                          name='W_{a}', trainable=is_Train);
        W_he = tf.Variable(tf.truncated_normal([None, None], dtype=tf.float32),
                           name='W_{he}', trainable=is_Train);
        W_ce = tf.Variable(tf.truncated_normal([None, None], dtype=tf.float32),
                           name='W_{ce}', trainable=is_Train);
        b_a = tf.Variable(tf.truncated_normal([self.cfg.batch_size, self.state_size], dtype=tf.float32),
                          name='b_{a}', trainable=is_Train);

        # Initialize this with 196 dimensional vector of fc4 layer of VGG-16 model.
        self.C_I = None

    def runAttentionLSTMNetworek(self, v):
        sigmoid = tf.python.ops.math_ops.sigmoid
        tanh = tf.python.ops.math_ops.tanh

        e = tf.matmul(W_a, tanh(tf.matmul(W_he, h_prev) + tf.matmul(W_ce, C_I)), transpose_a=False) + b_a
        a = softmax(e)
        r = tf.matmul(a, C_I, transpose_a=False)

        # original equations had v * W_ri we interchanged for matrix dimension match
        i = sigmoid(tf.matmul(v, W_vi) + tf.matmul(W_hi, h_prev) + tf.matmul(W_vi, r) + b_i)
        f = sigmoid(tf.matmul(v, W_vf) + tf.matmul(W_hf, h_prev) + tf.matmul(W_vf, r) + b_f)
        o = sigmoid(tf.matmul(v, W_vo) + tf.matmul(W_ho, h_prev) + tf.matmul(W_vo, r) + b_o)
        g = tanh(tf.matmul(W_vg, v) + tf.matmul(W_hg, h_prev) + tf.matmul(W_vg, r) + b_g)
        c = f * c_prev + i * g;
        h = o * tanh(c);

        # @TODO: all of this in a loop and remember to set h_prev = h and c_prev = c

    def encode_with_attentions(self, inputs, encoder_input=None, dropout=1.0):
        """
        Use this method rather than the method encode when you want to modify the LSTM cell than the 
        default LSTM Cell.
        
        inputs: (?, T, D)
            - T: Maximum sequence length
            - D: Embedding size
        """
        lstm_fw_cell = tf.contrib.rnn.LSTMCell(self.state_size)
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell)

        lstm_fw_initial_state = encoder_input

        # @TODO: Use a for loop to construct the dynamic RNN for the attention to be included.


class fullyConnected(object):

    def __init__(self, config):
        self.wfc1 = tf.Variable(tf.truncated_normal(shape=(config.testSize, config.state_size)), dtype=tf.float32)
        self.bfc1 = tf.Variable(tf.zeros(config.state_size), dtype=tf.float32)

    def forward_pass(self, inputs):
        output = tf.add(tf.matmul(inputs, self.wfc1), self.bfc1)
        output_activated = tf.nn.relu(output)
        return output_activated
