from __future__ import print_function

import tensorflow as tf
import numpy as np
from modules import SelfAttention

batch_size = 2
max_len = 5
dim = 3
hidden_size = 4

self_attention_input = tf.reshape(tf.range(batch_size * max_len * dim, dtype=tf.float32), (batch_size, max_len, dim)) * 100.0

self_attention_layer = SelfAttention(1.0, dim, batch_size, hidden_size)
self_atten_output = self_attention_layer.build_graph(self_attention_input)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    W1values = self_attention_layer.W1values.eval()
    W2values = self_attention_layer.W2values.eval()
    values_expand = self_attention_layer.values_expand.eval()
    e = self_attention_layer.e.eval()

    test = np.dot(np.tanh(values_expand), self_attention_layer.V.eval())

    diff = np.abs(test[:,:,:,0] - e)


    out = self_atten_output.eval()

print(out)
