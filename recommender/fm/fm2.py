import tensorflow as tf
import numpy as np

class FM(object):
    def __init__(self, hyper_params, df_i, df_v):
        self.hyper_params = hyper_params
        tf.set_random_seed(self.hyper_params.seed)
        self.line_res = self.line_section(df_i, df_v)
        self.fm_res = self.fm_section(df_i, df_v)
        self.logits = self.line_res + self.fm_res #(batch_size, 1)

    def line_section(self, df_i, df_v):
        with tf.variable_scope('line'):
            weights = tf.get_variable('weights', 
                                        shape=[self.hyper_params.feature_nums,1],
                                        dtype=tf.float32,
                                        initializer=tf.initializers.glorot_uniform()) ##(feature_size, 1)
            batch_weights = tf.nn.embedding_lookup(weights, df_i)                     ##(batch_size, field_size, 1)
            batch_weights = tf.squeeze(batch_weights, axis=2)   ##删除维度是1的维度      ##(batch_size, field_size)
            line_res = tf.multiply(df_v, batch_weights, name='wixi')            ##(batch_size, field_size) multiply (batch_size, field_size) = (batch_size, field_size)

            biase = tf.get_variable('biase',
                                    shape=[1,1],
                                    dtype=tf.float32,
                                    initializer=tf.initializers.zeros())                ##(1,1)
            
            line_res = tf.add(tf.reduce_sum(line_res, axis=1),biase)     ##(batch_num,1)

            return line_res

            
    def fm_section(self, df_i, df_v):
        with tf.variable_scope('fm'):
            embedding = tf.get_variable('embedding',
                                        shape=[self.hyper_params.feature_nums,
                                               self.hyper_params.embedding_size],
                                        dtype=tf.float32,
                                        initializer=tf.initializers.random_normal()) ##(feature_size,facrtor_size)
            batch_embedding = tf.nn.embedding_lookup(embedding, df_i)                ##(batch_size, field_size, factor_size)
            df_v = tf.expand_dims(df_v, axis=2)                                      ##(batch_size, field_size) expand_dims (batch, field_size, 1)
            self.xv = tf.multiply(df_v, batch_embedding)                             ##(batch_size, field_size, factor_size),元素值是vif*xi
            sum_square = tf.square(tf.reduce_sum(self.xv, axis=1))                   ##(batch_size, factor_size）先sum再square
            square_sum = tf.reduce_sum(tf.square(self.xv), axis=1)                   ##(batch_size, factor_size) 先square再sum
            substract = 0.5 * (tf.subtract(sum_square, square_sum))                  ##
            fm_res = tf.reduce_sum(substract, axis=1)                                ##(batch_size, 1)

            return fm_res


