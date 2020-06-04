import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.initializers import Zeros
from tensorflow.keras.regularizers import l2
#from .activation import activation_layer

class DNN(Layer):
    def __init__(self, hidden_units, activation="relu", l2_reg=0, dropout_rate=0,  use_bn=False, seed=2020, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        self.kernels = [self.add_weight(name="kernal"+str(i),
                                        shape=(hidden_units[i],hidden_units[i+1]),
                                        initializer=glorot_normal(seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name="bias"+str(i),
                                     shape=(self.hidden_units[i],),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in range(len(self.hidden_units))]
        self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed+i) for i in range(len(self.hidden_units))]
        self.activation_layers = [self.activation for _ in range(len(self.hidden_units))]

        super(DNN, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        deep_input = inputs
        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(deep_input, self.kernels[i], axes=(-1,0)), self.bias[i])
            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)
            fc = self.activation_layers[i](fc)
            fc = self.dropout_layers[i](fc, training=training)
            deep_input = fc
        return deep_input

    def compute_output_shape(self, input_shape): #TODO 搞明白这个函数的作用
        pass
    def get_config(self,):#TODO 搞明白这个函数的作用
        pass


class PredictionLayer(Layer):
    def __init__(self, task="binary", use_bias=True, **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")
        self.task = task
        self.use_bias = use_bias
        super(PredictionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.use_bias:
            self.global_bias = self.add_weight(shape=(1,), initializer=Zeros(), name="global_bias")
        super(PredictionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        if self.use_bias:
            x = tf.nn.bias_add(x, self.global_bias, data_format="NHWC")
        if self.task == "binary":
            x = tf.sigmoid(x)
        output = tf.reshape(x, (-1, 1))  # reshape中的-1的意思是，本维度不定，第二维设定为1；根据总数据和第二维为1来推断“-1”位置的维度。
        return output

    def compute_output_shape(self, input_shape):  #TODO 搞明白这个函数的作用
        return (None, 1)

    def get_config(self, ):  #TODO 搞明白这个函数的作用
        config = {'task': self.task, 'use_bias': self.use_bias}
        base_config = super(PredictionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))