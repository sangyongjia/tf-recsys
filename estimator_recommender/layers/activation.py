import sys
import tensorflow as tf
from tensorflow.keras.layers import Layer


def activation_layer(activation):
    if activation == "dice" or activation =="Dice":
        #act_layer = Dice()
        pass
    elif (isinstance(activation,str)) or (sys.version_info.major == 2 and isinstance(activation, (str, unicode))):
        act_layer = tf.keras.layers.Activation(activation)
        print("in")
    elif issubclass(activation, Layer):
        print("IN?")
        act_layer = activation()
    else:
        raise ValueError("Invalid activation,found %s.You should use a str or a Activation Layer Class." % (activation))
    return act_layer
