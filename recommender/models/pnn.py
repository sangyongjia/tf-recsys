import tensorflow as tf 
from ..inputs import input_from_feature_columns
from ..inputs import build_input_features
from ..inputs import combined_dnn_input

from ..layers.core import PredictionLayer
from ..layers.core import DNN

from ..layers.interaction import InnerProductLayer
from ..layers.interaction import OutterProductLayer
from ..layers.utils import concat_func





def PNN(dnn_feature_columns, embedding_size=8, dnn_hidden_units=(128,128), l2_reg_embedding=1e-5, l2_reg_dnn=0,
        init_std=1e-4, seed=2020, dnn_dropout=0, dnn_activation="relu", use_inner=True, use_outter=False,
        kernel_type="mat", task="binary"):
    if kernel_type not in ['mat', 'vec', 'num']:
        raise ValueError("kernel_type must be mat,vec or num")
    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())

    sparse_embedding_list , dense_value_list = input_from_feature_columns(features, dnn_feature_columns,l2_reg_embedding, init_std, seed)

    inner_product = tf.keras.layers.Flatten()(InnerProductLayer()(sparse_embedding_list))
    outter_product = tf.keras.layers.Flatten()(OutterProductLayer()(sparse_embedding_list))

    linear_signal = tf.keras.layers.Reshape([len(sparse_embedding_list) * embedding_size])(concat_func(sparse_embedding_list))

    if use_inner and use_outter:
        deep_input = tf.keras.layers.Concatenate()([linear_signal, inner_product, outter_product])
    elif use_inner:
        deep_input = tf.keras.layers.Concatenate()([linear_signal, inner_product])
    elif use_outter:
        deep_input = tf.keras.layers.Concatenate()([linear_signal, outter_product])
    else:
        deep_input = linear_signal

    dnn_input = combined_dnn_input([deep_input], dense_value_list)
    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed)(dnn_input)
    dnn_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(dnn_out)
    output = PredictionLayer(task)(dnn_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)

    return model 