import tensorflow as tf 

from ..layers.utils import add_func
from ..inputs import input_from_feature_columns
from ..inputs import build_input_features
from ..inputs import combined_dnn_input
from ..inputs import get_linear_logit

from ..layers.core import DNN
from ..layers.core import PredictionLayer
from ..layers.interaction import CrossNet

def DCN(linear_feature_columns, dnn_feature_columns, cross_num=2, dnn_hidden_units=(128,128),  l2_reg_linear=1e-5,
        l2_reg_embedding=1e-5, l2_reg_cross=1e-5, l2_reg_dnn=0, init_std=0.0001, seed=2020, dnn_dropout=0, dnn_use_bn=False,
        dnn_activation="relu", task="binary"):
    if len(dnn_hidden_units) == 0 and cross_num == 0:
        raise ValueError("Either hidden_layer or cross layer must > 0")
    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding, init_std, seed)
    linear_logit = get_linear_logit(features, linear_feature_columns, init_std=init_std, seed=seed, prefix="linear", l2_reg=l2_reg_linear)
    
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    if len(dnn_hidden_units)> and cross_num > 0:
        deep_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed)(dnn_input)
        cross_out = CrossNet(cross_num, l2_reg=l2_reg_cross,)(dnn_input)
        stack_out = tf.keras.layers.Concatenate()([cross_out, deep_out])
        final_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(stack_out)
    elif len(dnn_hidden_units) > 0:
        deep_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed)(dnn_input)
        final_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(deep_out)
    elif cross_num > 0:
        cross_out = CrossNet(cross_num, l2_reg=l2_reg_cross)(dnn_input)
        final_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(cross_out)
    else:
        raise NotImplementedError

    final_logit = add_func([final_logit, linear_logit])
    output = PredictionLayer(task)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)

    return model
