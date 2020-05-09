import tensorflow as tf

from ..inputs import input_from_feature_columns, build_input_features, combined_dnn_input, get_linear_logit
from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import InteractingLayer
from ..layers.utils import concat_func, add_func

def AutoInt(linear_feature_columns, dnn_feature_columns, att_layer_num=3, att_embedding_size=8, att_head_num=2,
            att_res=True, dnn_hidden_units=(256,256), dnn_activation="relu", l2_reg_linear=1e-5, l2_reg_embedding=1e-5,
            l2_reg_dnn=0, dnn_use_bn=False, dnn_dropout=0, init_std=1e-4, seed=2020, task="binary"):
    if len(dnn_hidden_units) <= 0 and att_layer_num <= 0:
        raise ValueError("Either hidden_layer or att_layer_num must > 0")
    features = build_input_features(dnn_feature_columns) ##只有dnn部分的？搞错了吧
    inputs_list = list(features.values())
    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, 
                                                                        l2_reg_embedding, init_std, seed)
    linear_logit = get_linear_logit(features, linear_feature_columns, init_std=init_std, 
                                    seed=seed, prefix="linear", l2_reg=l2_reg_linear)
    att_input = concat_func(sparse_embedding_list, axis=1)

    for _ in range(att_layer_num):
        att_input = InteractionLayer(att_embedding_size, att_head_num, att_res)(att_input)
    att_output = tf.keras.layers.Flatten()(att_input)
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

    if len(dnn_hidden_units) > 0 and att_layer_num > 0:
        deep_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed)(dnn_input)
        stack_out = tf.keras.layers.Concatenate()([att_output, deep_out])
        final_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(stack_out)
    elif len(dnn_hidden_units) > 0:
        deep_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed)(dnn_input)
        final_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(deep_out)
    elif att_layer_num > 0:
        final_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(att_output)
    else:  # Error
        raise NotImplementedError

    final_logit = add_func([final_logit, linear_logit])
    output = PredictionLayer(task)(final_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    