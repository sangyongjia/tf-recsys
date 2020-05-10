import tensorflow as tf

from ..layers.utils import add_func
from ..inputs import input_from_feature_columns
from ..inputs import get_linear_logit
from ..inputs import build_input_features
from ..inputs import combined_dnn_input

from ..layers.core import PredictionLayer
from ..layers.core import DNN

def FNN(linear_feature_columns, dnn_features_columns, dnn_hidden_units=(128,128),
        l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_dnn=0, init_std=0.001,
        seed=2020, dnn_dropout=0, dnn_activation="relu", task="binary"):

    features = build_input_features(linear_feature_columns, dnn_features_columns)
    inputs_list = list(features.values())
    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_features_columns, l2_reg_embedding, init_std, seed)

    linear_logit = get_linear_logit(features, linear_feature_columns, init_std=init_std, seed=seed, prefix="linear", l2_reg=l2_reg_linear)

    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    deep_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed)(dnn_input)
    dnn_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(deep_out)

    final_logit = add_func([dnn_logit, linear_logit])

    output = PredictionLayer(task)(final_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model

