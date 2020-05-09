import tensorflow as tf

from ..inputs import input_from_feature_columns, get_linear_logit, build_input_features, combined_dnn_input
from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import CIN
from ..layers.utils import concat_func, add_func


def xDeepFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256,256), cin_layer_size=(128,128),
            cin_split_half=True, cin_activation="relu", l2_reg_linear=1e-5, l2_reg_embedding=1e-5,
            l2_reg_dnn=0, l2_reg_cin=0, init_std=1e-4, seed=2020, dnn_dropout=0, dnn_activation="relu",
            dnn_use_bn=False, task="binary"):
    features = build_input_features(linear_feature_columns + dnn_feature_columns)
    inputs_list = list(features.values())
    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding, init_std, seed)
    get_linear_logit = get_linear_logit(features, linear_feature_columns, init_std=init_std,
                                        seed=seed, prefix="linear", l2_reg=l2_reg_linear)
    fm_input = concat_func(sparse_embedding_list, axis=1)

    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed)(dnn_input)
    dnn_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(dnn_output)
    final_logit = add_func([linear_logit, dnn_logit])

    if len(cin_layer_size) > 0:
        exFM_out = CIN(cin_layer_size, cin_activation, cin_split_half, l2_reg_cin, seed)(fm_input)
        exFM_logit = tf.keras.layers.Dense(1, activation=None)(exFM_out)
        final_logit = add_func([final_logit, exFM_logit])
    output = PredictionLayer(task)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)

    return model
