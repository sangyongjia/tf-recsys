import tensorflow as tf 
from ..inputs import input_from_feature_columns, get_linear_logit, build_input_features, combined_dnn_input
from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import BiInteractionPooling
from ..layers.utils import concat_func, add_func

def NFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(128,128),
        l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_dnn=0, init_std=0.0001,
        seed=2020, bi_dropput=0, dnn_dropout=0, dnn_activation="relu", task="binary"):

    features = build_input_features(linear_feature_columns + dnn_feature_columns)
    inputs_list = list(features.values())
    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, init_std, seed)

    linear_logit = get_linear_logit(features, linear_feature_columns, init_std=init_std, 
                                    seed=seed, prefix='linear',l2_reg=l2_reg_linear)

    fm_input = concat_func(sparse_embedding_list, axis=1)
    bi_out = BiInteractionPooling()(fm_input)
    if bi_dropout:
        bi_out = tf.keras.layers.Dropout(bi_dropout)(bi_out, training=None) #这个位置的training参数应该改为True
        
    dnn_input = combined_dnn_input([bi_out], dense_value_list)
    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,False, seed)(dnn_input)
    dnn_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(dnn_output)

    final_logit = add_func([linear_logit, dnn_logit])
    output = PredictionLayer(task)(final_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)

    return model