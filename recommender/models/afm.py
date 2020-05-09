import tensorflow as tf 

from ..inputs import input_from_feature_columns
from ..inputs import get_linear_logit
from ..inputs import build_input_features
from ..inputs import DEFAULT_GROUP_NAME

from ..layers.core import PredictionLayer
from ..layers.interaction import AFMLayer
from ..layers.interaction import FM

from ..layers.utils import concat_func, add_func


def AFM(linear_feature_columns, dnn_feature_columns, fm_group=DEFAULT_GROUP_NAME, use_attention=True,
        attention_factor=8, l2_reg_linear=1e-5, l2_reg_embedding=1e-5, l2_reg_att=1e-5, afm_dropout=0,
        init_std=0.0001, seed=1024, task="binary"):
    features = build_input_features(linear_feature_columns + dnn_feature_columns)
    inputs_list = list(features.values())
    group_embedding_dict, _ = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding, 
                                                        init_std, seed, support_dense=False, support_group=True)
    linear_logit = get_linear_logit(features, linear_feature_columns, init_std=init_std, seed=seed, prefix="linear", l2_reg=l2_reg_linear)
    if use_attention:
        fm_logit = add_func([AFMLayer(attention_factor, l2_reg_att, afm_dropout, 
                                    seed)(list(v)) for k, v in group_embedding_dict.items() if k in fm_group)])
    else:
        fm_logit = add_func([FM()(concat_func(v, axis=1)) for k, v in group_embedding_dict.items() if k in fm_group])

    final_logit = add_func([linear_logit, fm_logit])
    output = PredictionLayer(task)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
