import tensorflow as tf 
from ..inputs import DEFAULT_GROUP_NAME
from ..inputs import build_input_features
from ..inputs import input_from_feature_columns
from ..inputs import get_linear_logit

from ..layers.utils import add_func
from ..layers.utils import concat_func
from ..layers.interaction import FM
from ..layers.core import PredictionLayer


def FM(linear_feature_columns, dnn_feature_columns, fm_group=[DEFAULT_GROUP_NAME],l2_reg_linear=0.00001, 
            l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=2020, task="binary"):

    features = build_input_features(linear_feature_columns + dnn_feature_columns)
    inputs_list = list(features.values())

    group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,
                                                                        init_std, seed, support_group=True)
    
    linear_logit = get_linear_logit(features, linear_feature_columns, init_std=init_std,
                                    seed=seed, prefix="linear",l2_reg=l2_reg_linear)

    fm_logit = add_func([FM()(concat_func(v, axis=1)) for k, v in group_embedding_dict.items() if k in fm_group])
    final_logit = add_func([linear_logit, fm_logit])

    output = PredictionLayer(task)(final_logit)
    model = tf.keras.models.Model(inputs_list, outputs=output)
    return model
