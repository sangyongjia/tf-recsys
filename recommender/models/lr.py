from tensorflow.keras.models import Model
from ..inputs import build_input_features
from ..inputs import input_from_feature_columns
from ..inputs import get_linear_logit
from ..layers.core import PredictionLayer
from ..layers.utils import add_func


def LR(line_feature_columns, l2_reg_linear=1e-5,l2_reg_embedding=1e-5,
        init_std=0.0001, seed=2020, task='binary'):

        features  = build_input_features(line_feature_columns)
        inputs_list = list(features.values())
        #sparse_embedding_list, dense_value_list = input_from_feature_columns(features,None,l2_reg_embedding, init_std, seed)

        linear_logit = get_linear_logit(features, line_feature_columns, init_std=init_std, seed=seed, prefix="linear", l2_reg=l2_reg_linear)
        
        final_logit = add_func([linear_logit])

        output = PredictionLayer(task)(final_logit)

        model = Model(inputs=inputs_list, outputs=output)

        return model
