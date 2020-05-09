from tensorflow.keras.models import Model
from ..inputs import build_input_features
from ..inputs import input_from_feature_columns
from ..inputs import get_linear_logit
from ..inputs import combined_dnn_input

#from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from ..layers.core import PredictionLayer
from ..layers.core import DNN
from ..layers.utils import add_func


def WDL(line_feature_columns, dnn_feature_columns, dnn_hidden_units=(128,128), l2_reg_linear=1e-5,l2_reg_embedding=1e-5,
        l2_reg_dnn=0, init_std=0.0001, seed=2020, dnn_dropout=0, dnn_activation='relu', task='binary'):

        features  = build_input_features(line_feature_columns + dnn_feature_columns)
        inputs_list = list(features.values())
        sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,l2_reg_embedding, init_std, seed)

        linear_logit = get_linear_logit(features, line_feature_columns, init_std=init_std, seed=seed, prefix="linear", l2_reg=l2_reg_linear)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed)(dnn_input)

        dnn_logit = Dense(1, use_bias=False, activation=None)(dnn_out)
        
        final_logit = add_func([dnn_logit, linear_logit])

        output = PredictionLayer(task)(final_logit)

        model = Model(inputs=inputs_list, outputs=output)

        return model