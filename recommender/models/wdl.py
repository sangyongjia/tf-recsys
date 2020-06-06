import tensorflow as tf
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

        features  = build_input_features(line_feature_columns + dnn_feature_columns) #创建所有输入特征的Input；字典形式，字典的key是feature_name
        inputs_list = list(features.values()) #d所有的Input
        #Sparse：Input->Embedding后的list结果，Dense：Input的list；注：只有DNN部分的
        sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,l2_reg_embedding, init_std, seed)
        #从Input开始，现将sparse特征的embedding_dim修改为1，然后对对sparse特征：Input->Embedding->Concatenate->Flatten得到sparse部分的输入（不需要乘以weights），直接reducesum后得到值linear_logit1
        #直接将dense特征经过Input->Concatenate->Flatte后得到 dense部分的输入，和weights和bias进行相应的操作后，得到linear_logit2； linear_logit1和linear_logit2的和即为Linear部分的最终输出。
        linear_logit = get_linear_logit(features, line_feature_columns, init_std=init_std, seed=seed, prefix="linear", l2_reg=l2_reg_linear)

        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        ##WDL图右侧部分的DNN部分到最后sigmod部分的输入。
        dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed)(dnn_input)

        dnn_logit = Dense(1, use_bias=False, activation=None)(dnn_out)
        
        final_logit = add_func([dnn_logit, linear_logit])

        output = PredictionLayer(task)(final_logit)
        #output = tf.squeeze(PredictionLayer(task)(final_logit))
        print("output***:",tf.shape(output))

        model = Model(inputs=inputs_list, outputs=output)

        return model