import tensorflow as tf
from layers.core import DNN
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomNormal

from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.initializers import Zeros

import yaml
config=tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.03
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


tf.logging.set_verbosity(tf.logging.INFO)

column_names=['userid','item','category','buy_flag']  #decided by data in order
label_name="buy_flag" #also decide by data

def input_fn(file_path,epochs,batch_size=1000,columns_name=['userid','item','category','buy_flag'],label_name="buy_flag"):
    dataset = tf.data.experimental.make_csv_dataset(file_path,
                                                    batch_size=batch_size,
                                                    column_names=column_names,
                                                    label_name=label_name,
                                                    na_value="?",
                                                    num_epochs=epochs)
    dataset = dataset.shuffle(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def load_conf(filename):
    with open("./conf/"+filename, 'r') as f:
        return yaml.load(f)

def build_model_columns():
    feature_conf = load_conf("features.yaml")
    wide_part = []
    deep_part = []
    #print(feature_conf)
    for feature_name, conf in feature_conf.items():
        f_type, f_trans, f_param = conf['type'], conf['transform'],  conf['parameter']
        if f_type == "category":
            hash_bucket_size = f_param["hash_bucket_size"]
            embed_dim = f_param["embed_size"]
            col = tf.feature_column.categorical_column_with_hash_bucket(feature_name, hash_bucket_size, dtype=tf.int64)

            wide_part.append(tf.feature_column.embedding_column(col,
                                                                dimension=1,
                                                                combiner="sum",
                                                                initializer=RandomNormal(mean=0.0, stddev=0.0001, seed=2020),
                                                                ckpt_to_load_from=None,
                                                                tensor_name_in_ckpt=None,
                                                                max_norm=None,
                                                                trainable=True))
            deep_part.append(tf.feature_column.embedding_column(col,
                                                                dimension=embed_dim,
                                                                combiner="sum",
                                                                initializer=None,
                                                                ckpt_to_load_from=None,
                                                                tensor_name_in_ckpt=None,
                                                                max_norm=None,
                                                                trainable=True))

        else:
            normalization, boundaries = f_param["normalization"], f_param["boundaries"]
            if f_trans is None:
                normalizer_fn = None
            else:
                normalizer_fn = normalizer_fn_builder(f_trans, tuple(normalization))#TODO:analyze
            col = tf.feature_column.numeric_column(feature_name,
                                                   shape=(1,),
                                                   default_value=0,
                                                   dtype=tf.float32,
                                                   normalizer_fn=normalizer_fn)
            if boundaries: #TODO:  现在这部分的理解是有问题的
                wide_part.append(tf.feature_column.bucketized_column(col,boundaries=boundaries))
                wide_dim  += (len(boundaries)+1)
            deep_part.append(col)
    tf.logging.info("Build total {} wide columns".format(len(wide_part)))
    for col in wide_part:
        tf.logging.info("wide part:{}".format(col))
    tf.logging.info("Build total {} deep columns".format(len(deep_part)))
    for col in deep_part:
        tf.logging.info("deep part:{}".format(col))
    #tf.logging.info
    return wide_part,deep_part

def model(wide_part, deep_part):
    train_conf = load_conf("train.yaml")["train"]
    model_name = train_conf["model_name"]

    model_conf = load_conf("model.yaml")
    activation = model_conf["activation"]
    initializer = model_conf["initializer"]
    l2_reg_dnn = model_conf["l2_reg_dnn"]
    dnn_dropout_rate = model_conf["dnn_dropout_rate"]
    seed = model_conf["seed"]

    #features_conf = load_conf(features.yaml)
    #embeding_dim = features_conf[""]
    embedding_dim = 4 ##TODO 这个embedding_dim如何读进来？
    features_num = 3

    if initializer == "glorot_normal":#TODO 修改 initiliazer,现在并未使用到initializer
        initializer_fn = tf.keras.initializers.glorot_normal()
    else:
        raise TypeError("initializer_fn {} is not implement".format(initializer))

    if activation == "relu":
        activation_fn = tf.nn.relu
    elif activation == "selu":
        activation_fn = tf.nn.selu
    else:
        raise TypeError("activation_fn {} is not implement".format(activation))

    if model_name == "WDL":
        dnn_hidden_units =model_conf["dnn_hidden_units"]
        dnn_output = DNN(dnn_hidden_units, activation_fn, l2_reg_dnn, dnn_dropout_rate, False, seed)(deep_part)
        dnn_logit = Dense(1, use_bias=False, activation=None)(dnn_output)
        model_output = dnn_logit + tf.reduce_sum(wide_part, axis=1, keep_dims=True)
    elif model_name == "LR":
        linear_part_l2 = tf.contrib.layers.l2_regularizer(1e-5)(wide_part)
        #print("wide_part_l2",wide_part_l2)
        model_output = tf.reduce_sum(wide_part, axis=1, keep_dims=True)
        #print("model_output",model_output)
        regularizer = linear_part_l2

    elif model_name == "FM":

        #linear_part_l2 = tf.contrib.layers.l2_regularizer(1e-5)(wide_part)
        linear_part = tf.reduce_sum(wide_part, axis=1, keep_dims=True)


        print(deep_part)
        feature_matrix = tf.reshape(deep_part,(-1,features_num,embedding_dim))
        print("feature_matrix:",feature_matrix)

        #feature_matrix = tf.keras.layers.Concatenate(axis=1)([userid_fm, categoryid_fm, itemid_fm])
        sum_square = tf.square(tf.reduce_sum(feature_matrix, axis=1, keepdims=True))
        print("sum_square:", sum_square)
        square_sum = tf.reduce_sum(tf.square(feature_matrix), axis=1, keepdims=True)
        print("square_sum:", square_sum)
        cross_part = 0.5 * tf.reduce_sum(sum_square - square_sum, axis=2)

        output = tf.sigmoid(linear_part + cross_part)
        regularizer = None
    return model_output, regularizer


def model_fn(features, labels, mode, params, config):
    global_step = tf.train.get_or_create_global_step() 
    wide_part = tf.feature_column.input_layer(features,params["wide_part"])
    deep_part = tf.feature_column.input_layer(features,params["deep_part"])
    print('wide_part_features: {}'.format(wide_part.get_shape()))
    print('deep_part_features: {}'.format(deep_part.get_shape()))
    logits, regularizer = model(wide_part, deep_part,) #TODO, add is_training or

    predictions = tf.nn.sigmoid(logits)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={"output":predictions})
    #cross_entropy = tf.losses.sigmoid_cross_entropy()
    cross_entropy = tf.add(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels,tf.float32),logits=tf.squeeze(predictions))),regularizer)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions, name="acc")
    auc = tf.metrics.auc(labels=labels, predictions=predictions, name="auc")
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode,predictions={"output":predictions},loss=cross_entropy,eval_metric_ops={"acc":accuracy,"auc":auc},evaluation_hooks=None)

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss=cross_entropy,global_step=global_step)#TODO global_step= 加不加这个有什么区别？不加没有打印信息
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode,loss=cross_entropy,train_op=train_op)

def build_estimator(model_dir):
    wide_part, deep_part = build_model_columns()
    #config = tf.ConfigProto(device_count={"GPU": 1},  # limit to GPU usage   #TODO 待研究这个是做什么的
    #                        inter_op_parallelism_threads=0,
    #                        intra_op_parallelism_threads=0,
    #                        log_device_placement=True)
    run_config = load_conf("train.yaml")['runconfig']
    #run_config = tf.estimator.RunConfig(**run_config).replace(session_config=config)  #TODO 直接赋值不行吗？这么复杂的操作
    run_config = tf.estimator.RunConfig(**run_config)
    model_config = load_conf("model.yaml")
    params = {"wide_part": wide_part,"deep_part":deep_part,}
    return tf.estimator.Estimator(model_dir=model_dir,
                                  model_fn=model_fn,
                                  params=params,
                                  config=run_config)#TODO need to add
def run():
    train_conf = load_conf("train.yaml")["train"]
    model_dir = train_conf["model_dir"]
    train_data = train_conf["train_data"]
    eval_data = train_conf["eval_data"]
    train_epochs = train_conf["train_epochs"]
    eval_epochs = train_conf["eval_epochs"]
    batch_size = train_conf["batch_size"]
    columns_name = train_conf["columns_name"]
    label_name = train_conf["label_name"]


    estimator = build_estimator(model_dir=model_dir)
    tf.estimator.train_and_evaluate(estimator,
                                    train_spec=tf.estimator.TrainSpec(input_fn=lambda: input_fn(file_path = train_data,
                                                                                                epochs = train_epochs,
                                                                                                batch_size = batch_size,
                                                                                                columns_name = columns_name,
                                                                                                label_name = label_name),
                                                                      hooks=None),
                                    eval_spec=tf.estimator.EvalSpec(input_fn=lambda: input_fn(file_path = eval_data,
                                                                                              epochs = eval_epochs,
                                                                                              batch_size = batch_size,
                                                                                              columns_name = columns_name,
                                                                                              label_name = label_name),
                                                                    steps=None,
                                                                    throttle_secs=6))

#estimator = tf.estimator.Estimator(model_fn=mode_fn,
#                                   model_dir=model_dir,
#                                   config=None,
#                                   params=None,
#                                   warm_start_from=None)#TODO 参数config, params待设定


if __name__=="__main__":
    run()

