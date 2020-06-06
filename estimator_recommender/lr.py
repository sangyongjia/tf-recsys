import tensorflow as tf 
from tensorflow import feature_column
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1
columns=['userid','item','category','buy_flag']  #decide by data source
label_name="buy_flag" #also decide by data source
batch_size=1000
tf.logging.set_verbosity(tf.logging.INFO)
def input_fn(file_path,epochs=1):
    dataset = tf.data.experimental.make_csv_dataset(file_path,
                                                    batch_size=batch_size,
                                                    column_names=columns,
                                                    label_name=label_name,
                                                    na_value="?",
                                                    num_epochs=epochs)
    dataset = dataset.shuffle(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def fc_column(feature_name, hash_bucket_size, embedding_size=1, dtype=tf.string):
    f = feature_column.categorical_column_with_hash_bucket(feature_name, hash_bucket_size=hash_bucket_size, dtype=dtype)
    f1 = feature_column.embedding_column(f, embedding_size, initializer=RandomNormal(mean=0.1, stddev=0.001, seed=2020),trainable=True)
    return f1

def fc_transform(feature_name, hash_bucket_size, embedding_size=1, dtype=tf.string):
    feature_layer = tf.keras.layers.DenseFeatures([fc_column(feature_name, hash_bucket_size, embedding_size, dtype)])
    return feature_layer

def model_fn(features, labels, mode, params):

    global_step = tf.train.get_or_create_global_step()  #?

    user_id = fc_transform('userid', 60000, dtype=tf.int32)(features)
    category_id = fc_transform('category', 14000, dtype=tf.int32)(features)
    item_id = fc_transform('item', 1500000, dtype=tf.int32)(features)

    layer1 = tf.keras.layers.Concatenate(axis=-1)([user_id, category_id, item_id])
    print("\n\nlayer1:\n\n",layer1)
    output = tf.keras.layers.Dense(units=1, 
				   use_bias=True, 
                                   activation=tf.sigmoid,
                                   kernel_regularizer=l2(1e-5),
                                   kernel_initializer=tf.keras.initializers.glorot_normal(),
                                   bias_initializer=tf.keras.initializers.Zeros())(layer1)  #kernel_regularizer=tf.keras.regularizers.l1(l=0.01)
    #output = tf.sigmoid(output1)
    print("\n\noutput:",output)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={"output":output})

    print("**********labels:",labels)
    print("**********output:",output)
    #tf.losses.sigmoid_cross_entropy
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels,tf.float32),logits=tf.squeeze(output)))
    #accuracy = tf.metrics.accuracy(labels=labels, predictions=output, name="acc")
    auc = tf.metrics.auc(labels=labels, predictions=output, name="auc")

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=cross_entropy, eval_metric_ops={"auc":auc},evaluation_hooks=None)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001,epsilon=1e-07,)
    train_op = optimizer.minimize(loss=cross_entropy, global_step=global_step)

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode, loss=cross_entropy, train_op=train_op)


estimator = tf.estimator.Estimator(model_fn=model_fn, 
                                    model_dir="./mode_dir",
                                    config=tf.estimator.RunConfig(save_checkpoints_steps=2000, keep_checkpoint_max=5),
                                    params={"optimizer":"adam"})
#hook = tf.train.ProfilerHook(save_steps=20,output_dir="./tracing", show_dataflow=True,show_memory=True)
#hooks = [hook]
tf.estimator.train_and_evaluate(estimator,
                                train_spec=tf.estimator.TrainSpec(input_fn=lambda: input_fn("../dataset/taobao_data/data_train.csv",epochs=30),hooks=None),
                                eval_spec=tf.estimator.EvalSpec(throttle_secs=6,input_fn=lambda:input_fn("../dataset/taobao_data/data_test.csv",epochs=1), steps=None),
                                )

'''tf.estimator.train_and_evaluate(estimator,train_spec=tf.estimator.TrainSpec(input_fn=lambda: input_fn("../DeepCTR/data/taobao_data/data_train.csv",epochs=10),hooks=hooks),
                                          eval_spec=tf.estimator.EvalSpec(input_fn=lambda:input_fn("../DeepCTR/data/taobao_data/data_test.csv", epochs=1), steps=None),
                                )'''
#estimator.train(input_fn=lambda: input_fn("/Users/sangyongjia/Documents/MachineLearning/github/MovieRecommendation/LFM/tf-recsys/dataset/taobao_data/data_train.csv"))
#data = input_fn("/Users/sangyongjia/Documents/MachineLearning/github/MovieRecommendation/LFM/tf-recsys/dataset/taobao_data/data_train.csv")
print("hello world")
