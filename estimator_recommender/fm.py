import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
config=tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.1
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

column_names=['userid','item','category','buy_flag']  #decided by data in order
label_name="buy_flag" #also decide by data

def input_fn(file_path,epochs,batch_size,column_names,label_name):
    dataset = tf.data.experimental.make_csv_dataset(file_path,
                                                    batch_size=batch_size,
                                                    column_names=column_names,
                                                    label_name=label_name,
                                                    na_value="?",
                                                    num_epochs=epochs)
    dataset = dataset.shuffle(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def fc_column(feature_name, hash_bucket_size, embedding_dim=4, dtype=tf.int64):
    f1 = tf.feature_column.categorical_column_with_hash_bucket(feature_name, hash_bucket_size,dtype)
    fc = tf.feature_column.embedding_column(f1,dimension=embedding_dim,)
    return fc

def fc_columns_transform(feature_name, has_bucket_size, embedding_dim=4, dtype=tf.int64):
    dense_layer = tf.keras.layers.DenseFeatures([fc_column(feature_name, has_bucket_size, embedding_dim, dtype)])
    return dense_layer

def model_fn(features, labels, mode, params):
    global_step = tf.train.get_global_step()
    #linear part
    userid_linear = fc_columns_transform("userid",60000,1)(features)
    categoryid_linear = fc_columns_transform("category",14000,1)(features)
    itemid_linear = fc_columns_transform("item",1500000,1)(features)
    dense_layer = tf.keras.layers.Concatenate(axis=-1)([userid_linear,categoryid_linear,itemid_linear])
    linear_part = tf.keras.layers.Dense(units=1,activation='sigmoid',use_bias=True,)(dense_layer)

    #FM,feature cross part
    userid_fm = tf.expand_dims(fc_columns_transform("userid",60000,4)(features),1)
    categoryid_fm = tf.expand_dims(fc_columns_transform("category",14000,4)(features),1)
    itemid_fm = tf.expand_dims(fc_columns_transform("item",1500000,4)(features),1)

    feature_matrix = tf.keras.layers.Concatenate(axis=1)([userid_fm, categoryid_fm, itemid_fm])
    sum_square = tf.square(tf.reduce_sum(feature_matrix, axis=1, keepdims=True))
    print("sum_square:",sum_square)
    square_sum = tf.reduce_sum(tf.square(feature_matrix),axis=1, keepdims=True)
    minus =0.5*tf.reduce_sum(sum_square - square_sum, axis=2)

    output = tf.sigmoid(linear_part + minus)
    print("linear_part:",linear_part)
    print("minus:",minus)
    print("output:",output)

    if tf.estimator.ModeKeys.PREDICT == mode:
        return tf.estimator.EstimatorSpec(mode=mode,predictions={"output":output})

    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, tf.float32), logits=tf.squeeze(output)))
    auc = tf.metrics.auc(labels=labels,predictions=output,name="auc")
    #auc = tf.keras.metrics.AUC(labels=labels,predictions=output,name="auc")
    acc = tf.metrics.accuracy(labels=labels, predictions=output, name="accuracy")
    #acc = tf.keras.metrics.Accuracy(labels=labels, predictions=output, name="accuracy")

    if tf.estimator.ModeKeys.EVAL == mode:
        return tf.estimator.EstimatorSpec(mode=mode, loss=cross_entropy, eval_metric_ops={"auc":auc,"accurancy":acc})#loss去除掉？eval算什么loss


    #optimizer = tf.keras.optimizers.Adam()
    #train_op = optimizer.minimize(loss=cross_entropy)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss=cross_entropy, global_step=global_step)
    if tf.estimator.ModeKeys.TRAIN == mode:
        return tf.estimator.EstimatorSpec(mode=mode, loss=cross_entropy, train_op=train_op)
    pass
checkpointing_config = tf.estimator.RunConfig(
    model_dir="./model_dir",
    tf_random_seed = 2020,
    log_step_count_steps=2000,
    #save_checkpoints_secs=30,  # Save checkpoints every 20 minutes.
    save_checkpoints_steps=2000,
    keep_checkpoint_max=5,  # Retain the 10 most recent checkpoints.
)

estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=None, config=checkpointing_config, params={}, warm_start_from=None)

#saved model part
feature_columns = [fc_column("userid", 60000, 1), fc_column("category", 14000, 1), fc_column("item",1500000,1)]
feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec=feature_spec)
best_exporter = tf.estimator.BestExporter(serving_input_receiver_fn=serving_input_receiver_fn,exports_to_keep=2)
exporters = [best_exporter]

tf.estimator.train_and_evaluate(estimator=estimator,
                                train_spec=tf.estimator.TrainSpec(input_fn=lambda:input_fn(file_path="../dataset/taobao_data/data_train.csv",
                                                                                           epochs=30,
                                                                                           batch_size=1000,
                                                                                           column_names=column_names,
                                                                                           label_name=label_name),
                                                                  max_steps=None,
                                                                  hooks=None),
                                eval_spec=tf.estimator.EvalSpec(input_fn=lambda:input_fn(file_path="../dataset/taobao_data/data_test.csv",
                                                                                         epochs=1,
                                                                                         batch_size=1000,
                                                                                         column_names=column_names,
                                                                                         label_name=label_name),
                                                                steps=None,
                                                                name="taobao_dataset",
                                                                hooks=None,
                                                                exporters=exporters,
                                                                start_delay_secs=120,
                                                                throttle_secs=6))






