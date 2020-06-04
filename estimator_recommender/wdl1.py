import tensorflow as tf
from recommender.models.wdl import  WDL
from recommender.models.lr import LR
from recommender.inputs import SparseFeat, DenseFeat

config=tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.1
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


tf.logging.set_verbosity(tf.logging.INFO)

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

sparse_features= ["userid","item","category"]
vocabulary_size = {"item": 1500000, "category": 14000, "userid": 60000}
fixlen_feature_columns = [
    SparseFeat(feat, vocabulary_size=vocabulary_size[feat], embedding_dim=4, use_hash=True, dtype='string')
    for i, feat in enumerate(sparse_features)]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns

#feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

model = LR(linear_feature_columns, task='binary')
#model = WDL(linear_feature_columns,dnn_feature_columns,dnn_hidden_units=[1024,512,256],dnn_dropout=0.7,l2_reg_dnn=0,task='binary')
model.compile("adam", "binary_crossentropy", metrics=[tf.keras.metrics.BinaryCrossentropy()]) #tf.keras.metrics.AUC() tf.keras.metrics.BinaryCrossentropy()

checkpointing_config = tf.estimator.RunConfig(
    model_dir="./model_dir",
    log_step_count_steps=10,
    #save_checkpoints_secs=30,  # Save checkpoints every 20 minutes.
    save_checkpoints_steps=10,
    keep_checkpoint_max=3,  # Retain the 10 most recent checkpoints.
)
estimator = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir="./model_dir", config=checkpointing_config)


def my_auc(labels, predictions):
    auc_metric = tf.keras.metrics.AUC(name="my_auc")
    auc_metric.update_state(y_true=labels, y_pred=predictions)
    return {'auc': auc_metric}
estimator=tf.estimator.add_metrics(estimator=estimator,metric_fn=my_auc)
#estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=None, config=checkpointing_config, params={}, warm_start_from=None)

#saved model part
def fc_column(feature_name, hash_bucket_size, embedding_dim=4, dtype=tf.int64):
    f1 = tf.feature_column.categorical_column_with_hash_bucket(feature_name, hash_bucket_size,dtype)
    fc = tf.feature_column.embedding_column(f1,dimension=embedding_dim,)
    return fc
feature_columns = [fc_column("userid", 60000, 1), fc_column("category", 14000, 1), fc_column("item",1500000,1)]
feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec=feature_spec)
best_exporter = tf.estimator.BestExporter(serving_input_receiver_fn=serving_input_receiver_fn,exports_to_keep=2)
exporters = [best_exporter]

tf.estimator.train_and_evaluate(estimator=estimator,
                                train_spec=tf.estimator.TrainSpec(input_fn=lambda:input_fn(file_path="../dataset/taobao_data/data_train.csv",
                                                                                           epochs=1,
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






