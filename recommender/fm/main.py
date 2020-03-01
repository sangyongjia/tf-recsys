import tensorflow as tf
from util import FieldHandler, transformation_data, dataGenerate
from fm_model_fn import create_model_fn
from input_fn import create_input_fn
from hparams import create_hparams, FLAGS
from fm2 import FM
#from model.FFM import FFM

tf.logging.set_verbosity(FLAGS.logging_level)


def main(_):
    fh = FieldHandler(train_file_path=FLAGS.train_file_path,
                    category_columns=FLAGS.category_columns,
                    continuation_columns=FLAGS.continuation_columns)
    #features, labels = fh.get_features_labels()
    features, labels = transformation_data(file_path=FLAGS.train_file_path, field_hander=fh, label=FLAGS.label)
    test_features, test_labels = transformation_data(file_path=FLAGS.test_file_path, field_hander=fh, label=FLAGS.label)
    hparams = create_hparams(fh.feature_nums)

    train_input_fn = create_input_fn(features,
                                           labels=labels,
                                           batch_size=hparams.batch_size,
                                           num_epochs=hparams.epoches)

    test_input_fn = create_input_fn(test_features,
                                           labels=test_labels,
                                           batch_size=hparams.batch_size)
    model_fn = create_model_fn(FM)
    estimator = tf.estimator.Estimator(
        model_fn = model_fn,
        model_dir=FLAGS.model_path,
        params=hparams,
        config=tf.estimator.RunConfig(
            tf_random_seed=hparams.seed,
            log_step_count_steps=500
        )
    )

    #show_dict = {
    #    "loss":"loss",
    #    "accuracy":"accuracy/value",
    #    "auc":"auc/value"
    #}
    show_dict = {
        "mse":"mse/value"
    }
   
    log_hook = tf.train.LoggingTensorHook(show_dict, every_n_iter=1000)
    #estimator.train(input_fn=train_input_fn, hooks=[log_hook])

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, hooks=[log_hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=test_input_fn, )
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    
if __name__ == "__main__":
    tf.app.run()
