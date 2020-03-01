import tensorflow as tf
from fm2 import FM


def create_model_fn(model):
    def model_fn(features, labels, params, mode):

        if params.threshold:
            threshold = params.threshold
        else:
            threshold = 0.5

        
        df_i = features['df_i']
        df_v = features['df_v']

        logits = model(params, df_i, df_v).logits


        if mode == tf.estimator.ModeKeys.PREDICT:
            pre = tf.nn.sigmoid(logits, name="sigmoid")
            predict = tf.cast(pre > threshold, dtype=tf.int32)
            predictions = {
                "predict_pro": pre,
                "predict": predict
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)


        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), name="loss")
            loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=labels, predictions=logits))
            
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.train.get_global_step(),
                learning_rate=params.lr,
                clip_gradients=10.0,
                optimizer=params.opt_type
            )

            #pre = tf.nn.sigmoid(logits, name="sigmoid")
            #auc = tf.metrics.auc(labels=labels, predictions=pre, name="auc")
            #accuracy = tf.metrics.accuracy(labels=labels, predictions=tf.cast(pre > threshold, tf.float32), name="accuracy")

            #tf.summary.scalar('train_accuracy', accuracy[1])
            #tf.summary.scalar('train_auc', auc[1])

            pre = logits
            mse = tf.reduce_mean(tf.metrics.mean_squared_error(labels=labels, predictions=pre, name="mse"))
            tf.summary.scalar('mse', mse)
            tf.summary.scalar('loss', loss)
            #tf.summary.scalar('train_auc', auc[1])
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


        if mode == tf.estimator.ModeKeys.EVAL:
            #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), name="loss")
            loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=labels, predictions=logits))

            #pre = tf.nn.sigmoid(logits, name="sigmoid")
            #predict = tf.cast(pre > threshold, dtype=tf.int32)
            #auc = tf.metrics.auc(labels=labels, predictions=pre, name="auc")
            #accuracy = tf.metrics.accuracy(labels=labels, predictions=tf.cast(pre > threshold, tf.float32), name="accuracy")
            pre = logits
            mse = tf.metrics.mean_squared_error(labels=labels, predictions=pre, name="mse")
            #metrics = {
            #    "auc":auc,
            #    "accuracy":accuracy
            #}
            metrics = {
                "mse":mse
            }

            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    return model_fn
