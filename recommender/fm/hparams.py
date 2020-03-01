import tensorflow as tf
from collections import namedtuple



tf.flags.DEFINE_string("opt_type", "Adam", "optimizer type (Adagrad, Adam, Ftrl, Momentum, RMSProp, SGD).")
tf.flags.DEFINE_string("train_file_path", "./dataset/train.csv", "train file path.")
tf.flags.DEFINE_string("test_file_path", "./dataset/test.csv", "test file path.")
tf.flags.DEFINE_string("label", "Rating", "target column name.")
#tf.flags.DEFINE_string("activation", "relu", "deep mid activation function(tanh, relu, tanh, sigmoid).")
tf.flags.DEFINE_float("threshold", 0.5, "bi-classification threshold." )
tf.flags.DEFINE_string("loss_type", "log_loss", "bi-classification is log_loss, regression is mse.")
tf.flags.DEFINE_string("model_path", "./checkpoint/", "save model path.")
#tf.flags.DEFINE_bool("use_deep", True, "Whether to use deep or not.")
tf.flags.DEFINE_string("model", "fm", "fm or ffm.")
#tf.flags.DEFINE_list("layers", [30,30], "deep mid layers.")
tf.flags.DEFINE_list("category_columns", ["UserID", "Genres","Gender","Age","OccupationID","year","zip-code"], "category columns.")
tf.flags.DEFINE_list("continuation_columns", [], "continuation columns.")
tf.flags.DEFINE_float("lr", 0.01, "learning rate.")
#tf.flags.DEFINE_float("line_output_keep_dropout", 0.9, "line output keep dropout in deep schema.")
#tf.flags.DEFINE_float("fm_output_keep_dropout", 0.9, "fm output keep dropout in deep schema.")
#tf.flags.DEFINE_float("deep_output_keep_dropout", 0.9, "deep output keep dropout in deep schema.")
#tf.flags.DEFINE_float("deep_input_keep_dropout", 0.9, "deep input keep dropout in deep schema.")
#tf.flags.DEFINE_float("deep_mid_keep_dropout", 0.8, "deep mid keep dropout in deep schema.")
tf.flags.DEFINE_integer("embedding_size", 3, "field embedding size")
tf.flags.DEFINE_bool("use_batch_normal", False, "Whether to use batch normal or not.")
tf.flags.DEFINE_integer("batch_size", 64, "batch size.")
tf.flags.DEFINE_integer("epoches", 10, "epoches.")
tf.flags.DEFINE_integer("logging_level", 20, "tensorflow logging level.")
tf.flags.DEFINE_integer("seed", 20, "tensorflow seed num.")


FLAGS = tf.flags.FLAGS



HParams = namedtuple(
  "HParams",
  [
    "opt_type",
    "threshold",
    "loss_type",
    "model",
    "lr",
    "embedding_size",
    "use_batch_normal",
    "batch_size",
    "epoches",
    "feature_nums",
    "seed"
  ])


def create_hparams(feature_nums):
  return HParams(
    model=FLAGS.model,
    opt_type=FLAGS.opt_type,
    threshold=FLAGS.threshold,
    loss_type=FLAGS.loss_type,

    lr=FLAGS.lr,

    embedding_size=FLAGS.embedding_size,
    use_batch_normal=FLAGS.use_batch_normal,
    batch_size=FLAGS.batch_size,
    epoches=FLAGS.epoches,
    seed=FLAGS.seed,
    feature_nums=feature_nums
    )
