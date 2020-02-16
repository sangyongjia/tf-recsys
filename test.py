
import numpy as np
import tensorflow as tf
from tfcf.metrics import mae
from tfcf.metrics import rmse
from tfcf.datasets import ml1m
from tfcf.config import Config
from tfcf.models.svd import SVD
from tfcf.models.svdpp import SVDPP
from sklearn.model_selection import train_test_split

# Note that x is a 2D numpy array, 
# x[i, :] contains the user-item pair, and y[i] is the corresponding rating.
x, y = ml1m.load_data()

##在划分数据集的时候没有考虑时间序。待改进
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

config = Config()
config.num_users = np.max(x[:, 0]) + 1
config.num_items = np.max(x[:, 1]) + 1
config.min_value = np.min(y)
config.max_value = np.max(y)

with tf.Session() as sess:
    '''
    # For SVD++ algorithm, if `dual` is True, then the dual term of items' 
    # implicit feedback will be added into the original SVD++ algorithm.
    # model = SVDPP(config, sess, dual=False)
    # model = SVDPP(config, sess, dual=True)
    model = SVD(config, sess)
    model.train(x_train, y_train, validation_data=(
        x_test, y_test), epochs=20, batch_size=1024)
        
    y_pred = model.predict(x_test)
    print('rmse: {}, mae: {}'.format(rmse(y_test, y_pred), mae(y_test, y_pred)))
        
    # Save model
    model = model.save_model('model/')
    '''
    # For SVD++ algorithm, if `dual` is True, then the dual term of items' 
    # implicit feedback will be added into the original SVD++ algorithm.
    model = SVDPP(config, sess, dual=False)
    # model = SVDPP(config, sess, dual=True)
    #model = SVD(config, sess)
    model.train(x_train, y_train, validation_data=(
        x_test, y_test), epochs=20, batch_size=1024)
        
    y_pred = model.predict(x_test)
    print('rmse: {}, mae: {}'.format(rmse(y_test, y_pred), mae(y_test, y_pred)))
        
    # Save model
    model = model.save_model('model/svd++')
    
    # Load model
    # model = model.load_model('model/')