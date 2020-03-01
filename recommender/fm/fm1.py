import tensorflow as tf 
import numpy as np 
from sklearn.metrics import roc_auc_score
from time import time

class FM(object):
    def __init__(self, feature_size, field_size, embedding_size,
                epoch=10, batch_size=256, learning_rate=0.001,
                optimizer_type='adam', verbose=False, random_seed=2020,
                loss_type='logloss', eval_metric=roc_auc_score)
        #在写这个 函数的时候发现FM没有正则化项。
        assert loss_type in ['logloss', 'mse'],\
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.train_result = []
        self.valid_result = []

        self._init_graph()
    
    def _init_graph(self):
        slef.graph = tf.Graph()
        with self.graph.as_default():##感受不到这么设置这个图有啥用。
            tf.set_random_seed(self.random_seed)
            self.feature_index = tf.placeholder(tf.int32, shape=[None, self.field_size], name='feature_index')
            self.feature_value = tf.placeholder(tf.float32, shape=[None, self.field_size], name='feature_value')
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')
            
            
            weights = tf.Variable(tf.random_uniform([self.feature_size, 1], 0.0, 1.0), name='wi')
            bias = tf.Variable(tf.random_uniform([1, 1], 0.0, 1.0), name='bias')

            #model
            #self.embeddings = tf.nn.embedding_lookup(self.feature_embedding, self.feature_index)
            feature_value = tf.reshape(self.feature_value, shape=[-1, self.field_size]) #fm中不涉及field的概念; batch_size * field_size
            #self.embeddings = tf.multiply()

            #first order
            self.first_order = tf.nn.embedding_lookup(weights, self.feature_index) #batch_size * field_size * 1
            self.first_order = tf.nn.reshape(self.first_order, shape=[-1, self.field_size]) #batch_size * field_size
            
            self.first_order = tf.reduce_sum(tf.multiply(self.first_order, feature_value), 1) #batch_size * 1
            self.line_res = tf.add(self.first_order, bias) #batch_size *1

            #second order
            feature_embedding = tf.Variable(tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01), name='feature_embedding') #feature_size * embedding_size
            self.embeddings = tf.nn.embedding_lookup(self.feature_embedding, self.feature_index) #batch_size * field_size * embedding_size
            self.feature_value = tf.reshape(self.feature_value, shape=[-1, self.field_size, 1]) #fm中不涉及field的概念; batch_size * field_size * 1
            self.embeddings = tf.multiply(self.embeddings, self.feature_value) #batch_size * field_size * embedding_size
            #sum square part
            self.summed_feature_emb = tf.reduce_sum(self.embeddings, 1) #batch_size * embedding_size
            self.summed_feature_emb_square = tf.square(self.summed_feature_emb) #batch_size * embedding_size
            #self.summed_feature_emb_square_batch = tf.reduce_sum(self.summed_feature_emb_square, 1) #batch_size * 1
            #square sum part
            self.squared_feature_emb = tf.square(self.embeddings) #batch_size * field_size * embedding_size
            self.squared_sum_feature_emb = tf.reduce_sum(self.squared_feature_emb, 1) #batch_size * embedding_size
            #self.squared_sum_feature_emb_batch = tf.reduce_sum(self.squared_sum_feature_emb, 1) #batch_size * 1
            #res
            self.fm_res = 0.5 * tf.subtract(self.summed_feature_emb_square, self.squared_sum_feature_emb) # batch_size * embedding_size
            self.fm_res_batch = tf.reduce_sum(self.fm_res, 1) #batch_size * 1

            self.out = tf.add(self.line_res, self.fm_res_batch)
            #loss
            if self.loss_type == 'logloss':
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            else self.loss_type == 'mse':
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))
            
            #optimizer
            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(self.loss)
            elif self.optimizer_type == "yellowfin":
                #self.optimizer = YFOptimizer(learning_rate=self.learning_rate, momentum=0.0).minimize(self.loss)
                pass

            #init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)

            #number of params
            total_parameters = 0
            #for variable in self.weights.values():
            #   pass
            if self.verbose > 0:
                print('#params:%d' % total_parameters)
    def _init_session(self):
        config = tf.ConfigProto(device_count={'gpu':0})
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def get_batch(self, Xi, Xv, y, batch_size, index):
        start = index * batch_size
        end = (index+1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end], [[y_] for y_ in y[start:end]]

    def fit_on_batch(self, Xi, Xv, y):
        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.label: y,
                     #self.dropout_keep_fm: self.dropout_fm,
                     #self.dropout_keep_deep: self.dropout_deep,
                     self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def training_termination(self, valid_result):
        if len(valid_result) > 5:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] and \
                    valid_result[-2] < valid_result[-3] and \
                    valid_result[-3] < valid_result[-4] and \
                    valid_result[-4] < valid_result[-5]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] and \
                    valid_result[-2] > valid_result[-3] and \
                    valid_result[-3] > valid_result[-4] and \
                    valid_result[-4] > valid_result[-5]:
                    return True
        return False
    
    def evaluate(self, Xi, Xv, y):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :param y: label of each sample in the dataset
        :return: metric of the evaluation
        """
        y_pred = self.predict(Xi, Xv)
        return self.eval_metric(y, y_pred)

    def fit(self, Xi_train, Xv_train, y_train,
            Xi_valid=None, Xv_valid=None, y_valid=None,
            early_stopping=False, refit=False):

            has_valid = Xv_valid is not None
            for epoch in range(self.epoch):
                t1 = time()
                # 待添加，shuffle self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
                total_batch = int(len(y_train) / self.batch_size)
                for i in range(total_batch):
                    Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train, self.bath_size, i)
                    self.fit_on_batch(Xi_batch, Xv_batch, y_batch)

            # evaluate training and validation datasets
            train_result = self.evaluate(Xi_train, Xv_train, y_train)
            self.train_result.append(train_result)
            if has_valid:
                valid_result = self.evaluate(Xi_valid, Xv_valid, y_valid)
                self.valid_result.append(valid_result)
            if self.verbose > 0 and epoch % self.verbose == 0:
                if has_valid:
                    print("[%d] train-result=%.4f, valid-result=%.4f [%.1f s]"
                        % (epoch + 1, train_result, valid_result, time() - t1))
                else:
                    print("[%d] train-result=%.4f [%.1f s]"
                        % (epoch + 1, train_result, time() - t1))
            if has_valid and early_stopping and self.training_termination(self.valid_result):
                break

