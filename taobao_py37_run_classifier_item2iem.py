import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow.keras.backend as K

from recommender.models import DeepFM,WDL,PNN,FNN,DCN,AFM,xDeepFM,AutoInt,FM,LR
from recommender.inputs import SparseFeat, DenseFeat, get_feature_names
config=tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.4
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
#tf.compat.v1.disable_eager_execution()
#tf.enable_eager_execution()
if __name__ == "__main__":
    print("start to read data")
    #sparse_features = ['C' + str(i) for i in range(1, 4)]
    #dense_features = ['I' + str(i) for i in range(1, 14)]
    dense_features = ['e'+str(i) for i in range(128)]
    sparse_features= ["userid","item","category"]
    column_names =["userid","item","category"] + dense_features
    label_name = 'buy_flag'
    #LABELS = [0, 1]
    l = [0.0]
    b = [[""]]*3
    dense = [[0.0]]*128
    
    def my_input_fn1(file_path=None, perform_shuffle=True, repeat_count=1, batch_size=1000):
        def decode_csv(line):
            parsed_line = tf.io.decode_csv(line, record_defaults=b+l+dense, field_delim=',')
            label = parsed_line[3]  
            del parsed_line[3]  
            #for i in range(13):
            #    parsed_line[i] = tf.cond(parsed_line[i] < 2.0, lambda: parsed_line[i], lambda: tf.square(tf.math.log(parsed_line[i])))
            features = parsed_line  # Everything but last elements are the features
            d = dict(zip(column_names, features)), label
            return d

        dataset = (tf.data.TextLineDataset(file_path, num_parallel_reads=40)  # Read text file
               .map(decode_csv,num_parallel_calls=40))  # Trans:form each elem by applying decode_csv fn
        if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
            dataset = dataset.shuffle(buffer_size = batch_size*8)
        dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
        dataset = dataset.batch(batch_size=batch_size)  # Batch size to use
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        #dataset = dataset.cache()
        #datatset = dataset.make_initializable_iterator()
        #datatset = dataset.make_one_shot_iterator()
        dataset = dataset.make_one_shot_iterator()
        features, labels = dataset.get_next()
        return features, labels
        
    def my_input_fn(files):
       while (True):
         dataset = my_input_fn1(file_path=files)
         for tupl in dataset:
             yield tupl
    '''
    column_names = column_names + [label_name]
    def get_dataset(file_path):
        dataset = tf.data.experimental.make_csv_dataset(
      							file_path,
      							batch_size=256, # 为了示例更容易展示，手动设置较小的值
      							column_names=column_names,
      							label_name=label_name,
      							field_delim='\t',
                                                        num_parallel_reads=10,
                                                        #sloppy=True,
                                                        #prefetch_buffer_size=5,
                                                        #shuffle_buffer_size=1000000,
      							#na_value="?",
      							num_epochs=1)
        dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        #dataset.prefetch(buffer_size=1)
        k = dataset.make_initializable_iterator()
        return k
    '''
    
    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    # 2.count #unique features for each sparse field,and record dense feature field name
    #1460,579,9125891,2016487,305,24,12466,633,3,90709,5629,
    #7537623,3182,27,14808,4953014,10,5601,2172,4,
    #6374353,18,15,270878,105,137547
    #vocabulary_size = {"C1":1460,"C2":579,"C3":9125891,"C4":2016487,"C5":305,"C6":24,"C7":12466,"C8":1200,"C9":10,"C10":90709,"C11":11000,
    #                    "C12":7537623,"C13":6100,"C14":50,"C15":31000,"C16":4953014,"C17":100,"C18":10000,"C19":10000,"C20":10,
    #                    "C21":6374353,"C22":100,"C23":100,"C24":270878,"C25":200,"C26":137547}
    #item: 777745
    #category: 6814
    #userid: 29154
    vocabulary_size = {"item":1500000,"category":14000,"userid":60000}
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=vocabulary_size[feat], embedding_dim=8, use_hash=True,dtype='string')
                           for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,) for feat in dense_features]
    
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)


    # 3.generate input data for model
    model_name="WDL"
    callbacks = [tf.keras.callbacks.TensorBoard(
                      log_dir='./taobao_logs',
                      #histogram_freq=1,
                      #embeddings_freq=1,
                      profile_batch=1000,
		      update_freq='batch'),
                 tf.keras.callbacks.ModelCheckpoint(
                      filepath='./taobao_models/' + model_name,
                      monitor='binary_crossentropy',
                      verbose=1,
                      save_best_only=True,
                      save_weights_only=False,
                      mode='auto',
                      save_freq='epoch')]
    # 4.Define Model,train,predict and evaluate
    flag = "train"
    if flag == "train":
        epochs = 3  #note its relationship with var:validation_freq  in model.fit function
        batch_size = 2000
        train_steps = int(2196682/batch_size)
        val_steps = int(549170/batch_size)
        features, labels = my_input_fn1(file_path='./dataset/taobao_data/data_train_embedding', perform_shuffle=True, repeat_count=epochs, batch_size=batch_size)
        val_features, val_labels = my_input_fn1(file_path='./dataset/taobao_data/data_test_embedding', perform_shuffle=False, repeat_count=epochs,batch_size=batch_size)
        pred_features, pred_labels = my_input_fn1(file_path='./dataset/taobao_data/data_test_embedding', perform_shuffle=False, repeat_count=1,batch_size=batch_size)
        pred_steps = val_steps
    else:
        epochs = 1
        batch_size = 1000
        train_steps = int(100000/batch_size)
        val_steps = int(100000/batch_size)
        features, labels = my_input_fn1(file_path='../dataset/dac_sample.txt', perform_shuffle=True, repeat_count=epochs, batch_size=batch_size)
        val_features, val_labels = my_input_fn1(file_path='../dataset/dac_sample.txt', perform_shuffle=False, repeat_count=epochs, batch_size=batch_size)
        pred_features, pred_labels = my_input_fn1(file_path='../dataset/dac_sample.txt', perform_shuffle=False, repeat_count=1, batch_size=batch_size)
        pred_steps = val_steps
    #with tf.device('gpu:0'):
    #model_name="FNN"
    if model_name=="WDL":   
        model = WDL(linear_feature_columns,dnn_feature_columns,dnn_hidden_units=[1024,512,256],dnn_dropout=0.7,l2_reg_dnn=0,task='binary')
        model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy',tf.keras.metrics.AUC()])
        #model.fit_generator(generator=get_dataset('./data/train_criteo.txt'), steps_per_epoch=None, epochs=10, verbose=2, callbacks=callbacks, 
        #                    validation_data=get_dataset('./data/test_criteo.txt'), validation_steps=None, validation_freq=2, class_weight=None,
        #                    max_queue_size=100, workers=10, use_multiprocessing=False, shuffle=True, initial_epoch=0)
        #model.fit_generator(generator=my_input_fn('./data/train_criteo.txt'), epochs=10, verbose=2, steps_per_epoch=5100*8, callbacks=callbacks,
        #                    validation_data=my_input_fn('./data/train_criteo.txt'), validation_steps=5100*8, validation_freq=2, class_weight=None,
        #                    max_queue_size=1000, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
        history = model.fit(features, labels, batch_size=batch_size, epochs=epochs, verbose=2, steps_per_epoch=train_steps,validation_steps=val_steps,
                        use_multiprocessing=True, workers=10, max_queue_size=100, validation_data=(val_features, val_labels),
                        shuffle=True, validation_freq=1, callbacks=callbacks)
        pred = model.predict(x=pred_features,steps=pred_steps,batch_size=batch_size)
        np.savetxt("wdl_pred.txt",pred)
        #pred_ans = model.predict(generator=my_input_fn('./data/test_criteo.txt'))
        #print("WDL test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
        #print("WDL test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
        #print("WDL test MSE", round(mean_squared_error(test[target].values, pred_ans), 4))
    elif model_name=="DeepFM":
        model2 = DeepFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=[256,128,64], dnn_dropout=0.5, l2_reg_dnn=0.0001,task='binary')
        model2.compile("adam","binary_crossentropy", metrics=['binary_crossentropy', tf.keras.metrics.AUC()])
        history = model2.fit(features, labels, batch_size=batch_size, epochs=epochs, verbose=2, steps_per_epoch=train_steps,validation_steps=val_steps,
                        use_multiprocessing=True, workers=10, max_queue_size=100, validation_data=(val_features, val_labels),
                        shuffle=True, validation_freq=1, callbacks=callbacks)
        pred = model2.predict(x=pred_features,steps=pred_steps,batch_size=batch_size)
        np.savetxt("deepfm_pred.txt",pred)
    
        #model2 = DeepFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=[1024,512,256], task='binary')
        #model2.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'])
        #model2.fit_generator(generator=generator_train, epochs=10, verbose=2, callbacks=callbacks,
        #                    validation_data=generator_test, validation_steps=None, validation_freq=2, class_weight=None,
        #                    max_queue_size=100, workers=10, use_multiprocessing=False, shuffle=True, initial_epoch=0)
    elif model_name=="PNN":
        model3 = PNN(dnn_feature_columns, embedding_size=8, dnn_hidden_units=[1024,512,256],use_inner=True,use_outter=True, task='binary')
        model3.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy',tf.keras.metrics.AUC()])
        history = model3.fit(features, labels, batch_size=batch_size, epochs=epochs, verbose=2, steps_per_epoch=train_steps,validation_steps=val_steps,
                        use_multiprocessing=True, workers=10, max_queue_size=100, validation_data=(val_features, val_labels),
                        shuffle=True, validation_freq=1, callbacks=callbacks)
        pred = model3.predict(x=pred_features,steps=pred_steps,batch_size=batch_size)
        np.savetxt("pnn_pred.txt",pred)
    elif model_name=="FNN":
        model4 = FNN(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=[1024,512,256], task='binary')
        model4.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy',tf.keras.metrics.AUC()])
        history = model4.fit(features, labels, batch_size=batch_size, epochs=epochs, verbose=2, steps_per_epoch=train_steps,validation_steps=val_steps,
                        use_multiprocessing=True, workers=10, max_queue_size=100, validation_data=(val_features, val_labels),
                        shuffle=True, validation_freq=1, callbacks=callbacks)
        pred = model3.predict(x=pred_features,steps=pred_steps,batch_size=batch_size)
        np.savetxt("fnn_pred.txt",pred)
    elif model_name=="DCN":
        model7 = DCN(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=[1024,512,256], task='binary')
        model7.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy',tf.keras.metrics.AUC()])
        history = model7.fit(features, labels, batch_size=batch_size, epochs=epochs, verbose=2, steps_per_epoch=train_steps,validation_steps=val_steps,
                        use_multiprocessing=True, workers=10, max_queue_size=100, validation_data=(val_features, val_labels),
                        shuffle=True, validation_freq=1, callbacks=callbacks)
        pred = model7.predict(x=pred_features,steps=pred_steps,batch_size=batch_size)
        np.savetxt("dcn_pred.txt",pred)
        pass
    elif model_name=="FM":
        model5 = FM(linear_feature_columns, dnn_feature_columns, task='binary')
        model5.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy',tf.keras.metrics.AUC()])
        history = model5.fit(features, labels, batch_size=batch_size, epochs=epochs, verbose=2, steps_per_epoch=train_steps,validation_steps=val_steps,
                        use_multiprocessing=True, workers=10, max_queue_size=100, validation_data=(val_features, val_labels),
                        shuffle=True, validation_freq=1, callbacks=callbacks)
        pred = model5.predict(x=pred_features,steps=pred_steps,batch_size=batch_size)
        np.savetxt("fm_pred.txt",pred)
    elif model_name=="LR":
        model6 = LR(linear_feature_columns, task='binary')
        model6.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy',tf.keras.metrics.AUC()])
        history = model6.fit(features, labels, batch_size=batch_size, epochs=epochs, verbose=2, steps_per_epoch=train_steps,validation_steps=val_steps,
                        use_multiprocessing=True, workers=10, max_queue_size=100, validation_data=(val_features, val_labels),
                        shuffle=True, validation_freq=1, callbacks=callbacks)
        pred = model6.predict(x=pred_features,steps=pred_steps,batch_size=batch_size)
        np.savetxt("lr_pred.txt",pred)
    elif model_name=="AFM":
        model8 = AFM(linear_feature_columns, dnn_feature_columns, task='binary')
        model8.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy', tf.keras.metrics.AUC()])
        history = model8.fit(features, labels, batch_size=batch_size, epochs=epochs, verbose=2, steps_per_epoch=train_steps,validation_steps=val_steps,
                        use_multiprocessing=True, workers=10, max_queue_size=100, validation_data=(val_features, val_labels),
                        shuffle=True, validation_freq=1, callbacks=callbacks)
        pred = model8.predict(x=pred_features,steps=pred_steps,batch_size=batch_size)
        np.savetxt("afm_pred.txt",pred)
    elif model_name=="xDeepFM":
        model9 = xDeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
        model9.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy', tf.keras.metrics.AUC()])
        history = model9.fit(features, labels, batch_size=batch_size, epochs=epochs, verbose=2, steps_per_epoch=train_steps,validation_steps=val_steps,
                        use_multiprocessing=True, workers=10, max_queue_size=100, validation_data=(val_features, val_labels),
                        shuffle=True, validation_freq=1, callbacks=callbacks)
        pred = model9.predict(x=pred_features,steps=pred_steps,batch_size=batch_size)
        np.savetxt("xdeepfm_pred.txt",pred)
    elif model_name == "AutoInt":
        model10 = AutoInt(linear_feature_columns, dnn_feature_columns, task='binary')
        model10.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy', tf.keras.metrics.AUC()])
        history = model10.fit(features, labels, batch_size=batch_size, epochs=epochs, verbose=2, steps_per_epoch=train_steps,validation_steps=val_steps,
                        use_multiprocessing=True, workers=10, max_queue_size=100, validation_data=(val_features, val_labels),
                        shuffle=True, validation_freq=1, callbacks=callbacks)
        pred = model10.predict(x=pred_features,steps=pred_steps,batch_size=batch_size)
        np.savetxt("autoint_pred.txt",pred)
    else:
        pass
