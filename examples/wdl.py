import os
import tensorflow as tf

from deepctr.models import DeepFM,WDL,PNN,FNN,DCN,AFM,xDeepFM,AutoInt
from deepctr.inputs import SparseFeat, DenseFeat, get_feature_names

if __name__ == "__main__":
    print("start to read data")
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    column_names = ["label"]+ dense_features + sparse_features
    label_name = 'label'
    #LABELS = [0, 1]
    def get_dataset(file_path):
        dataset = tf.data.experimental.make_csv_dataset(
                                                        file_path,
                                                        batch_size=256, # 为了示例更容易展示，手动设置较小的值
                                                        column_names=column_names,
                                                        label_name=label_name,
                                                        field_delim='\t',
                                                        #na_value="?",
                                                        sloppy=True,
                                                        num_epochs=1)
        #dataset.prefetch(batch_size=1)
        dataset = dataset.apply(tf.data.experimental.prefetch_to_device("/cpu:0"))
        dataset = dataset.apply(tf.data.experimental.copy_to_device("/gpu:0"))
        #dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        #dataset = dataset.batch(batch_size=256)
        #raw_train_data = get_dataset().batch(32)
        #k = dataset.make_initializable_iterator()
        #features, labels = k.get_next()
        return dataset
    #raw_train_data = get_dataset("./data/train_criteo.txt")
    raw_train_data = get_dataset("./data/dac_sample.txt")
    #k = raw_train_data.make_initializable_iterator()
    #features, labels = k.get_next()
    '''
    data = pd.read_csv('./data/train_Criteo_full.txt', sep='\t', index_col = False, names=column_names)
    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    '''
    target = ['label']
    #UserID,MovieID,Rating,timestamps,Title,Genres,Gender,Age,OccupationID,Zip-code
    print("read data end")

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    '''
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])
    '''
    # 2.count #unique features for each sparse field,and record dense feature field name
    vocabulary_size = {"C1":100,"C2":100,"C3":100,"C4":100,"C5":100,"C6":100,"C7":100,"C8":100,"C9":100,
                        "C10":100,"C11":100,"C12":100,"C13":100,"C14":100,"C15":100,"C16":100,"C17":100,"C18":100,
                        "C19":100,"C20":100,"C21":100,"C22":100,"C23":100,"C24":100,"C25":100,"C26":100,"C27":100}
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=vocabulary_size[feat], embedding_dim=4, use_hash=True,dtype='string')
                           for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                          for feat in dense_features]
    if tag == "AFM":
        dnn_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4) for i,feat in enumerate(sparse_features)]
    else:
        dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)


    # 3.generate input data for model

    #train, test = train_test_split(data, test_size=0.2)
    #train_model_input = {name:train[name] for name in feature_names}
    #test_model_input = {name:test[name] for name in feature_names}
    #log_dir = os.path.join("logs","tf")
    callbacks = [tf.keras.callbacks.TensorBoard(
                      log_dir='./logs25600',
                      #histogram_freq=1,
                      #embeddings_freq=1,
                      update_freq='epoch')]
    # 4.Define Model,train,predict and evaluate
    #model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression')
    #print("\nlinear_feature_columns",linear_feature_columns,"\n")
    #print("\ndnn_feature_columns",dnn_feature_columns,"\n")

    inputs, output = WDL(linear_feature_columns,dnn_feature_columns,dnn_hidden_units=[1024,512,256],task='binary')
    #
    model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'])
    model.fit_generator(generator=raw_train_data, steps_per_epoch=None, epochs=10, verbose=2, callbacks=callbacks,
                        validation_data=raw_train_data, validation_steps=None, validation_freq=2, class_weight=None,
                        max_queue_size=100, workers=10, use_multiprocessing=True, shuffle=True, initial_epoch=0)
    #model.fit_generator(generator=get_dataset('./data/train_criteo.txt'), steps_per_epoch=None, epochs=10, verbose=2, callbacks=callbacks,
    #                    validation_data=get_dataset('./data/test_criteo.txt'), validation_steps=None, validation_freq=2, class_weight=None,
    #                    max_queue_size=100, workers=10, use_multiprocessing=True, shuffle=True, initial_epoch=0)
    #history = model.fit(train_model_input, train[target].values,batch_size=256, epochs=10, verbose=2, validation_split=0.2, shuffle=True, callbacks=callbacks)
    #history = model.fit(train_model_input, train[target].values,batch_size=256, epochs=10, verbose=2, validation_split=0.2, shuffle=True, callbacks=callbacks)
    pred_ans = model.predict(generator=get_dataset('./data/test_criteo_full.txt'))
