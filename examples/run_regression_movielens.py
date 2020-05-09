import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from recommender.models import DeepFM
from recommender.models import WDL
from recommender.inputs import SparseFeat
from recommender.inputs import get_feature_names

if __name__ == "__main__":
    #UserID,MovieID,Rating,timestamps,Title,Genres,Gender,Age,OccupationID,Zip-code
    data = pd.read_csv("../dataset/deep_train.csv")
    data = data.sort_values(by=['timestamps'])
    print("same?:\n",data[-1:])

    sparse_features = ["UserID", "MovieID","Gender", "Age", "OccupationID"]
    target = ['Rating']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        print(feat)
        data[feat] = lbe.fit_transform(data[feat])
    # 2.count #unique features for each sparse field
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(),embedding_dim=4) for feat in sparse_features]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    # 3.generate input data for model
    train = data[:int(0.8*len(data))]
    test = data[int(0.8*len(data))+1:]
    print("same?:\n",test[-1:])

    #train, test = train_test_split(data, test_size=0.2)
    train_model_input = {name:train[name].values for name in feature_names}
    test_model_input = {name:test[name].values for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    #model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression')
    print("\nlinear_feature_columns",linear_feature_columns,"\n")
    print("\ndnn_feature_columns",dnn_feature_columns,"\n")

    model = WDL(linear_feature_columns,dnn_feature_columns,dnn_hidden_units=[1024,512,256],task='regression')
    model.compile("adam", "mse", metrics=['mse'], )

    history = model.fit(train_model_input, train[target].values,batch_size=256, epochs=20, verbose=2, validation_split=0.2, shuffle=True)
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("WDL test MSE", round(mean_squared_error(test[target].values, pred_ans), 4))

    model2 = DeepFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=[1024,512,256], task='regression')
    model2.compile("adam", "mse", metrics=['mse'])
    history2 = model2.fit(train_model_input, train[target].values,batch_size=256, epochs=20, verbose=2, validation_split=0.2, shuffle=True)
    pred_ans2 = model2.predict(test_model_input, batch_size=256)
    print("DeepFM test MSE",round(mean_squared_error(test[target].values, pred_ans2),4))

    model3 = PNN(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=[1024,512,256], task='regression')
    model3.compile("adam", "mse", metrics=['mse'])
    history3 = model3.fit(train_model_input, train[target].values,batch_size=256, epochs=20, verbose=2, validation_split=0.2, shuffle=True)
    pred_ans3 = model3.predict(test_model_input, batch_size=256)
    print("PNN test MSE",round(mean_squared_error(test[target].values, pred_ans3),4))

    model4 = FNN(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=[1024,512,256], task='regression')
    model4.compile("adam", "mse", metrics=['mse'])
    history4 = model4.fit(train_model_input, train[target].values,batch_size=256, epochs=20, verbose=2, validation_split=0.2, shuffle=True)
    pred_ans4 = model4.predict(test_model_input, batch_size=256)
    print("FNN test MSE", round(mean_squared_error(test[target].values, pred_ans4),4))
