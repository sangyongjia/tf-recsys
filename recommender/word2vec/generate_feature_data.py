#merge vector
import pandas as pd
from pandas import DataFrame
import numpy as np
from itertools import combinations
import json
data = pd.read_csv("../../dataset/taobao_data/data_train.csv", dtype=str )
with open("../../dataset/taobao_data/word2id_im.txt") as f:
    word2id = json.load(f)
    #print(word2id)
    print(type(word2id))
    word2id_df = pd.DataFrame.from_dict(word2id, orient='index',columns=['encoderid'])
word2id_df.reset_index(inplace=True)
names = ['e'+str(i) for i in range(128)]
train_data = pd.read_csv("../../dataset/taobao_data/embeddings.txt",index_col=False,sep=' ',names=['encoderid']+names)
data1 = pd.merge(data,word2id_df,left_on=["item"], right_on=["index"])
data2 = pd.merge(data1,train_data,left_on=["encoderid"],right_on=["encoderid"])
data3 = data2[["userid","item","category","buy_flag"]+names]
data3.to_csv("../../dataset/taobao_data/data_train_embedding",index=False)

data_test = pd.read_csv("../../dataset/taobao_data/data_test.csv", dtype=str)
data4 = pd.merge(data_test,word2id_df,left_on=["item"], right_on=["index"])
data5 = pd.merge(data4,train_data,left_on=["encoderid"],right_on=["encoderid"])
data6 = data5[["userid","item","category","buy_flag"]+names]
data6.to_csv("../../dataset/taobao_data/data_test_embedding",index=False)

