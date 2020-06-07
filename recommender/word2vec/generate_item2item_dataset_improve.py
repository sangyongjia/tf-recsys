import pandas as pd
from pandas import DataFrame
import numpy as np
from itertools import combinations
import json

column_names = ["userid", "item", "category","buy_flag"]

data = pd.read_csv("../../dataset/taobao_data/data_train.csv",names=column_names, dtype=str )

userid = '1'
items = []
final_items = []
k=0
f = open("../../dataset/taobao_data/item2item_im.txt","a+")
f1=open("../../dataset/taobao_data/word2id_im.txt","w")
f2=open("../../dataset/taobao_data/id2word_im.txt","w")
word2id = {}
id2word = {}
count = 0

for i in range(len(data)):
    if userid == data.iloc[i]['userid']:
        items.append(data.iloc[i]['item'])
    else:
        #print(items)
        temp_items = combinations(items,2)
        temp_items_list = list(temp_items)
        for j in range(len(temp_items_list)):
            for n in range(len(temp_items_list[j])):
                itemid = temp_items_list[j][n]
                if word2id.get(itemid) is not None:
                    pass
                else:
                    word2id[itemid] = count
                    id2word[count] = itemid
                    count += 1 
            final_items.append((word2id[temp_items_list[j][0]],word2id[temp_items_list[j][1]]))
                          
        lines=""
        for m in range(len(final_items)):
            lines += str(final_items[m][0])+',' + str(final_items[m][1]) +"\n"
        f.write(lines)
        
        '''print(final_items)
        print(len(final_items))
        print("\n")'''
    
        items.clear()
        final_items.clear()
            
        userid = data.iloc[i]['userid']
        #print(userid)
        items.append(data.iloc[i]['item'])
    if i % 100000==0:
        print("i:",i)
    k += 1
    if k==15:
        break
f.close()
json.dump(word2id, f1)       
f1.close()
json.dump(id2word, f2)
f2.close()
