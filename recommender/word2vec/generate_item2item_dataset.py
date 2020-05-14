import pandas as pd
from pandas import DataFrame
import numpy as np
from itertools import combinations
import json
data = pd.read_csv("../../dataset/taobao_data/data_train.csv", dtype=str )
userid = '1'
items = []
final_items = []
k=0
f = open("../../dataset/taobao_data/item2item.txt","a+")
for i in range(len(data)):
    if userid == data.iloc[i]['userid']:
        items.append(data.iloc[i]['item'])
    else:
        #print(items)
        temp_items = combinations(items,2)
        temp_items_list = list(temp_items)
        for j in range(len(temp_items_list)):
            final_items.append(temp_items_list[j])
            final_items.append((temp_items_list[j][-1],temp_items_list[j][0]))
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
    #k += 1
    #if k==150:
    #    break
f.close()

#names=['item1','item2']
f1=open("../../dataset/taobao_data/word2id.txt","w")
f2=open("../../dataset/taobao_data/id2word.txt","w")
item_data = pd.read_csv("../../dataset/taobao_data/item2item.txt",names=['item','label'], dtype=str)
word2id = {}
id2word = {}
count = 0
for i in range(len(item_data)):
    itemid = item_data.iloc[i]['item']
    if word2id.get(itemid) is not None:
        pass
    else:
        word2id[itemid] = count
        id2word[count] = itemid
        count += 1
    if i % 10000000==0:
        print("i in word2id and id2word :",i)
json.dump(word2id, f1)       
f1.close()
json.dump(id2word, f2)
f2.close()
print("word2id and id2word have been generated")

train_data = pd.read_csv("../../dataset/taobao_data/item2item.txt", dtype=str ,names=['item','label'])
f = open("../../dataset/taobao_data/item2item_encoder.txt","a+")
for i in range(len(train_data)):
    line=""
    line = str(word2id[train_data.iloc[i]['item']]) + ',' + str(word2id[train_data.iloc[i]['label']]) + '\n'
    f.write(line)
    if i % 10000000==0:
        print("i in item2item encoder:",i)
    #if i >10:
    #   break
f.close()
