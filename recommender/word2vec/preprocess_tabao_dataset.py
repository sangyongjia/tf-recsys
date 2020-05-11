
import pandas as pd
data = pd.read_csv('../../dataset/taobao_data/data_train.csv', dtype={'item': 'Int64'} ,names=["userid","item","category","buy","time"])
from itertools import combinations
userid = 100
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
    #if k==15:
    #    break
f.close()

