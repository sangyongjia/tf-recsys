from tfcf.models.algorithms_package import *
from surprise import Dataset
from surprise.model_selection import train_test_split

# Load the movielens-100k dataset (download it if needed),
data = Dataset.load_builtin('ml-100k')
##Dataset.load_builtin('ml-1m') 这个函数调用结束后返回一个 DatasetAutoFolds的对象，
##该对象含有3个成员变量：
    #raw_ratings:一个tuple三元or四元组：（user id, item id, rating） or （user id, item id, rating, timestamp）
    #ratings_file:打分文件的名称
    #has_been_split:标记是否split过
    #or,如果传入的是一个文件名，则变量是上边三个，如果传入的是一个dataframe则是下边三个：
    #raw_ratings
    #df
    #has_been_split
    
#data = Dataset.load_builtin('ml-1m')


# sample random trainset and testset
# test set is made of 25% of the ratings.
##train_test_split调用结束后返回两个参数：trainset 和 testset
    #trainset是一个Trainset对象，包含的8个主要的成员变量：
        #ur :{u1:[(i1,r1),(i2,r2),...],u2:[(i4,r1),(i6,r2),...]}
        #ir :{i1:[(u1,r1),(u2,r2),...],i2:[(u4,r1),(u6,r2),...]}
        #n_users : 用户数
        #n_items ： 物品数
        #n_ratings： 打分数据的总数
        #rating_scale：打分的范围（1，5）
        #raw2inner_id_users：一个字典，raw id 到 inner id的映射（user id）
        #raw2inner_id_items：一个字典，raw id 到 inner id的映射（item id）
trainset, testset = train_test_split(data, test_size=.25)
'''
normal_predictor(trainset, testset)
baseline_only(trainset, testset)
knn_basic(trainset, testset)
knn_with_means(trainset, testset)
knn_with_zscore(trainset, testset)
'''
#knn_baseline(trainset, testset)

#enhance_knn_baseline(trainset, testset)
#enhance_knn_baseline_imp(trainset, testset)
integrated_model(trainset, testset)
'''
svd(trainset, testset)
svdpp(trainset, testset)
nmf(trainset, testset)
'''