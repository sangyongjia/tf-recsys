from surprise.prediction_algorithms.random_pred import NormalPredictor
from surprise.prediction_algorithms.baseline_only import BaselineOnly
from surprise.prediction_algorithms.knns import KNNBasic
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.prediction_algorithms.knns import KNNWithZScore
from surprise.prediction_algorithms.knns import KNNBaseline

from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.prediction_algorithms.matrix_factorization import SVDpp
from surprise.prediction_algorithms.matrix_factorization import NMF

from tfcf.models.EnhanceKNNBaseline import EnhanceKNNBaseline
from tfcf.models.EnhanceKNNBaseline import EnhanceKNNBaselineImp
from tfcf.models.EnhanceKNNBaseline import IntegratedModel
from surprise import accuracy



##
def normal_predictor(trainset, testset):
    algo = NormalPredictor()
    # Train the algorithm on the trainset, and predict ratings for the testset
    algo.fit(trainset)
    predictions = algo.test(testset)

    # Then compute RMSE
    #accuracy.rmse(predictions)
    print('normal predictor end, RMSE:{}'.format(accuracy.rmse(predictions)))

def baseline_only(trainset, testset):
    algo = BaselineOnly()
    algo.fit(trainset)
    predictions = algo.test(testset)

    # Then compute RMSE
    #accuracy.rmse(predictions)
    print('baseline only, RMSE:{}'.format(accuracy.rmse(predictions)))

def knn_basic(trainset, testset): ##k可以再区分下user和item，相似性度量方法也可以再调整下
    algo = KNNBasic(sim_options={'name':'MSD','user_based':False,'min_support':1,'shrinkage':100})
    algo.fit(trainset)
    predictions = algo.test(testset)

    # Then compute RMSE
    #accuracy.rmse(predictions)
    print('knn basic, RMSE:{}'.format(accuracy.rmse(predictions)))

def knn_with_means(trainset, testset):
    algo = KNNWithMeans()
    algo.fit(trainset)
    predictions = algo.test(testset)

    # Then compute RMSE
    #accuracy.rmse(predictions)
    print('knn with means , RMSE:{}'.format(accuracy.rmse(predictions)))

def knn_with_zscore(trainset, testset):
    algo = KNNWithZScore()
    algo.fit(trainset)
    predictions = algo.test(testset)

    # Then compute RMSE
    #accuracy.rmse(predictions)
    print('knn with zscore , RMSE:{}'.format(accuracy.rmse(predictions)))

def knn_baseline(trainset, testset):
    algo = KNNBaseline()
    algo.fit(trainset)
    predictions = algo.test(testset)

    # Then compute RMSE
    #accuracy.rmse(predictions)
    print('knn baseline , RMSE:{}'.format(accuracy.rmse(predictions)))

  
##(matrix factorization)latent factor based algorithm
def svd(trainset, testset):
    algo = SVD()
    algo.fit(trainset)
    predictions = algo.test(testset)

    # Then compute RMSE
    #accuracy.rmse(predictions)
    print('svd, RMSE:{}'.format(accuracy.rmse(predictions)))
    
def svdpp(trainset, testset):
    algo = SVDpp()
    algo.fit(trainset)
    predictions = algo.test(testset)

    # Then compute RMSE
    #accuracy.rmse(predictions)
    print('svdpp, RMSE:{}'.format(accuracy.rmse(predictions)))

def nmf(trainset, testset):
    algo = NMF()
    algo.fit(trainset)
    predictions = algo.test(testset)

    # Then compute RMSE
    #accuracy.rmse(predictions)
    print('nmf , RMSE:{}'.format(accuracy.rmse(predictions)))

def slope_one(trainset, testset):
    pass
def co_cluster(trainset, testset):
    pass
def enhance_knn_baseline(trainset, testset):
    algo = EnhanceKNNBaseline()
    algo.fit(trainset)
    predictions = algo.test(testset)
    # Then compute RMSE
    #accuracy.rmse(predictions)
    print('enhance knn baseline , RMSE:{}'.format(accuracy.rmse(predictions)))

def enhance_knn_baseline_imp(trainset, testset):
    algo = EnhanceKNNBaselineImp()
    algo.fit(trainset)
    predictions = algo.test(testset)
    # Then compute RMSE
    #accuracy.rmse(predictions)
    print('enhance knn baseline implicit , RMSE:{}'.format(accuracy.rmse(predictions)))

def integrated_model(trainset, testset):
    algo = IntegratedModel()
    algo.fit(trainset)
    predictions = algo.test(testset)
    # Then compute RMSE
    #accuracy.rmse(predictions)
    print('integrated model , RMSE:{}'.format(accuracy.rmse(predictions)))

    