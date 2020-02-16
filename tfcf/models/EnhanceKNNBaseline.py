from surprise.prediction_algorithms.knns import SymmetricAlgo
from surprise.utils import get_rng
import numpy as np
import heapq
import math

##《Factorization meets neighborhood》 公式6
class EnhanceKNNBaseline(SymmetricAlgo):
    def __init__(self, n_factors=100, n_epochs=60, biased=True, init_mean=0,
                    init_std_dev=.1, lr_all=.005,
                    reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None, lr_wij=None,
                    reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None, reg_wij=None, k=300,
                    random_state=None, verbose=False):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.k = k 

        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        #self.lr_pu = lr_pu if lr_pu is not None else lr_all
        #self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.lr_wij = lr_wij if lr_wij is not None else lr_all

        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        #self.reg_pu = reg_pu if reg_pu is not None else reg_all
        #self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.reg_wij = reg_wij if reg_wij is not None else reg_all

        self.random_state = random_state
        self.verbose = verbose

        sim_options = {'name':'cosine','user_based':False}
        SymmetricAlgo.__init__(self,sim_options)

    def fit(self, trainset):

        SymmetricAlgo.fit(self, trainset)
        self.sim = self.compute_similarities()  ##这个位置怎么知道建立的是u-u 的相似度还是i-i 的相似度?,采用的是哪种相似度计算方法？
        self.sgd(trainset)

        return self
        
    def sgd(self, trainset):
        '''
        # user biases
        bu = np.ndarray([])[np.double_t] 
        # item biases
        bi = np.ndarray[np.double_t] 
        # user factors
        pu = np.ndarray[np.double_t, ndim=2] 
        # item factors
        qi = np.ndarray[np.double_t, ndim=2] 
        ## wij
        wij = np.ndarray[np.double_t, ndim=2] 
        '''
        '''
        cdef int u, i, f
        cdef double r, err, dot, puf, qif
        cdef double global_mean = self.trainset.global_mean
        '''
        global_mean = self.trainset.global_mean

        lr_bu = self.lr_bu
        lr_bi = self.lr_bi
        #lr_pu = self.lr_pu
        #lr_qi = self.lr_qi
        lr_wij = self.lr_wij 

        reg_bu = self.reg_bu
        reg_bi = self.reg_bi
        #reg_pu = self.reg_pu
        #reg_qi = self.reg_qi
        reg_wij = self.reg_wij 

        rng = get_rng(self.random_state)   ##get_rng 是啥东东？ 这个函数的引用还没有加。

        bu = np.zeros(trainset.n_users, np.double)
        bi = np.zeros(trainset.n_items, np.double)
        #pu = rng.normal(self.init_mean, self.init_std_dev,(trainset.n_users, self.n_factors))
        #qi = rng.normal(self.init_mean, self.init_std_dev,(trainset.n_items, self.n_factors))
        wij = rng.normal(self.init_mean, self.init_std_dev, (trainset.n_items, trainset.n_items))

        if not self.biased:
            global_mean = 0

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))
            for u, i, r in trainset.all_ratings(): ##函数每次返回一个tuple：(u,i,r)

                # compute current error
                '''
                dot = 0  # <q_i, p_u>
                for f in range(self.n_factors):
                    dot += qi[i, f] * pu[u, f]
                '''
                res = 0
                neighbors = [(self.sim[i,j], j, rr) for (j, rr) in self.trainset.ur[u]] 
                k_neighbors = heapq.nlargest(self.k, neighbors,key=lambda t:t[0])   ##优化成算一遍
                for (sim, j, rrr) in k_neighbors:
                    buj = global_mean + bu[u] + bi[j]
                    res += (rrr - buj) * wij[i][j]
                res = res / math.sqrt(self.k)
            
                err = r - (global_mean + bu[u] + bi[i] + res)    ##global_mean 从trainset中来。

                # update biases
                if self.biased:
                    bu[u] += lr_bu * (err - reg_bu * bu[u])
                    bi[i] += lr_bi * (err - reg_bi * bi[i])
                ##update wij
                #neighbors = [(self.sim[i,j], j, rr) for (j, rr) in self.trainset.ur[u]
                #k_neighbors = heapq.nlargest(self.k, neighbors,key=lambda t:t[0])
                for (sim, j, rrr) in k_neighbors:
                    buj = global_mean + bu[u] + bi[j]
                    wij[i][j] += lr_wij * (err*(rrr-buj)/math.sqrt(self.k) - reg_wij*wij[i][j])

                ##update cij
                '''
                # update factors
                for f in range(self.n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += lr_pu * (err * qif - reg_pu * puf)
                    qi[i, f] += lr_qi * (err * puf - reg_qi * qif)
                '''
        self.bu = bu
        self.bi = bi
        self.wij = wij
        #self.pu = pu
        #self.qi = qi

    def estimate(self, u, i):
        # Should we cythonize this as well?

        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if self.biased:
            est = self.trainset.global_mean

            if known_user:
                est += self.bu[u]

            if known_item:
                est += self.bi[i]

            if known_user and known_item:
                res = 0
                neighbors = [(self.sim[i,j], j, rr) for (j, rr) in self.trainset.ur[u]]
                k_neighbors = heapq.nlargest(self.k, neighbors,key=lambda t:t[0])
                for (sim, j, rrr) in k_neighbors:
                    buj = self.trainset.global_mean + self.bu[u] + self.bi[j]
                    res += (rrr - buj) * self.wij[i][j]
                res = res / math.sqrt(self.k)
                est += res
        else:
            if known_user and known_item:
                est = np.dot(self.qi[i], self.pu[u])
            else:
                #raise PredictionImpossible('User and item are unknown.')
                pass
        return est


##《Factorization meets neighborhood》 公式7
class EnhanceKNNBaselineImp(SymmetricAlgo):
    def __init__(self, n_factors=100, n_epochs=60, biased=True, init_mean=0,
                    init_std_dev=.1, lr_all=.005,
                    reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None, lr_wij=None, lr_cij=None,
                    reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None, reg_wij=None, reg_cij=None, k=300,
                    random_state=None, verbose=False):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.k = k 

        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        #self.lr_pu = lr_pu if lr_pu is not None else lr_all
        #self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.lr_wij = lr_wij if lr_wij is not None else lr_all
        self.lr_cij = lr_cij if lr_cij is not None else lr_all

        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        #self.reg_pu = reg_pu if reg_pu is not None else reg_all
        #self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.reg_wij = reg_wij if reg_wij is not None else reg_all
        self.reg_cij = reg_cij if reg_cij is not None else reg_all

        self.random_state = random_state
        self.verbose = verbose

        sim_options = {'name':'cosine','user_based':False}
        SymmetricAlgo.__init__(self,sim_options)

    def fit(self, trainset):

        SymmetricAlgo.fit(self, trainset)
        self.sim = self.compute_similarities()  ##这个位置怎么知道建立的是u-u 的相似度还是i-i 的相似度?,采用的是哪种相似度计算方法？
        self.sgd(trainset)

        return self
        
    def sgd(self, trainset):
        '''
        # user biases
        bu = np.ndarray([])[np.double_t] 
        # item biases
        bi = np.ndarray[np.double_t] 
        # user factors
        pu = np.ndarray[np.double_t, ndim=2] 
        # item factors
        qi = np.ndarray[np.double_t, ndim=2] 
        ## wij
        wij = np.ndarray[np.double_t, ndim=2] 
        '''
        '''
        cdef int u, i, f
        cdef double r, err, dot, puf, qif
        cdef double global_mean = self.trainset.global_mean
        '''
        global_mean = self.trainset.global_mean

        lr_bu = self.lr_bu
        lr_bi = self.lr_bi
        #lr_pu = self.lr_pu
        #lr_qi = self.lr_qi
        lr_wij = self.lr_wij 
        lr_cij = self.lr_cij

        reg_bu = self.reg_bu
        reg_bi = self.reg_bi
        #reg_pu = self.reg_pu
        #reg_qi = self.reg_qi
        reg_wij = self.reg_wij 
        reg_cij = self.reg_cij

        rng = get_rng(self.random_state)   ##get_rng 是啥东东？ 这个函数的引用还没有加。

        bu = np.zeros(trainset.n_users, np.double)
        bi = np.zeros(trainset.n_items, np.double)
        #pu = rng.normal(self.init_mean, self.init_std_dev,(trainset.n_users, self.n_factors))
        #qi = rng.normal(self.init_mean, self.init_std_dev,(trainset.n_items, self.n_factors))
        wij = rng.normal(self.init_mean, self.init_std_dev, (trainset.n_items, trainset.n_items))
        cij = rng.normal(self.init_mean, self.init_std_dev, (trainset.n_items, trainset.n_items))

        if not self.biased:
            global_mean = 0

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))
            for u, i, r in trainset.all_ratings(): ##函数每次返回一个tuple：(u,i,r)

                # compute current error
                '''
                dot = 0  # <q_i, p_u>
                for f in range(self.n_factors):
                    dot += qi[i, f] * pu[u, f]
                '''
                res = 0
                res_imp = 0
                neighbors = [(self.sim[i,j], j, rr) for (j, rr) in self.trainset.ur[u]] 
                k_neighbors = heapq.nlargest(self.k, neighbors,key=lambda t:t[0])   ##优化成算一遍
                for (sim, j, rrr) in k_neighbors:
                    buj = global_mean + bu[u] + bi[j]
                    res += (rrr - buj) * wij[i][j]
                    res_imp += cij[i][j]
                res = res / math.sqrt(self.k)
                res_imp = res_imp / math.sqrt(self.k)

                err = r - (global_mean + bu[u] + bi[i] + res + res_imp)    ##global_mean 从trainset中来。

                # update biases
                if self.biased:
                    bu[u] += lr_bu * (err - reg_bu * bu[u])
                    bi[i] += lr_bi * (err - reg_bi * bi[i])
                ##update wij
                #neighbors = [(self.sim[i,j], j, rr) for (j, rr) in self.trainset.ur[u]
                #k_neighbors = heapq.nlargest(self.k, neighbors,key=lambda t:t[0])
                for (sim, j, rrr) in k_neighbors:
                    buj = global_mean + bu[u] + bi[j]
                    wij[i][j] += lr_wij * (err*(rrr-buj)/math.sqrt(self.k) - reg_wij*wij[i][j])
                    cij[i][j] += lr_cij * (err/math.sqrt(self.k) - reg_cij*cij[i][j])

                ##update cij
                '''
                # update factors
                for f in range(self.n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += lr_pu * (err * qif - reg_pu * puf)
                    qi[i, f] += lr_qi * (err * puf - reg_qi * qif)
                '''
        self.bu = bu
        self.bi = bi
        self.wij = wij
        self.cij = cij
        #self.pu = pu
        #self.qi = qi

    def estimate(self, u, i):
        # Should we cythonize this as well?

        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if self.biased:
            est = self.trainset.global_mean

            if known_user:
                est += self.bu[u]

            if known_item:
                est += self.bi[i]

            if known_user and known_item:
                res = 0
                res_imp = 0
                neighbors = [(self.sim[i,j], j, rr) for (j, rr) in self.trainset.ur[u]]
                k_neighbors = heapq.nlargest(self.k, neighbors,key=lambda t:t[0])
                for (sim, j, rrr) in k_neighbors:
                    buj = self.trainset.global_mean + self.bu[u] + self.bi[j]
                    res += (rrr - buj) * self.wij[i][j]
                    res_imp += self.cij[i][j]
                res = res / math.sqrt(self.k)
                res_imp = res_imp / math.sqrt(self.k)
                est += res
                est += res_imp
        else:
            if known_user and known_item:
                est = np.dot(self.qi[i], self.pu[u])
            else:
                #raise PredictionImpossible('User and item are unknown.')
                pass
        return est


##《Factorization meets neighborhood》 公式16
class IntegratedModel(SymmetricAlgo):
    def __init__(self, n_factors=100, n_epochs=1, biased=True, init_mean=0,
                    init_std_dev=.1, lr_all=.005,
                    reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None, lr_wij=None, lr_cij=None, lr_yj=None,
                    reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None, reg_wij=None, reg_cij=None, reg_yj=None, k=300,
                    random_state=None, verbose=False):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.k = k 

        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.lr_wij = lr_wij if lr_wij is not None else lr_all
        self.lr_cij = lr_cij if lr_cij is not None else lr_all
        self.lr_yj = lr_yj if lr_yj is not None else lr_all

        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.reg_wij = reg_wij if reg_wij is not None else reg_all
        self.reg_cij = reg_cij if reg_cij is not None else reg_all
        self.reg_yj = reg_yj if reg_yj is not None else reg_all 

        self.random_state = random_state
        self.verbose = verbose

        sim_options = {'name':'cosine','user_based':False}
        SymmetricAlgo.__init__(self,sim_options)

    def fit(self, trainset):

        SymmetricAlgo.fit(self, trainset)
        self.sim = self.compute_similarities()  ##这个位置怎么知道建立的是u-u 的相似度还是i-i 的相似度?,采用的是哪种相似度计算方法？
        self.sgd(trainset)

        return self
        
    def sgd(self, trainset):
        '''
        # user biases
        bu = np.ndarray([])[np.double_t] 
        # item biases
        bi = np.ndarray[np.double_t] 
        # user factors
        pu = np.ndarray[np.double_t, ndim=2] 
        # item factors
        qi = np.ndarray[np.double_t, ndim=2] 
        ## wij
        wij = np.ndarray[np.double_t, ndim=2] 
        '''
        '''
        cdef int u, i, f
        cdef double r, err, dot, puf, qif
        cdef double global_mean = self.trainset.global_mean
        '''
        global_mean = self.trainset.global_mean

        lr_bu = self.lr_bu
        lr_bi = self.lr_bi
        lr_pu = self.lr_pu
        lr_qi = self.lr_qi
        lr_wij = self.lr_wij 
        lr_cij = self.lr_cij
        lr_yj = self.lr_yj
        

        reg_bu = self.reg_bu
        reg_bi = self.reg_bi
        reg_pu = self.reg_pu
        reg_qi = self.reg_qi
        reg_wij = self.reg_wij 
        reg_cij = self.reg_cij
        reg_yj  = self.reg_yj

        rng = get_rng(self.random_state)   ##get_rng 是啥东东？ 这个函数的引用还没有加。

        bu = np.zeros(trainset.n_users, np.double)
        bi = np.zeros(trainset.n_items, np.double)
        pu = rng.normal(self.init_mean, self.init_std_dev,(trainset.n_users, self.n_factors))
        qi = rng.normal(self.init_mean, self.init_std_dev,(trainset.n_items, self.n_factors))
        wij = rng.normal(self.init_mean, self.init_std_dev, (trainset.n_items, trainset.n_items))
        cij = rng.normal(self.init_mean, self.init_std_dev, (trainset.n_items, trainset.n_items))
        yj = rng.normal(self.init_mean, self.init_std_dev, (trainset.n_items, self.n_factors))

        if not self.biased:
            global_mean = 0

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {},rating nums {}".format(current_epoch,trainset.n_ratings))
                #num = 0
            for u, i, r in trainset.all_ratings(): ##函数每次返回一个tuple：(u,i,r)
                #num+=1
                #if num%10000==0:
                #    print(num)
                # compute current error
                '''
                dot = 0  # <q_i, p_u>
                for f in range(self.n_factors):
                    dot += qi[i, f] * pu[u, f]
                '''
                # items rated by u. This is COSTLY
                Iu = [j for (j, _) in trainset.ur[u]]
                sqrt_Iu = np.sqrt(len(Iu))

                # compute user implicit feedback
                u_impl_fdb = np.zeros(self.n_factors, np.double)
                for j in Iu:
                    for f in range(self.n_factors):
                        u_impl_fdb[f] += yj[j, f] / sqrt_Iu
                dot = 0  # <q_i, (p_u + sum_{j in Iu} y_j / sqrt{Iu}>
                for f in range(self.n_factors):
                    dot += qi[i, f] * (pu[u, f] + u_impl_fdb[f])

                res = 0
                res_imp = 0
                neighbors = [(self.sim[i,j], j, rr) for (j, rr) in self.trainset.ur[u]] 
                k_neighbors = heapq.nlargest(self.k, neighbors,key=lambda t:t[0])   ##优化成算一遍
                for (sim, j, rrr) in k_neighbors:
                    buj = global_mean + bu[u] + bi[j]
                    res += (rrr - buj) * wij[i][j]
                    res_imp += cij[i][j]

                res = res / math.sqrt(self.k)
                res_imp = res_imp / math.sqrt(self.k)

                err = r - (global_mean + bu[u] + bi[i] + res + res_imp + dot)    ##global_mean 从trainset中来。

                # update biases
                if self.biased:
                    bu[u] += lr_bu * (err - reg_bu * bu[u])
                    bi[i] += lr_bi * (err - reg_bi * bi[i])
                ##update wij, cij
                for (sim, j, rrr) in k_neighbors:
                    buj = global_mean + bu[u] + bi[j]
                    wij[i][j] += lr_wij * (err*(rrr-buj)/math.sqrt(self.k) - reg_wij*wij[i][j])
                    cij[i][j] += lr_cij * (err/math.sqrt(self.k) - reg_cij*cij[i][j])

                ##update factors pu,qi,yj
                for f in range(self.n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += lr_pu * (err * qif - reg_pu * puf)
                    qi[i, f] += lr_qi * (err * (puf + u_impl_fdb[f]) -
                                         reg_qi * qif)
                    for j in Iu:
                        yj[j, f] += lr_yj * (err * qif / sqrt_Iu - reg_yj * yj[j, f])
                '''
                # update factors
                for f in range(self.n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += lr_pu * (err * qif - reg_pu * puf)
                    qi[i, f] += lr_qi * (err * puf - reg_qi * qif)
                '''
        self.bu = bu
        self.bi = bi
        self.wij = wij
        self.cij = cij
        self.pu = pu
        self.qi = qi
        self.yj = yj

    def estimate(self, u, i):
        # Should we cythonize this as well?

        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if self.biased:
            est = self.trainset.global_mean

            if known_user:
                est += self.bu[u]

            if known_item:
                est += self.bi[i]

            if known_user and known_item:
                res = 0
                res_imp = 0
                neighbors = [(self.sim[i,j], j, rr) for (j, rr) in self.trainset.ur[u]]
                k_neighbors = heapq.nlargest(self.k, neighbors,key=lambda t:t[0])
                for (sim, j, rrr) in k_neighbors:
                    buj = self.trainset.global_mean + self.bu[u] + self.bi[j]
                    res += (rrr - buj) * self.wij[i][j]
                    res_imp += self.cij[i][j]
                res = res / math.sqrt(self.k)
                res_imp = res_imp / math.sqrt(self.k)
                est += res
                est += res_imp

                # items rated by u. This is COSTLY
                Iu = [j for (j, _) in self.trainset.ur[u]]
                sqrt_Iu = np.sqrt(len(Iu))

                # compute user implicit feedback
                u_impl_fdb = np.zeros(self.n_factors, np.double)
                for j in Iu:
                    for f in range(self.n_factors):
                        u_impl_fdb[f] += self.yj[j, f] / sqrt_Iu
                
                dot = 0  # <q_i, (p_u + sum_{j in Iu} y_j / sqrt{Iu}>
                for f in range(self.n_factors):
                    dot += self.qi[i, f] * (self.pu[u, f] + u_impl_fdb[f])
                est += dot
        else:
            if known_user and known_item:
                est = np.dot(self.qi[i], self.pu[u])
            else:
                #raise PredictionImpossible('User and item are unknown.')
                pass
        return est