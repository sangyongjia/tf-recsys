from surprise import AlgoBase
import numpy as np
class FMN(AlgoBase):
    def __init__(self,n_factors=100, n_epoch=20, init_mean=0, init_std_dev=0.1,
                lr_all=0.007, reg_all=0.02, lr_bu=None, lr_bi=None, lr_pu=None,
                lr_qi=None, lr_yj=None, reg_bu=None, reg_bi=None, reg_pu=None,
                reg_qi=None, reg_yj=None, random_state=None, verbose=False):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all   
        self.lr_yj = lr_yj if lr_yj is not None else lr_all

        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.reg_yj = reg_yj if reg_yj is not None else reg_all

        self.random_state = random_state
        self.verbose = verbose
        
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.sgd(trainset)

        return self
    def sgd(self, trainset):
        #user biases
        cdef np.ndarray[np.doublt_t] bu 
        #item biases
        cdef np.ndarray[np.doublt_t] bi
        #user factors
        cdef np.ndarray[np.double_t, ndim=2] pu
        cdef np.ndarray[np.double_t, ndim=2] qi
        cdef np.ndarray[np.double_t, ndim=2] yj 

        cdef int u, i, j, f
        cdef double r, err, dot, puf, qif, sqrt_Iu, _
        cdef double global_mean = self.trainset.global_mean
        cdef np.ndarray[np.double_t] u_impl_fdb
        
        cdef double lr_bu = self.lr_bu
        cdef double lr_bi = self.lr_bi
        cdef double lr_pu = self.lr_pu
        cdef double lr_pi = self.lr_qi
        cdef double lr_yj = self.lr_yj

        cdef double reg_bu = self.reg_bu
        cdef double reg_bi = self.reg_bi
        cdef double reg_pu = self.reg_pu
        cdef double reg_qi = self.reg_qi
        cdef double reg_yj = self.reg_yj

        bu = np.zeros(trainset.n_users, np.double)
        bi = np.zeros(trainset.n_items, np.double)

        rng = get_rng(self.random_state) #没有此方法。

        bu = np.zeros(trainset.n_users, np.double)
        bi = np.zeros(trainset.n_items, np.double)
        pu = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_users, self.n_factors))
        qi = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_items, self.n_factors))
        
        u_impl_fdb = np.zeros(self.n_factors, np.double) ##用户隐式反馈的内容
        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print(" processing epoch {}".format(current_epoch))
            for u, i, r in trainset.all_ratings():

                # items rated by u. This is COSTLY
                Iu = [j for (j, _) in trainset.ur[u]]
                sqrt_Iu = np.sqrt(len(Iu))

                # compute user implicit feedback
                u_impl_fdb = np.zeros(self.n_factors, np.double)
                for j in Iu:
                    for f in range(self.n_factors):
                        u_impl_fdb[f] += yj[j, f] / sqrt_Iu

                # compute current error
                dot = 0  # <q_i, (p_u + sum_{j in Iu} y_j / sqrt{Iu}>
                for f in range(self.n_factors):
                    dot += qi[i, f] * (pu[u, f] + u_impl_fdb[f])

                err = r - (global_mean + bu[u] + bi[i] + dot)

                # update biases
                bu[u] += lr_bu * (err - reg_bu * bu[u])
                bi[i] += lr_bi * (err - reg_bi * bi[i])

                # update factors
                for f in range(self.n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += lr_pu * (err * qif - reg_pu * puf)
                    qi[i, f] += lr_qi * (err * (puf + u_impl_fdb[f]) -
                                         reg_qi * qif)
                    for j in Iu:
                        yj[j, f] += lr_yj * (err * qif / sqrt_Iu -
                                             reg_yj * yj[j, f])
                #update wij and cij
                

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi
        self.yj = yj

    def estimate(self, u, i):

        est = self.trainset.global_mean

        if self.trainset.knows_user(u):
            est += self.bu[u]

        if self.trainset.knows_item(i):
            est += self.bi[i]

        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            Iu = len(self.trainset.ur[u])  # nb of items rated by u
            u_impl_feedback = (sum(self.yj[j] for (j, _)
                               in self.trainset.ur[u]) / np.sqrt(Iu))
            est += np.dot(self.qi[i], self.pu[u] + u_impl_feedback)

        return est