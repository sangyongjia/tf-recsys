import numpy as np
import time 

d = 640                           # dimension
nb = 1000000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

import faiss                   # make faiss available
'''
index = faiss.IndexFlatL2(d)   # build the index
print("index.is_trained",index.is_trained)
index.add(xb)                  # add vectors to the index
print("index.ntotal",index.ntotal)

k = 10                          # we want to see 4 nearest neighbors
a=time.time()
D, I = index.search(xb[:5], k) # sanity check
b=time.time()
print("time1:",b-a)
print("I:",I)
print("D:",D)

c = time.time()
D, I = index.search(xq, k)     # actual search
d=time.time()
print("t2:",d-c)
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries
'''

##faster ways
nlist = 100
k = 4
quantizer = faiss.IndexFlatL2(d)  # the other index
index = faiss.IndexIVFFlat(quantizer, d, nlist)
assert not index.is_trained
index.train(xb)
assert index.is_trained

index.add(xb)                  # add may be a bit slower as well
e= time.time()
D, I = index.search(xq[3:4,:], k)     # actual search
f=time.time()
print("time3:",f-e)
print(I[-5:])                  # neighbors of the 5 last queries
index.nprobe = 10              # default nprobe is 1, try a few more
g= time.time()
D, I = index.search(xq[3:13,:], k)
h= time.time()
print("time3:",g-f)
print(I[-5:])    
