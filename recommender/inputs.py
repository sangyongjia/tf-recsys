from collections import namedtuple
from collections import OrderedDict
from collections import defaultdict
from copy import copy
from itertools import chain

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding 
from tensorflow.keras.layers import Flatten
from tensorflow.keras.initializer import RandomNormal
from tensorflow.keras.regularizers import l2

from .layers.utils import concat_func
from .layers.utils import Linear
from .layers.utils import add_func

DEFAULT_GROUP_NAME = "default_group"

##就是namedtuple中的属性都是不可变的
class SparseFeat(namedtuple('SparseFeat',['name','vocabulary_size','embedding_dim',
                                         'use_hash','dtype','embedding_name','group_name'])):
    __slots__=()##目的是限制给class添加属性
    ##__new__方法在__init__之前执行，
    ##__new__必须要有返回值，返回实例化出来的实例，这点在自己实现__new__时要特别注意，可以return父类__new__出来的实例，或者直接是object的__new__出来的实例
    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, 
                    dtype="int32", embedding_name=None,group_name=DEFAULT_GROUP_NAME):
        if embedding_name is None:
            embedding_name = name
        if embedding_dim == 'auto':
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25)) ##这个选择由什么玄机？
        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, 
                                                    use_hash, dtype, embedding_name, group_name)  

    def __hash__(self):#放过自己
        return self.name.__hash__()
class VarLenSparseFeat(namedtuple('VarLenSparseFeat':['sparsefeat', 'maxlen', 'combiner', 'length_name', 'weight_name', 'weight_norm'])):
    __slots__()

    def __new__(cls, sparsefeat, maxlen, combiner="mean", length_name=None, weight_name=None, weight_norm=True):
         return super(VarLenSparseFeat,cls).__new__(cls, sparsefeat, maxlen, combiner, length_name, weight_name,weight_norm)
    @property  ##Python内置的@property装饰器就是负责把一个方法变成属性调用的
    def name(self):
        return self.sparsefeat.name
    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size

    def __hash__(self):
        return self.name.__hash__()

class DenseFeat(namedtuple('DenseFeat',['name', 'dimension', 'dtype'])):
    __slots__()

    def __new__(cls, name, dimension=1, dtype="float32"):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)
    def __hash__(self):
        return self.name.__hash__()

def build_input_features(feature_columns, prefix=''):
    input_features = OrderedDict() ##如名字所示，顺序字典。
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            input_features[fc.name] = Input(shape=(1,), name=prefix+fc.name, dtype=fc.dtype)
        elif isinstance(fc, DenseFeat):
            input_features[fc.name] = Input(shape=(1,), name=prefix+fc.name, dtype=fc.dtype)
        elif isinstance(fc, VarLenSparseFeat):
            input_features[fc.name] = Input(shape=(fc.maxlen,), name=prefix+fc.name, dtype=fc.dtype)
            if fc.weight_name is not None:
                input_features[fc.weight_name] = Input(shape=(fc.maxlen, 1), name=prefix+fc.weight_name, dtype="float32")
            if fc.length_name is not None:
                input_features[fc.length_name] = Input(shape=(1,), name=prefix+fc.length_name, dtype="int32")
        else:
            raise TypeError("Invalid feature column type", type(fc)) 
    return input_features
    
def get_feature_names(feature_columns):
    features = build_input_features(feature_columns)
    return list(features)

def create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, init_std, seed, l2_reg, prefix="sparse_", seq_mask_zero=True): #prefix没用了
    sparse_embedding = {feat.embedding_name: Embedding(feat.vocabulary_size, feat.embedding_dim,
                                                        embedding_initializer=RandomNormal(mean=0.0, stddev=init_std, seed=seed),
                                                        embeddings_regularizer=l2(l2_reg)
                                                        ) for feat in sparse_feature_columns}
    if varlen_sparse_feature_columns and len(varlen_sparse_feature_columns) > 0:
        for feat in varlen_sparse_feature_columns:
            sparse_embeddingp[feat.embedding_name] = Embedding(feat.vocabulary_size, feat.embedding_dim,
                                                                embeddings_initializer=RandomNormal(mean=0.0, stddev=init_std, seed=seed),
                                                                embeddings_regularizer=l2(l2_reg),mask_zero=seq_mask_zero)  ##mask_zero的含义不明白！
    return sparse_embedding
    
def create_embedding_matrix(feature_columns, l2_reg, init_std, seed, prefix="", seq_mask_zero=True):
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []
    sparse_emb_dict = create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, init_std, 
                                            seed, l2_reg, prefix=prefix+"sparse", seq_mask_zero=seq_mask_zero)
    return sparse_emb_dict
def embedding_lookup(sparse_embedding_dict, sparse_input_dict, sparse_feature_columns, 
                    return_feat_list=(),mask_feat_list=(), to_list=False):
    group_embedding_dict = defaultdict(list) ##defaultdict的作用是在于，当字典里的key不存在但被查找时，返回的不是keyError而是一个默认值
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if(len(return_feat_list)==0 or feature_name in return_feat_list)
            if fc.use_hash:
                #没涉及到hash
                pass
            else:
                lookup_idx = sparse_input_dict[feature_name] ##这儿的lookup_idx就是Input

            group_embedding_dict[fc.group_name].append(sparse_embedding_dict[embedding_name](lookup_idx))
    if to_list:
        ##return list(chain.from_iterable(group_embedding_dict.values())) ##暂时还未使用到
        pass
    return group_embedding_dict

def get_dense_input(features, feature_columns):
    dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if feature_columns else []
    dense_input_list = []
    for fc in dense_feature_columns:
        dense_input_list.append(features[fc.name])
    return dense_input_list
    
def varlen_embedding_lookup(embedding_dict, sequence_input_dict, varlen_sparse_feature_columns):
    varlen_embedding_vec_dict = {}
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            #lookup_idx = Hash(fc.vocabulary_size, mask_zero=True)(sequence_input_dict[feature_name])
            pass
        else:
            lookup_idx = sequence_input_dict[feature_name]
        varlen_embedding_vec_dict[feature_name] = embedding_dict[embedding_name](lookup_idx)
    return varlen_embedding_vec_dict
    
def get_varlen_pooling_list(embedding_dict, features, varlen_sparse_feature_columns, to_list=False):
    pooling_vec_list = defaultdict(list)
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        combiner = fc.combiner
        feature_length_name = fc.length_name
        if feature_length_name is not None:
            if fc.weight_name is not None:
                seq_input = WeightedSequencelayer(weight_normalization=fc.weight_norm)(
                    [embedding_dict[feature_name],features[feature_length_name],features[fc.weight_name]])
            else:
                seq_input = embedding_dict[feature_name]
            vec = SequencePoolingLayer(combiner, supports_masking=False)(seq_input, features[feature_length_name])
        else:
            if fc.weight_name is not None:
                seq_input = WeightedSequencelayer(weight_normalization=fc.weight_norm, supports_masking=True)(
                    [embedding_dict[feature_name], features[fc.weight_name]])
            else:
                seq_input = embedding_dict[feature_name]
            vec = SequencePoolingLayer(combiner, supports_masking=True)(seq_input)
        pooling_vec_list[fc.group_name].append(vec)
    if to_list:
        return chain.from_iterable(pooling_vec_list.values())
    return pooling_vec_list
    
def mergeDict(a, b):
    c = defaultdict(list)
    for k, v in a.items():
        c[k].extend(v)
    for k, v in b.items():
        c[k].extend(v)
    return c


def input_from_feature_columns(features, feature_columns, l2_reg, init_std, seed, prefix='', 
                                seq_mask_zero=True, support_dense=True, support_group=False):
    sparse_feature_columns = list(filter(lambda x : isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat),feature_columns)) if feature_columns else []

    embedding_matrix_dict = create_embedding_matrix(feature_columns, l2_reg, init_std, seed, prefix, seq_mask_zero=seq_mask_zero)
    group_sparse_embedding_dict = embedding_lookup(embedding_matrix_dict, features, sparse_feature_columns)

    dense_value_list = get_dense_input(features, feature_columns)
    if not support_dense and len(dense_value_list) > 0:
        raise ValueError("DenseFeat is not supported in dnn_feature_columns") ##这个if的判断挺奇怪，不需要的呀。

    sequence_embed_dict = varlen_embedding_lookup(embedding_matrix_dict, features, varlen_sparse_feature_columns)
    group_varlen_sparse_embedding_dict = get_varlen_pooling_list(sequence_embed_dict, features, varlen_sparse_feature_columns)

    group_embedding_dict = mergeDict(group_sparse_embedding_dict, group_varlen_sparse_embedding_dict)
    if not support_group:
        group_embedding_dict = list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict, dense_value_list  ##这里返回的已经是一个向量了。
    

def get_linear_logit(features, feature_columns, units=1, use_bias=False, init_std=0.0001, seed=2020, prefix="linear", l2_reg=0):
    linear_feature_columns = copy(feature_columns)  ##copy
    for i in range(len(linear_feature_columns)):
        if isinstance(linear_feature_columns[i], SparseFeat):
            linear_feature_columns[i] = linear_feature_columns[i]._replace(embedding_dim=1)
        if isinstance(linear_feature_columns[i], VarLenSparseFeat):
            linear_feature_columns[i] = linear_feature_columns[i]._replace(sparsefeat=linear_feature_columns[i].sparsefeat._replace(embedding_dim=1)) 
    linear_emb_list = [input_from_feature_columns(features, linear_feature_columns, l2_reg, 
                                                    init_std, seed, prefix=prefix+str(i))[0] 
                                                    for i in range(units)]
    , dense_input_list = input_from_feature_columns(features, linear_feature_columns, l2_reg, init_std, seed, prefix=prefix)

    linear_logit_list = []
    for i in range(units):
        if len(linear_emb_list[i]) > 0 and len(dense_input_list) > 0:
            sparse_input = concat_func(linear_emb_list[i])
            dense_input = concat_func(dense_input_list)
            linear_logit = Linear(l2_reg, mode=2, use_bias=use_bias)([sparse_input, dense_input])
        elif len(linear_emb_list[i]) > 0:
            sparse_input = concat_func(linear_emb_list[i])
            linear_logit = Linear(l2_reg, mode=0, use_bias=use_bias)(sparse_input)
        elif len(dense_input_list) > 0:
            dense_input = concat_func(dense_input_list)
            linear_logit = Linear(l2_reg, mode=1, use_bias=use_bias)(dense_input)
        else:
            pass
        linear_logit_list.append(linear_logit)

    return concat_func(linear_logit_list)
def combined_dnn_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = Flatten()(concat_func(sparse_embedding_list))
        dense_dnn_input = Flatten()(concat_func(dense_value_list))
        return concat_func(sparse_dnn_input, dense_dnn_input)
    elif len(sparse_embedding_list) > 0:
        return Flatten()(concat_func(sparse_embedding_list))
    elif len(dense_value_list) > 0:
        return Flatten()(concat_func(dense_value_list))
    else:
        raise NotImplementedError
    pass