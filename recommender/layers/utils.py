import tensorflow as tf
class Hash(tf.keras.layers.Layer):
    """
    hash the input to [0,num_buckets)
    if mask_zero = True,0 or 0.0 will be set to 0,other value will be set in range[1,num_buckets)
    """

    def __init__(self, num_buckets, mask_zero=False, **kwargs):
        self.num_buckets = num_buckets
        self.mask_zero = mask_zero
        super(Hash, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(Hash, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        if x.dtype != tf.string:
            x = tf.as_string(x, )
        try:
            hash_x = tf.string_to_hash_bucket_fast(x, self.num_buckets if not self.mask_zero else self.num_buckets - 1,
                                                    name=None)  # weak hash
        except:
            hash_x = tf.strings.to_hash_bucket_fast(x, self.num_buckets if not self.mask_zero else self.num_buckets - 1,
                                               name=None)  # weak hash
        if self.mask_zero:
            mask_1 = tf.cast(tf.not_equal(x, "0"), 'int64')
            mask_2 = tf.cast(tf.not_equal(x, "0.0"), 'int64')
            mask = mask_1 * mask_2
            hash_x = (hash_x + 1) * mask
        return hash_x

    def compute_mask(self, inputs, mask):
        return None

    def get_config(self, ):
        config = {'num_buckets': self.num_buckets, 'mask_zero': self.mask_zero}
        base_config = super(Hash, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class NoMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NoMask, self).__init__(**kwargs)

    def build(self, input_shape):
        super(NoMask, self).build(input_shape)
    
    def call(self, x, mask=None, **kwargs):
        return x
    
    def compute_mask(self, inputs, mask):
        return None


class Linear(tf.keras.layers.Layer):
    def __init__(self, l2_reg=0.0, mode=0, use_bias=False, **kwargs):
        self.l2_reg = l2_reg
        if mode not in [0, 1, 2]:
            raise ValueError("mode must be 0,1,2")
        self.mode = mode
        self.use_bias = use_bias
        super(Linear,self).__init__(**kwargs)

    def build(self, input_shape):
        if self.use_bias:
            self.bias = self.add_weight(name="linear_bias", shape=(1,), initializer=tf.keras.initializer.Zeros(), trainable=True)
        if self.mode == 1:
            self.kernel = self.add_weight(name="linear_kernel", 
                                            shape=[int(input_shape[-1]),1], 
                                            initializer=tf.keras.initializers.glorot_normal(), 
                                            regularizer=tf.keras.regularizers.l2(self.l2_reg), 
                                            trainable=True)
        elif self.mode == 2:
            self.kernel = self.add_weight(name="linear_kernel",
                                            shape=[int(input_shape[1][-1]),1],    ##在mode==2的情况下，input_shape[0]是sparse特征，input_shape[1]是dense特征
                                            initializer=tf.keras.initializers.glorot_normal(),
                                            regularizer=tf.keras.regularizers.l2(self.l2_reg),
                                            trainable=True)
        super(Linear, self).build(input_shape)
        pass
    def call(self, inputs, **kwargs):
        if self.mode == 0:
            sparse_input = inputs
            linear_logit = reduce_sum(sparse_input, axis=-1, keep_dims=True)    ##之所以sparse部分不用再乘以weight，是因为sparse部分是学习的维度为1的embedding！
        elif self.mode == 1:
            dense_input = inputs
            fc = tf.tensordot(dense_input, self.kernel, axes=(-1,0))
            linear_logit =fc
        else:
            sparse_input, dense_input = inputs
            fc = tf.tensordot(dense_input, self.kernel, axes=(-1,0))
            linear_logit = reduce_sum(sparse_input, axis=-1, keep_dims=False) + fc
        if self.use_bias:
            linear_logit += self.bias
        return linear_logit

    def compute_output_shape(self, input_shape):  #不知道存在的原因
        return (None, 1)

    def compute_mask(self, inputs, mask):        #不知道存在的原因
        return None

    def get_config(self, ):                      #不知道存在的原因
        config = {"mode":self.mode, "l2_reg":self.l2_reg, "use_bias":self.use_bias}
        base_config = super(Linear, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def reduce_sum(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None):
    if tf.__version__ < '2.0.0':
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keep_dims, name=name, reduction_indices=reduction_indices)
    else:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keep_dims, name=name)

def concat_func(inputs, axis=-1, mask=False):
    #print("*****inputs:",inputs)
    if not mask:
        inputs = list(map(NoMask(), inputs))   #NoMask??
    #print("*****888inputs:",inputs)
    if len(inputs) == 1:
        return inputs[0]
    else:
        return tf.keras.layers.Concatenate(axis=axis)(inputs)


class Add(tf.keras.layers.Layer):   ##存在的意义就是啥？ 
    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)
    def build(self, input_shape):
        super(Add, self).build(input_shape)
    def call(self, inputs, **kwargs):
        if not isinstance(inputs, list):
            return inputs
        if len(inputs) == 1:
            return inputs[0]
        if len(inputs) == 0:  ##存在的意义就是这个？
            return tf.constant([[0.0]])   #[0.0]就可以了呀

        return tf.keras.layers.add(inputs)
    

def add_func(inputs):
    return Add()(inputs)

def softmax(logits, dim=-1, name=None):
    if tf.__version__ < '2.0.0':
        return tf.nn.softmax(logits, dim=dim, name=name)
    else:
        return tf.nn.softmax(logits, axis=dim, name=name)
