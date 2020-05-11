import pandas as pd
import numpy as np
import random
import tensorflow as tf
import math

train_data = pd.read_csv("../../dataset/taobao_data/item2item.txt",names=['item','label'])
def generate_batch(batch_size=1000):
    global data_index
    data = tain_data[data_index*batch_size:(data_index+1)*batch_size]
    print data
'''files = "../../dataset/taobao_data/item2item.txt"
column_names = ["item"]
lalbel = [""]
item = [""]

def generate_batch(file_path=files, perform_shuffle=True, repeat_count=1,batch_size=1000):
    def decode_csv(line):
        parsed_line = tf.io.decode_csv(line, record_defaults=label+item, field_delim=',')
        label = parsed_line[1]
        del parsed_line[1]
        features = parsed_line  # Everything but last elements are the features
        d = dict(zip(column_names, features)), label
        return d

    dataset = (tf.data.TextLineDataset(file_path, num_parallel_reads=20)  # Read text file
            .map(decode_csv,num_parallel_calls=20))  # Transform each elem by applying decode_csv fn
    if perform_shuffle:
    # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256*8*8)
    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
    dataset = dataset.batch(batch_size=batch_size)  # Batch size to use
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    dataset = dataset.make_one_shot_iterator()
    features, labels = dataset.get_next()
    return features, labels'''


batch_size = 128
embedding_size =128
#skip_window=1
valid_size = 16
valid_window = 100
valid_example = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64
item_size = 5163068

graph = tf.Graph()
with graph.as_default():
    #输入数据
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size,1])
    valid_dataset = tf.constant(valid_example, dtype=tf.int32)

    #定义变量
    embeddings = tf.Variable(tf.random_uniform([item_size,embedding_size],-1.0,1.0))
    softmax_weights = tf.Variable(tf.truncated_normal([item_size, embedding_size],stddev=1.0/math.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([item_size]))

    #本次训练数据对应的embedding
    embed = tf.nn.embedding_lookup(embeddings, train_dataset)
    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, 
                                                     inputs=embed, labels=train_labels, num_sampled=num_sampled, 
                                                     num_classes=item_size))
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
    
    #归一化
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keep_dims=True))
    normalized_embeddings = embeddings / norm

    #计算已有的。。。。
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

    ##train
    num_steps = 100000
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        average_loss = 0 
        for step in range(num_steps+1):
            batch_data, batch_labels = generate_batch(file_path=files, perform_shuffle=True, repeat_count=1,batch_size=100)
            feed_dict = {train_dataset:batch_data, train_labels:batch_labels}
            _,l = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += 1
            if step % 200 ==0:
                if step > 0:
                    average_loss = average_loss / 2000
                print("average_loss at step: {} is:{}".format(step, average_loss))
                average_loss = 0
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    pass
        final_embeddings = normalized_embeddings.eval()
