

batch_size = 128
embedding_size =128
#skip_window=1
valid_size = 16
valid_window = 100
valid_example = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64
item_size = 15000

graph = tf.Graph()
with graph.as_default():
    #输入数据
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size,1])
    valid_dataset = tf.constant(valid_example, dtype=tf.int32)

    #定义变量
    embeddings = tf.Variable(tf.random_uniform([item_size,embedding_size],-1.0,1.0))
    softmax_weight = tf.Variable(tf.truncated_normal([item_size, embedding_size],stddev=1.0/math.sqrt(embedding_size)))
    softmax_baises = tf.Variable(tf.zeros([item_size]))

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
    with tf.Session(grap=grap) as session:
        tf.global_variables_initialiazer().run()
        average_loss = 0 
        for step in range(num_steps+1):
            batch_data, batch_labels = generate_batch()
            feed_dict = {train_dataset:batch_data, train_labels:batch_labels}
            _,l = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += 1
            if step % 2000 ==0:
                if step > 0:
                    average_loss = average_loss / 2000
                print("average_loss at step: {} is:{}".format(step, average_loss))
                average_loss = 0
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    pass
        final_embeddings = normalized_embeddings.eval()