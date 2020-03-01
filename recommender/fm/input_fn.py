import tensorflow as tf

def create_input_fn(features, labels, batch_size=32, num_epochs=1):
    def input_fn():
        print("\n\n***************sango\n\n\n")
        print(features)
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.shuffle(1000).repeat(num_epochs).batch(batch_size)
        a = dataset.make_one_shot_iterator().get_next()
        print("\n\n***************sango\n\n\n",a)
        return dataset.make_one_shot_iterator().get_next()
    return input_fn