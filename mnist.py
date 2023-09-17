import numpy as np
import tensorflow as tf
import tensoeflow_datasets as tfds

mnist_dataset , mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=True)

mnist_train, mnist_test, mnist_dataset['trian'], mnist_dataset['test']

num_validation_sample = 0.1 * mnist_info.splits['train'].num_example

num_validation_sample = tf.cast(num_validation_sample, tf.int64)


num_test_sample = mnist_info.splits['test'].num_examples
num_test_sample = tf.cast(num_test_sample, tf.int64)

def scale(image, lable):
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image, lable

scaled_train_validation_data = mnist_train.map(scale)
test_data = mnist_test.map(scale)

BUFFER_SIZE = 10000

shuffled_train_validation_data = scaled_train_validation_data.shuffle(BUFFER_SIZE)

validation_data = shuffled_train_validation_data.take(num_validation_sample)
train_data = shuffled_train_validation_data.skip(num_validation_sample)

BATCH_SIZE = 100

train_data =train_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(num_validation_sample)
test_data = test_data.batch(num_test_sample)

validation_inputs ,validation_targets = next(iter(validation_data))