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

