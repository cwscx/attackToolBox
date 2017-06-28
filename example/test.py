from attackToolBox import *
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

atb = attackToolBox(mnist.train.images[:10000], mnist.train.labels[:10000])
print("finish initializing")
atb.polluteByKNN(mnist.test.images[:100], mnist.test.labels[:100], 0.01)
print("finish polluting")
atb.savePollutedImages()
atb.testKNN()

