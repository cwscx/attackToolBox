from attackToolBox import *
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

atb = attackToolBox(mnist.train.images[:20000], mnist.train.labels[:20000])
print("finish initializing")
atb.polluteByNN(mnist.test.images[:2000], mnist.test.labels[:2000], 0.01, 50)
print("finish polluting")
atb.savePollutedImages()
atb.testKNN()
atb.showPollutedImage(25)
atb.showNorm()
