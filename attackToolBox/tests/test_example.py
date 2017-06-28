from unittest import TestCase

from attackToolBox import *
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

class Test(TestCase):
	
	def test_init(self):
		atb = attackToolBox(mnist.train.images[:10], mnist.train.labels[:10], n_neighbors=10)
		self.assertEqual(atb.model.get_params()["n_neighbors"], 10)
