import copy, math, pickle, sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

def showImage(inputs, labels):
	side = len(inputs[0]) ** 0.5
	if side % 1 == 0:
		side = int(side)
		for i in range(len(inputs)):
			p_image = inputs[i].reshape(side, side)		
			
			print(labels[i])
			plt.imshow(p_image, cmap="gray", vmin=0, vmax=1.0)
			plt.show()

def rescale(x):
	for i in range(len(x)):
		if x[i] < 0:
			x[i] = 0
	maximum = np.amax(x)
	x = x / maximum
	return x

class attackToolBox():
	
	def __init__(self, inputs, labels, **KNNparams):
		self.model = KNeighborsClassifier(**KNNparams)
		self.model.fit(inputs, labels)
		self.attack_model = None
		self.original_test_inputs = None
		self.polluted_test_inputs = None
		self.test_labels = None

	# Pollute by K-Nearest-Neighbor
	def polluteByKNN(self, inputs, labels, learning_rate, **KNNparams):
		self.original_test_inputs = copy.deepcopy(inputs)
		self.test_labels = copy.deepcopy(labels)

		# Model to attack the test inputs & labels
		self.attack_model = KNeighborsClassifier(**KNNparams)
		self.attack_model.fit(inputs, labels)

		view_neighbor_num = math.ceil(len(inputs) / len(labels[0])) + 1

		for i in range(len(inputs)):
			image = inputs[i]
			label = labels[i]

			nearest_neighbors = self.attack_model.kneighbors([image], view_neighbor_num, True)
			indices = nearest_neighbors[1][0]		# Get the indices by decreasing distances

			target_label = None
			target_image = None

			# Find the closest target label
			for j in indices:
				if np.argmax(labels[j]) != np.argmax(label):
					target_label = labels[j]
					target_image = inputs[j]
					break

			# If no different labels are found, skip
			if target_label is not None and target_image is not None:

				# Update until polluted
				# Use threshold to avoid infinite loop
				j = 0
				threshold = math.ceil(1 / learning_rate)
				
				while np.argmax(target_label) != np.argmax(self.model.predict([image])[0]) and j < threshold:
					diff = target_image - image
					image = image + diff * learning_rate

					j+=1

			if self.polluted_test_inputs is None:
				self.polluted_test_inputs = np.array([image])
			else:
				self.polluted_test_inputs = np.append(self.polluted_test_inputs, [image], axis=0)

			if (i * 100 / len(inputs)) % 1 == 0:
				print((i * 100 / len(inputs)), "%...")

	# Pollute by Neural Net
	def polluteByNN(self, inputs, labels, learning_rate, batch_size, **KNNparams):
		self.original_test_inputs = copy.deepcopy(inputs)
		self.test_labels = copy.deepcopy(labels)

		input_d = len(inputs[0])
		label_d = len(labels[0])

		x = tf.placeholder(tf.float32, [None, input_d])
		W = tf.Variable(tf.zeros([input_d, label_d]))
		b = tf.Variable(tf.zeros([label_d]))
		y = tf.nn.softmax(tf.matmul(x, W) + b)
		y_ = tf.placeholder(tf.float32, [None, label_d])

		cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		class_gradient = optimizer.compute_gradients(y, tf.trainable_variables())
		get_gradient = optimizer.compute_gradients(cross_entropy, tf.trainable_variables())
		train_step = optimizer.minimize(cross_entropy)

		sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()

		# Finish training NN model
		for i in range(math.ceil(len(inputs) / batch_size)):
			sess.run(train_step, feed_dict={x: inputs[i * batch_size : (i + 1) * batch_size], y_: labels[i * batch_size : (i + 1) * batch_size]})

		prediction = tf.argmax(y, 1)
		correct = tf.argmax(y_, 1)

		threshold = 1000

		for i in range(len(inputs)):
			image = inputs[i]
			label = labels[i]

			predicted_label = sess.run(prediction, feed_dict={x : image.reshape(1, input_d)})[0]
			correct_label = sess.run(correct, feed_dict={y_: label.reshape(1, label_d)})[0]

			prob_of_correct_label = sess.run(y, feed_dict={x: image.reshape(1, input_d)})[0][correct_label]

			j = 0
			while predicted_label == correct_label and j < threshold:
				j += 1
				
				minimum = sys.maxsize
				minimum_confidence_diff = None
				minimum_gradient_diff = None

				for k in range(label_d):
					if k == predicted_label:
						continue
					else:
						correct_gradient = np.array([ ws[correct_label] for ws in sess.run(class_gradient, feed_dict={x: image.reshape(1, input_d)})[0][1]])
						target_gradient = np.array([ ws[k] for ws in sess.run(class_gradient, feed_dict={x: image.reshape(1, input_d)})[0][1]])

						delta_gradient = target_gradient - correct_gradient
		
						prob_of_k_label = sess.run(y, feed_dict={x: image.reshape(1, input_d)})[0][k]
						confidence_diff = prob_of_k_label - prob_of_correct_label

						n = np.linalg.norm(delta_gradient)

						if n == 0:
							l = sys.maxsize
						else:
							l = abs(confidence_diff) / n

						if l < minimum:
							minimum_confidence_diff = confidence_diff
							minimum_gradient_diff = delta_gradient

				if minimum_gradient_diff is None or minimum_gradient_diff is None:
					continue
				else:
					r = abs(minimum_confidence_diff) / (np.linalg.norm(minimum_gradient_diff) ** 2) * minimum_gradient_diff
					image = image + r
					
					image = rescale(image)
					
				# predicted_label = sess.run(prediction, feed_dict={x: image.reshape(1, input_d)})[0]
				predicted_label =  np.argmax(self.model.predict([image])[0])

			if self.polluted_test_inputs is None:
				self.polluted_test_inputs = np.array([image])
			else:
				self.polluted_test_inputs = np.append(self.polluted_test_inputs, [image], axis=0)

			if (i * 100 / len(inputs)) % 1 == 0:
				print((i * 100 / len(inputs)), "%...")

	def testKNN(self):
		print(self.model.score(self.original_test_inputs, self.test_labels))
		print(self.model.score(self.polluted_test_inputs, self.test_labels))

	def savePollutedImages(self, filename="polluted_images_by_KNN"):
		if self.polluted_test_inputs is not None:
			pickle.dump(self.polluted_test_inputs, open(filename, "wb"))

	def loadPollutedImages(self, filename="polluted_images_by_KNN"):
		self.polluted_test_inputs = pickle.load(open(filename, "rb"))

	def showPollutedImage(self, size=None):
		if size is None or size > len(self.polluted_test_inputs):
			showImage(self.polluted_test_inputs, self.test_labels)
		else:
			showImage(self.polluted_test_inputs[:size], self.test_labels[:size])

