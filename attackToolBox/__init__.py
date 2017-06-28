import copy, math, pickle
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

class attackToolBox():
	
	def __init__(self, inputs, labels, **KNNparams):
		self.model = KNeighborsClassifier(**KNNparams)
		self.model.fit(inputs, labels)
		self.original_test_inputs = None
		self.polluted_test_inputs = None
		self.test_labels = None

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



	def testKNN(self):
		print(self.model.score(self.original_test_inputs, self.test_labels))
		print(self.model.score(self.polluted_test_inputs, self.test_labels))

	def savePollutedImages(self, filename="polluted_images_by_KNN"):
		if self.polluted_test_inputs is not None:
			pickle.dump(self.polluted_test_inputs, open(filename, "wb"))

	def loadPollutedImages(self, filename="polluted_images_by_KNN"):
		self.polluted_test_inputs = pickle.load(open(filename, "rb"))