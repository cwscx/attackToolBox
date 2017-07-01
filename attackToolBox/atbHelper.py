import matplotlib.pyplot as plt
import numpy as np

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