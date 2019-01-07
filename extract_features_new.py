import os
import cv2
import keras.backend as K
from utils_new import load_model
import numpy as np


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def load_convnet():
	model = load_model()
	model.summary()
	return model

def find_paths(path):
	paths = []
	level_a = os.listdir(path)
	for level_name in level_a:
		for image_name in os.listdir(os.path.join(path, level_name)):
			paths += [os.path.join(path, level_name, image_name)]

	return paths

def main():
	
	net = load_convnet()
	get_features = K.function([net.layers[0].input, K.learning_phase()], [net.get_layer("flatten_2").output])

	base_path = 'data/train'
	paths = find_paths(base_path)
	print paths

	dict_features = {}

	for image_path in paths:
		print image_path
		im_data = cv2.imread(image_path)
		fname = image_path[34:]
		print fname
		features = get_features([im_data[np.newaxis,...],0])[0]
		print 'features: ', features
		dict_features[fname] = features
		
	np.savez_compressed("cars_test.npz",**dict_features)

if __name__ == "__main__":
	main()