import os
import cv2
import keras.backend as K
from utils import load_model
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
	#print level_a
	for level_name in level_a:
	#	print level_name
		for image_name in os.listdir(os.path.join(path, level_name)):
			paths += [os.path.join(path, level_name, image_name)]

	return paths

# def load_image_data(path):

# 	image = cv2.imread(path)	
# 	if image.shape[0] < image.shape[1]:
# 		image = cv2.transpose(image)
# 		image = cv2.flip(image, flipCode=1)
# 	im_data = cv2.resize(image, (224,224))
# 	#im_data = image1 / 255. #Comentar esta linha resolveu o problema das imagens ficarem pretas
# 	return im_data

def main():
	
	net = load_convnet()
	get_features = K.function([net.layers[0].input, K.learning_phase()], [net.get_layer("flatten_2").output])

	base_path = 'data/DeepLearningFilesPosAug'
	paths = find_paths(base_path)

	# ensure_folder('data/DeepLearningFilesPosRede')
	# dst_folder = 'data/DeepLearningFilesPosRede'

	#print paths
	dict_features = {}
	inc = 0
	for image_path in paths:
		print image_path
		#im_data = load_image_data(image_path)
		im_data = cv2.imread(image_path)
		fname = image_path[29:]
		#print('fname eh: '+ fname)
		features = get_features([im_data[np.newaxis,...],0])[0]
		inc+= 1
		key = "f" + str(inc)
		print key
		dict_features[key] = features
		#print features.shape, features.max(), features.min(), features.mean()

		#dst_path = os.path.join(dst_folder, fname)
		#print im_data
		#cv2.imwrite(dst_path, im_data)
		#cv2.imshow("frameB",im_data)
		#cv2.waitKey(0)
	np.savez_compressed("cars.npz",**dict_features)

if __name__ == "__main__":
	main()