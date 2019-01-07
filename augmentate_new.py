import os
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
from utils import load_model

def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def find_paths(path):
	paths = []
	level_a = os.listdir(path)
	for level_name in level_a:
		for image_name in os.listdir(os.path.join(path, level_name)):
			paths += [os.path.join(path, level_name, image_name)]

	return paths

#Deixa as imagens no mesmo tamanho
def load_image_data(path):

	image = cv2.imread(path)	
 	if image.shape[0] < image.shape[1]:
 		image = cv2.flip(image, flipCode=1)
	im_data = cv2.resize(image, (224,224))
	return im_data

def main():

	aug_seq = iaa.Sequential(
            [
                iaa.Add((-20, 20)),
                iaa.ContrastNormalization((0.8, 1.6)),
                iaa.AddToHueAndSaturation((-21, 21)),
                iaa.SaltAndPepper(p=0.1),
                iaa.Scale({"width":224, "height":"keep-aspect-ratio"}, 1),
                iaa.CropAndPad(
	                percent=(-0.05, 0.1),
	                pad_mode=ia.ALL,
	                pad_cval=(0, 255)
            	)
            ],
            random_order=True)

	ensure_folder('data/DeepLearningFilesPosAug')
	dst_folder = 'data/DeepLearningFilesPosAug'
	base_path = "/home/ml/datasets/DeepLearningFiles"

	paths = find_paths(base_path)

	for im_path in paths:
		#Deixa as imagens originais no mesmo tamanho
		im_data = load_image_data(im_path)
		fname = im_path[41:]
		paste = im_path[36:40]
		ensure_folder('data/DeepLearningFilesPosAug/'+str(paste))
		#print paste

		#Salva a imagem 'original' sem augmentate no diretorio 'dst_folder'
		dst_path = os.path.join(dst_folder, paste, fname)
		cv2.imwrite(dst_path, im_data)

		#Faz o augmentate da imagem original e salva no diretorio 'dst_folder'
		for i in range(10):
			im_data_aug = aug_seq.augment_image(im_data)
			fname = im_path[41:-4] + str(i) + '.jpg'
			#print('Fname Aug: ' + fname)
			# cv2.imshow("frame", im_data_aug)
			dst_path = os.path.join(dst_folder, paste, fname)
			#print dst_path
			cv2.imwrite(dst_path, im_data_aug)

if __name__ == "__main__":
    if __name__ == "__main__":
    	main()
