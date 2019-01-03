import cv2
import imgaug as ia
from imgaug import augmenters as iaa
import random


class ImageGenerator():
	def __init__(self):
		self.im_paths = [
			"/home/ml/Pictures/car.jpg",
			"/home/ml/Pictures/cat.jpg",
			"/home/ml/Pictures/dog.jpg",
			"/home/ml/Pictures/frog.jpg"
		]

	def gen(self):
		index = 0
		while(True):
			path = self.im_paths[index]
			yield cv2.imread(path), path[18:-4]
			index += 1
			index = index % len(self.im_paths)
			if index == 0:
				random.shuffle(self.im_paths)

def main():
	im_gen = ImageGenerator().gen()
	for data, cls in im_gen:
		print cls
		cv2.imshow("frame", data)
		if cv2.waitKey(120) & 0xFF == ord('q'):
			break
	
	cv2.destroyAllWindows()


if __name__ == "__main__":
    if __name__ == "__main__":
    	main()
