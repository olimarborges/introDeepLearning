import numpy as np

def main():
	features = np.load("cars.npz")
	for idx in features:
		print idx, features[idx].shape, features[idx].max(), features[idx].min(), features[idx].mean()

if __name__ == "__main__":
	main()