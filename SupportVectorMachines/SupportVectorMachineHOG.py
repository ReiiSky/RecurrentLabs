#Python Version 3.5.4

import os, cv2, random
import numpy as np
import pandas as pd


TRAIN_DIR = 'Train/Orginal/'
TRAIN_DIR_ROOT = 'Train/'
TRAIN_RESIZED_DIR = 'Train/Resized/'
TEST_DIR = '../Test/'

ROWS = 128
COLS = 128
CHANNELS = 3
CELL_SIZE = 8

numberOfTestSample = .5

bin_n = 9 # Number of bins 


def hog(img):
	gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
	gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
	mag, ang = cv2.cartToPolar(gx, gy)
	bins = np.int32(bin_n*ang/(2*np.pi))

	mag_cells = []
	bin_cells = []

	for i in range(0,COLS, CELL_SIZE):
		if (i+(CELL_SIZE*2) <= COLS):
			for j in range(0,ROWS, CELL_SIZE):
				if (j+(CELL_SIZE*2) <= ROWS):
					mag_cells.append(mag[j:j + (CELL_SIZE*2), i:i + (CELL_SIZE*2)])
					bin_cells.append(bins[j:j + (CELL_SIZE*2), i:i + (CELL_SIZE*2)])


	hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
	hist = np.hstack(hists)     # hist is a 64 bit vector
	return hist


def convert_2_grayscale(i, file_path):
	img = cv2.imread(file_path, cv2.IMREAD_COLOR)
	filename, file_extension = os.path.splitext(file_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	height = img.shape[0]
	width =  img.shape[1]

	#Height is bigger than width
	if(height > width):
		ratio = float(ROWS)/float(height)
		newWidth = int(float(ratio) * width);
		image = cv2.resize(img, (newWidth, ROWS), interpolation=cv2.INTER_CUBIC)
		blackPadding = int((COLS - newWidth))
		if(int(blackPadding/2)*2 + newWidth == COLS):
			blackPadding = int(float(COLS - newWidth)/2)
			image = cv2.copyMakeBorder(image,0,0,blackPadding,blackPadding,cv2.BORDER_CONSTANT,value=[0, 0, 0])
		if(int(blackPadding/2)*2 + newWidth == COLS-1):
			blackPadding = int(float(COLS - newWidth)/2)
			image = cv2.copyMakeBorder(image,0,0,blackPadding+1,blackPadding,cv2.BORDER_CONSTANT,value=[0, 0, 0])
		return image
	if(height < width):
		ratio = float(COLS)/float(width)
		newHeight = int(float(ratio) * height);
		image =  cv2.resize(img, (COLS, newHeight), interpolation=cv2.INTER_CUBIC)
		blackPadding = int((ROWS - newHeight))

		if(int(blackPadding/2)*2 + newHeight == ROWS):
			blackPadding = int(float(ROWS - newHeight)/2)
			image = cv2.copyMakeBorder(image,blackPadding,blackPadding,0,0,cv2.BORDER_CONSTANT,value=[0, 0, 0])
			
		if(int(blackPadding/2)*2 + newHeight == ROWS-1):
			blackPadding = int(float(ROWS - newHeight)/2)
			image = cv2.copyMakeBorder(image,blackPadding+1,blackPadding,0,0,cv2.BORDER_CONSTANT,value=[0, 0, 0])

		return image
	if(height == width):
		ratio = float(height) /float(ROWS)
		newWidth = float(ratio) * width;
		return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)	


def prep_data(images):
	count = len(images)
	data = []

	for i in range(count):
		data.append([0,np.ndarray((1,64))])

	total_gray = np.zeros(shape=(ROWS,COLS))
	onpointe = 0
	notOnPointe = 0
	other_total_gray = np.zeros(shape=(ROWS,COLS))

	for i, image_file in enumerate(images):
		filename, file_extension = os.path.splitext(image_file)
		if(file_extension == ".jpg"):
			if "pointe" in filename.lower():
				gray_image = convert_2_grayscale(i, image_file)
				total_gray = np.add(total_gray, gray_image.astype(float))
				onpointe += 1
				data[i] = [[1], hog(gray_image)]
			else:
				gray_image = convert_2_grayscale(i, image_file)
				other_total_gray = np.add(other_total_gray, gray_image.astype(float))
				notOnPointe += 1
				data[i] = [[0], hog(gray_image)]

			if i%50 == 0: print('Processed {} of {}'.format(i, count))

	total_gray = total_gray/onpointe		
	other_total_gray = other_total_gray/notOnPointe
	cv2.imwrite(TRAIN_DIR_ROOT + "OnPointeAverage.jpg",total_gray)
	cv2.imwrite(TRAIN_DIR_ROOT + "OtherAverage.jpg",other_total_gray)

	return data

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if ".jpg" in i]

random.shuffle(train_images)

print("Total Number of Images: " + str(len(train_images)))

print()

enPointeData = [image_file for i, image_file in enumerate(train_images) if "pointe" in os.path.splitext(image_file)[0].lower()]
otherData = [image_file for i, image_file in enumerate(train_images) if "pointe" not in os.path.splitext(image_file)[0].lower()]

print("Total Number of En Pointe Images: " + str(len(enPointeData)))
print("Total Number of Other Images: " + str(len(otherData)))

print()


enPointeTrainingData = enPointeData[:int(len(enPointeData) * (1 - numberOfTestSample))]
otherTrainingData = otherData[:int(len(otherData) * (1 - numberOfTestSample))]


print("Total Number of Training En Pointe Images: " + str(len(enPointeTrainingData)))
print("Total Number of Training Other Images: " + str(len(otherTrainingData)))

print()


enPointeTestData = enPointeData[:int(len(enPointeData) * numberOfTestSample)]
otherTestData = otherData[:int(len(otherData) * numberOfTestSample)]


print("Total Number of Test En Pointe Images: " + str(len(enPointeTestData)))
print("Total Number of Test Other Images: " + str(len(otherTestData)))

print()

trainingData = enPointeTrainingData + otherTrainingData
testData = enPointeTestData + otherTestData

random.shuffle(trainingData)
random.shuffle(testData)

print("Processing Training Image")
trainingData = prep_data(trainingData)

svm = cv2.ml.SVM_create()

svm.setGamma(5.383)
svm.setC(2.67)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.train(np.array([np.float32(i) for i in np.array(trainingData)[:,1]]), cv2.ml.ROW_SAMPLE, np.array([i[0] for i in np.array(trainingData)[:,0]]))

print()

print("Processing Test Image")
testData = prep_data(testData)

svm.save('svm_data.dat')

######     Now testing      ########################

result = svm.predict(np.array([np.float32(i) for i in np.array(testData)[:,1]]))


#######   Check Accuracy   ########################
#mask = [int(i) for i in result[1]]==[i for i in responses]

correct = 0

print()

print("Test Perdictions:")
print(np.array([int(i[0]) for i in result[1]]))

print()

print("Test Results:")
print(np.array([i[0] for i in np.array(testData)[:,0]]))


for i in range(len(result[1])):
    if(int(np.array([i[0] for i in np.array(testData)[:,0]])[i]) == int(result[1][i][0])):
        correct += 1

print()

print("Correct: " + str(correct))
print (correct*100.0/result[1].size)

print()
