import cv2
import numpy as np
from matplotlib import pyplot as plt
import math



def VisualationHOG(imageColor, cellSize):
	"""

	"""
	image = cv2.imread('Pointe349.jpg', cv2.IMREAD_COLOR)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	height = image.shape[0]
	width = image.shape[1]

	if(height % cellSize != 0  or width % cellSize != 0):
		raise Exception("Height or Width does not evenly divide into the celll size")

	#print ("Height: " + str(height) + " | Width: " + str(width))

	gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
	gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
	mag, ang = cv2.cartToPolar(gx, gy)

	ang = np.int32(bin_n*ang/(2*np.pi))*((2*np.pi)/16)

	mag_cells = []
	bin_cells = []
	ang_cells = []
	#print("Cell Size:" + str(cellSize))
	#print(ang.size)
	#print(mag.size)
	for i in range(0,width, cellSize):
		for j in range(0,height, cellSize):
			#print("j: " + str(i) + " | j: " + str(j))
			mag_cells.append(mag[i:i + (cellSize), j:j + (cellSize)])
			ang_cells.append(ang[i:i + (cellSize), j:j + (cellSize)])

	#print(len(mag_cells))
	#print(len(ang_cells))

	if(cellSize % 2 == 0):
		center = int(cellSize/2)
	else:
		center = int(cellSize/2) + 1

	maxMagintude = np.amax(mag_cells)
	minMagintude = np.amin(mag_cells)

	maxAngle = np.amax(ang_cells)
	minAngle = np.amin(ang_cells

	#print ("Minimum Magintude: " + str(minMagintude))
	#print ("Maximum Magintude: " + str(maxMagintude))

	#print ("Minimum Angle: " + str(minAngle))
	#print ("Maximum Angle: " + str(maxAngle))
	rangeMangintude = maxMagintude - minMagintude
	intervalMagnitude = rangeMangintude/cellSize

	normalizedMagintude = []

	for indexCell, cell in enumerate(mag_cells):
		normalizedCell = []
		for row in cell:
			normalizedCell.append([int(pixel/(intervalMagnitude*2))for pixel in row])
		normalizedMagintude.append(normalizedCell)

	row = 0
	for indexCell, (a_cell, m_cell) in enumerate(zip(ang_cells, normalizedMagintude)):
		row = int((indexCell) / (height/cellSize))
		col = int(indexCell % (width/cellSize))

		yCenter = (row)*cellSize+center
		xCenter = (col)*cellSize+center
		#print("X: " + str(xCenter) + " | Y: " + str(yCenter))

		for indexRow, (a, m) in enumerate(zip(a_cell, m_cell)):
			print (np.array(a))
			print (np.array(m))

				
			#print ()		

			for index, (a_pixel, m_pixel) in enumerate(zip(a, m)):
				#print("Index:" + str(index) + " | Magnitude: " + str(int(m_pixel)))
				xAngle1 = xCenter + int(math.cos(a_pixel)*m_pixel)
				yAngle1 = yCenter + int(math.sin(a_pixel)*m_pixel)

				xAngle2 = xCenter - int(math.cos(a_pixel)*m_pixel)
				yAngle2 = yCenter - int(math.sin(a_pixel)*m_pixel)

				cv2.line(imageColor,(xAngle2,yAngle2),(xAngle1,yAngle1),(255,255,255), 1)
				#cellCenterX
				#cellCenterY

				#P2.x =  (int)round( + length * cos(angle * CV_PI / 180.0));
				#P2.y =  (int)round(P1.y + length * sin(angle * CV_PI / 180.0));
					

	return img


		

ROWS = 100
COLS = 100
CHANNELS = 3
CELL_SIZE = 10

numberOfTestSample = .20

bin_n = 16 # Number of bins

def hog(img):
	gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
	gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
	mag, ang = cv2.cartToPolar(gx, gy)
	bins = np.int32(bin_n*ang/(2*np.pi))   # quantizing binvalues in (0...16)
    #print(bins)

	mag_cells = []
	bin_cells = []

	for i in range(0,COLS, CELL_SIZE):
		if (i+(CELL_SIZE*2) <= COLS):
			for j in range(0,ROWS, CELL_SIZE):
				if (j+(CELL_SIZE*2) <= ROWS):
					mag_cells.append(mag[j:j + (CELL_SIZE*2)-1, i:i + (CELL_SIZE*2)-1])
					bin_cells.append(bins[j:j + (CELL_SIZE*2)-1, i:i + (CELL_SIZE*2)-1])

	print(np.array(mag_cells[0]))
	print(bin_cells[0])

	#bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    #mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
	hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
	print(hists[0])
	hist = np.hstack(hists)     # hist is a 64 bit vector
	print (hist.size)
	return hist

img = cv2.imread('HipHop53.jpg',cv2.IMREAD_COLOR)
#img = cv2.imread('Pointe349.jpg',cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#img = cv2.imread('Pointe8.jpg',cv2.IMREAD_COLOR)
laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.S(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

cv2.imwrite('HipHop53SobelX.jpg',sobelx)
cv2.imwrite('HipHop53SobelY.jpg',sobely)
#imgVisual = VisualationHOG(img, 8)

#cv2.imwrite('Pointe349.jpg',imgVisual)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()


