import sigmoid
import numpy as np
import image2vector as imtovec
import cv2
import normalizeRows
import softmax

matrix = np.array([[0,3,4],
                   [1,6,4]])

image = cv2.imread('testImg.png', cv2.IMREAD_COLOR)
print(image.shape)
print(image)

print(softmax.softmax(image))
