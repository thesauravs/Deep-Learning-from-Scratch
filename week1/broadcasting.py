import numpy as np
            #Apples Beef Eggs Potatoes
A = np.array([[56.0, 0.0, 4.4, 68.0], #Carb
             [1.2, 104.0, 52.0, 8.0], #Protein
             [1.8, 135.0, 99.0, 0.9]]) #Fat
print(A)

cal = A.sum(axis = 0)
carbs = A.sum(axis = 1)
print("Vertical Summation: ")
print(cal)
print("Horizontal Summation: ")
print(carbs.reshape(3,1))
#print(cal.reshape(4,1))
#Matrix A is 3x4 matrix and Matrix Cal is 1x4
#Reshape command is O(1) so including that isn't expensive 
#and ensures size that you needed to be 
print("Percentage calculation using Broadcasting: ")
percentage = A / cal.reshape(1,4) * 100
print(percentage)