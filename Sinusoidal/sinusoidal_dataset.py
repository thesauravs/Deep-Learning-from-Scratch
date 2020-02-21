import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)

# Folder path
folder_path = r'/Users/thesauravs/Documents/GitHub/Deep-Learning-from-Scratch/Sinusoidal'
os.chdir(folder_path)

list_x = []
sin_x = []

for i in range(10000):
    x = np.random.randint(0, 100000) / 10000
    # x is in radians
    if(x not in list_x):
            list_x.append(x)
            sin_x.append(np.sin(x))

# =============================================================================
# To get data points with gaussian distribution 
            
#for i in range(1000):
#    x = np.random.randn()
#    if(x not in list_x):
#        list_x.append(x)
#        sin_x.append(np.sin(x))
# =============================================================================

print(list_x)
print(sin_x)

data = {'x': list_x, 'sin(x)': sin_x}
print(data)

dataFrame = pd.DataFrame(data, columns = ['x', 'sin(x)'])

dataFrame.to_csv('sine.csv', index = False)

plt.scatter(list_x, sin_x, color='g', s=0.05)
plt.xlabel('X')
plt.ylabel('sin(X)')
plt.title('Sine Wave')
plt.show()