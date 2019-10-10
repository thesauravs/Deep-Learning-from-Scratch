import numpy as np

a = np.random.randn(5)
print(a)
#Neither row nor column
print(a.shape)
#will output same as "a"
print(a.T)

print(np.dot(a, a.T))

#Column Vector
a = np.random.randn(5,1)
print(a)
print(a.T)

print(np.dot(a, a.T))