import sigmoid.py as sgmd

#sigmoid_derivative(x) = σ′(x) = σ(x) (1−σ(x))
def sigmoid_derivative(x):
    ds = (sgmd.sigmoid_np(x))* (1 - sgmd.sigmoid_np(x))
    return ds