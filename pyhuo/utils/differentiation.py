import numpy as np

def tenth_order_x_derivative(data,h):
    """
    10th-order explicit finite difference approximation 
    for the first x-derivative of a function f(x) with a uniform grid spacing of h
    f'(x) ≈ (c1 * f(x - 5h) - c2 * f(x - 4h) + c3 * f(x - 3h) - c4 * f(x - 2h) 
            + c5 * f(x - h) - c5 * f(x + h) + c4 * f(x + 2h) 
            - c3 * f(x + 3h) + c2 * f(x + 4h) - c1 * f(x + 5h)) / h
    """

    c1 = 1/1260
    c2 = 5/504
    c3 = 5 / 84
    c4 = 5 / 21
    c5 = 5 / 6

    # Get the dimensions of the input data
    nz, ny, nx = data.shape

    # Initialize the derivative array with zeros
    derivative = np.zeros_like(data)

    # Calculate the derivative using the 10th-order finite difference scheme
    for k in range(nz):
        for j in range(ny):
            for i in range(5, nx - 5):
                derivative[k, j, i] = (
                    c1 * (data[k, j, i - 5] - data[k, j, i + 5])
                    - c2 * (data[k, j, i - 4] - data[k, j, i + 4])
                    + c3 * (data[k, j, i - 3] - data[k, j, i + 3])
                    - c4 * (data[k, j, i - 2] - data[k, j, i + 2])
                    + c5 * (data[k, j, i - 1] - data[k, j, i + 1])
                ) / h

    return derivative