import numpy as np
import matplotlib.pyplot as plt

def coordinate_2d(D12, D13, D14):

    D23 = D13 - D12
    D24 = D14 - D12
    D34 = D14 - D13

    # Calculate 2D coordinates based on TDOA measurements
    X1 = np.array([0, 0])
    X2 = np.array([0, 4.8])
    X3 = np.array([4.8, 4.8])
    X4 = np.array([4.8, 0])
    
    norm_X1 = np.linalg.norm(X1)
    norm_X2 = np.linalg.norm(X2)
    norm_X3 = np.linalg.norm(X3)
    norm_X4 = np.linalg.norm(X4)
    
    B = np.array([
        [D12**2 - norm_X1**2 + norm_X2**2],
        [D13**2 - norm_X1**2 + norm_X3**2],
        [D14**2 - norm_X1**2 + norm_X4**2],
        [D23**2 - norm_X2**2 + norm_X3**2],
        [D24**2 - norm_X2**2 + norm_X4**2],
        [D34**2 - norm_X3**2 + norm_X4**2]
    ])
    
    X21 = X2 - X1
    X31 = X3 - X1
    X41 = X4 - X1
    X32 = X3 - X2
    X42 = X4 - X2
    X43 = X4 - X3
    
    A = np.array([
        [2 * X21[0], 2 * X21[1], -2 * D12, 0, 0],
        [2 * X31[0], 2 * X31[1], 0, -2 * D13, 0],
        [2 * X41[0], 2 * X41[1], 0, 0, -2 * D14],
        [2 * X32[0], 2 * X32[1], 0, -2 * D23, 0],
        [2 * X42[0], 2 * X42[1], 0, 0, -2 * D24],
        [2 * X43[0], 2 * X43[1], 0, 0, -2 * D34]
    ])
    A_inv = np.linalg.pinv(A)
    result = np.dot(A_inv, B)
    x = result[0, 0]
    y = result[1, 0]

    return x, y

# Sample D12, D13, D14 values
D12 = (np.sqrt(3.6**2+1.2**2)-np.sqrt(1.2**2+1.2**2))
D13 = (np.sqrt(3.6**2+3.6**2)-np.sqrt(1.2**2+1.2**2))
D14 = 0

x,y = coordinate_2d(D12, D13, D14)

# Range of epsilon values to test
print(D14)
print(D13)
print(D12)

print(x)
print(y)

