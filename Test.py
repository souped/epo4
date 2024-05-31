import numpy as np
import matplotlib.pyplot as plt

def coordinate_2d(xx,yy):

    D12 = (np.sqrt((4.8-yy)**2+xx**2)-np.sqrt(xx**2+yy**2)) #Calculate the value of D12 using the input x and y
    D13 = (np.sqrt((4.8-xx)**2+(4.8-yy)**2)-np.sqrt(xx**2+yy**2)) #Calculate the value of D13 using the input x and y
    D14 = (np.sqrt((4.8-xx)**2+yy**2)-np.sqrt(xx**2+yy**2)) #Calculate the value of D14 using the input x and y

    # When D12 is equal to 0, the y coordinate is 2.4, to calculate x we used D14 = sqrt((4.8-x)^2+2.4^2)-sqrt(x^2+2.4^)
    if D12 == 0:
        if D14 > 0:
            x = (-13824 + 600*D14**2 + 5*(np.sqrt(624*D14**6 - 43200*D14**4+663552*D14**2)))/(10*(25*D14**2-576))
            y = 2.4
        elif D14 < 0:
            x = (-13824 + 600*D14**2 - 5*(np.sqrt(624*D14**6 - 43200*D14**4+663552*D14**2)))/(10*(25*D14**2-576))
            y = 2.4
        else:
           x = 2.4
           y = 2.4
    # The same goes for D14, to calculate the x coordinate
    elif D14 == 0:
        if D12 > 0:
            x = 2.4
            y = (-13824 + 600*D12**2 + 5*(np.sqrt(624*D12**6 - 43200*D12**4+663552*D12**2)))/(10*(25*D12**2-576))
        elif D12 < 0:
            x = 2.4
            y = (-13824 + 600*D12**2 - 5*(np.sqrt(624*D12**6 - 43200*D12**4+663552*D12**2)))/(10*(25*D12**2-576))
        else:
           x = 2.4
           y = 2.4   
    
    else:
        D23 = D13 - D12
        D24 = D14 - D12
        D34 = D14 - D13

        # Locations of microphones
        X1 = np.array([0, 0])
        X2 = np.array([0, 4.8])
        X3 = np.array([4.8, 4.8])
        X4 = np.array([4.8, 0])
        
        # Calculate 2D coordinates based on TDOA measurements
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

# Testing to see if the output coordinates are the same as the input
x,y = coordinate_2d(2.4,4)
print(x,y)





