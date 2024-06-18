import numpy as np
import matplotlib.pyplot as plt



def TDOA_grid(grid_dimensions):
    #definition goal: return the manually calculated TDOA values for the given grid
    gridTDOA = []
    mic_locs = np.array([[0,0,50],[0,460,50],[460,460,50],[460,0,50],[0,230,80]]) #set mic locations
    
    # take the distance (norm) between the microphones and the grid-locations
    for row in grid_dimensions:
        distances = []
        for loc in mic_locs:
            dist = np.linalg.norm(loc-row)
            distances.append(dist)
        #conversion of distance (cm) to time (s) using speed of sound (m/s)
        times =  np.array(distances)/34300
        
        #calculate the TDOA (time differences) between each microphone pair
        TDOA = []
        for i in range(0,len(times)):
            for j in range(i+1,len(times)):
                TDOA = np.append(TDOA,(times[i]-times[j]))
        gridTDOA = np.concatenate((gridTDOA,TDOA))
    
    #reshape the list to a matrix so each column is corresponding to a different microphone pair
    gridTDOA = np.reshape(gridTDOA,(-1,10))
    return(gridTDOA)

def coordinates_2d(tdoa,min_x=0,max_x=460,min_y=0,max_y=460,grid_resolution=5,finetuning=5):     
    #definition goal: return the coordinates of the car using the measured and calculated TDOA's  
    for i in range(finetuning):
        
        #set the grid dimensions using the given boundaries
        xgrid = np.tile(np.linspace(min_x,max_x,grid_resolution+2)[1:-1],grid_resolution)
        ygrid = np.repeat(np.linspace(min_y,max_y,grid_resolution+2)[1:-1],grid_resolution)
        zgrid = np.repeat(30,grid_resolution**2)
        grid_dimensions = np.stack((xgrid,ygrid,zgrid),axis=1)
        
        #manually calculate the TDOA's for each microphone-pair using the function gridTDOA
        gridTDOA = TDOA_grid(grid_dimensions)
        
        #compare the calculated TDOA with the measure TDOA and find the point where their difference is the smallest
        comparison = np.linalg.norm(gridTDOA - tdoa, axis=1)
        best = grid_dimensions[np.argmin(comparison)]
        
        #To make the algorithm more accurate, once a point has been found, the algorithm will be looped
        #set the dimensions for a new grid to have a higher resolution around the found point (same gridpoints will be used for a smaller area)
        if i<finetuning:
            crop = 2/(grid_resolution**(i+1))
            min_x = best[0] - 460*crop
            max_x = best[0] + 460*crop
            min_y = best[1] - 460*crop
            max_y = best[1] + 460*crop
    return best

xx = 2.2
yy = 1.2

TDOA12 = (np.sqrt((4.6-yy)**2+xx**2)-np.sqrt(xx**2+yy**2))/343
TDOA13 = (np.sqrt((4.6-xx)**2+(4.6-yy)**2)-np.sqrt(xx**2+yy**2))/343
TDOA14 = (np.sqrt((4.6-xx)**2+yy**2)-np.sqrt(xx**2+yy**2))/343
TDOA15 = (np.sqrt(xx**2+(2.3-yy)**2)-np.sqrt(xx**2+yy**2))/343
TDOA23 = TDOA13 - TDOA12
TDOA24 = TDOA14 - TDOA12
TDOA25 = TDOA15 - TDOA12
TDOA34 = TDOA14 - TDOA13
TDOA35 = TDOA15 - TDOA13
TDOA45 = TDOA15 - TDOA14

TDOA = [TDOA12, TDOA13, TDOA14, TDOA15, TDOA23, TDOA24, TDOA25, TDOA34, TDOA35, TDOA45]

x = coordinates_2d(TDOA)

print(x)

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





