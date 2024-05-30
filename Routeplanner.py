import numpy as np
import math

"""
Inputs:
Starting x,y,theta coordinates
End x,y coordinates
Is this inputted as list? Or individual variables?

Output:
List of KITT commands
"""
speed=0.1


class RoutePlanner:
    def __init__(self,max_angle=28.6):
        self.max_angle=speed * np.sin(math.radians(max_angle)) / 33.5
        print("Max angle is ", self.max_angle)

    def angle_check(self,startX,startY,carT,endX,endY):
        dirvec=[endX - startX,endY - startY]  # Vector which points from the starting to the endpoint
        cardirvec=[carT[0],carT[1]]  # Direction vector of the car
        phi=np.arccos(np.dot(cardirvec,dirvec) / (np.linalg.norm(cardirvec) * np.linalg.norm(dirvec)))
        if endY <= startY: phi=-phi

        dtheta=speed * np.sin(phi) / 33.5
        print(dtheta)
        if np.abs(dtheta) > self.max_angle:
            print('Exceeding max angle!')


        # if phi >= max_angle:
        #     pass

    def det_rotation(self,cardirection,phi=None):
        """determines a direction vector derived from phi"""
        # determine rotation matrix
        dtheta=speed * np.sin(phi) / 33.5
        rotation_matrix=np.array([[np.cos(dtheta),-np.sin(dtheta)],[np.sin(dtheta),np.cos(dtheta)]])

        direction=np.matmul(rotation_matrix,cardirection)
        return direction

    def calculate_circle_radius(self, points):
        if len(points) == 3:
            (x1, y1), (x2, y2), (x3, y3) = points

        elif len(points) == 2:
            (x1, y1), (x2, y2) = points
            distance


rp=RoutePlanner()
rp.angle_check(0,0,[1,0],215,200)
