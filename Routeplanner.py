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


# class RoutePlanner:
#     def __init__(self,max_angle=28.6):
#         self.max_angle=speed * np.sin(math.radians(max_angle)) / 33.5
#         print("Max angle is ", self.max_angle)
#
#     def radius_check(self,startX,startY,carT,endX,endY):
#         # dirvec=[endX - startX,endY - startY]  # Vector which points from the starting to the endpoint
#         # cardirvec=[carT[0],carT[1]]  # Direction vector of the car
#         # phi=np.arccos(np.dot(cardirvec,dirvec) / (np.linalg.norm(cardirvec) * np.linalg.norm(dirvec)))
#         # if endY <= startY: phi=-phi
#         #
#         # dtheta=speed * np.sin(phi) / 33.5
#         # print(dtheta)
#         # if np.abs(dtheta) > self.max_angle:
#         #     print('Exceeding max angle!')
#
#
#
#     def det_rotation(self,cardirection,phi=None):
#         """determines a direction vector derived from phi"""
#         # determine rotation matrix
#         dtheta=speed * np.sin(phi) / 33.5
#         rotation_matrix=np.array([[np.cos(dtheta),-np.sin(dtheta)],[np.sin(dtheta),np.cos(dtheta)]])
#
#         direction=np.matmul(rotation_matrix,cardirection)
#         return direction
#
#     def calculate_circle_radius(self, points):
#         if len(points) == 3:
#             (x1, y1), (x2, y2), (x3, y3) = points
#
#         elif len(points) == 2:
#             (x1, y1), (x2, y2) = points
#             # distance
#
# rp=RoutePlanner()
# rp.angle_check(0,0,[1,0],215,200)


class RoutePlanner:
    def __init__(self, max_angle_deg = 25, tirewidth = 5):
        self.min_rad = 33.5/np.sin(math.radians(max_angle_deg)) + tirewidth/2
        self.carloc = [0,0]
        self.dest = [0,0]
        print(self.min_rad)

    def set_carloc(self, x, y):
        self.carloc = [x,y]

    def set_dest(self, x, y):
        self.dest = [x,y]

    def desired_vector(self, carloc, dest):  # carloc = location of car, des = destination coords
        # calculate the vector pointing from the car to the endpoint using current location
        vector = [dest[0]-carloc[0], dest[1]-carloc[1]]
        length = np.sqrt(np.square(vector[0]+np.square(vector[1])))
        direction = np.arccos(vector[0]/vector[1])
        return length, direction





rp = RoutePlanner()
# startX, startY, startT_rad, endX, endY, endT_rad,