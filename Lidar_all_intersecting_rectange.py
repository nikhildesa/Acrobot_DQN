# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 00:27:08 2020

@author: nikhil desai
"""


"""
checks

efficiency issue : Tries all side to detect, need to check only two side.
For loop : goes inside each endpoint --> each obstacle ---> checks all side 
"""

#comparing equation with all sides of rectangle
import math
from sklearn.metrics.pairwise import euclidean_distances
import gym
import highway_env
import numpy as np

"""
vehcile stats

LENGTH = 5.0
WIDTH = 2.0
    
"""

config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "absolute": True,
        "order": "sorted",
        #"normalized":True,
    }
}



env = gym.make('highway-v0')
env.configure(config)                       # Update our configuration in the environment
obs = env.reset()

obs = env.step(env.action_space.sample())
kinematics = obs[0]               # kinematics = (row = vehicles)  x  (columns = presence,x,y,vx,vy,cosh,sinh)
kinematics = kinematics[kinematics[:,0] == 1]  # consider only those vehicles whose presence is 1

# My vehicle x,y
center = (kinematics[0,1],kinematics[0,2])  # My vehicle x,y

#Other vehicle x,y
kinematics = kinematics[1:,1:3]


#<------------------------ Check intersection--------------------------->
def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
#<-----------------------------Variable declare---------------------------------------->
        

rectangle = []          # list stores point of each rectangle
all_cars = []           # list of list stores all points of rectangle
x_line = []             # list of x points of circle
y_line = []             # list of y points of circle
rectangle_line = []     # list store sides of each rectangle
sides_of_rectangle = [] # list of list stores all sides of rectangle
distance = [] # To store the length of all rays

#<-----------------------------From the points make rectangle-------------------------->


# dry check

center = (4,5)



#
for i in range(len(kinematics)):
    x = kinematics[i][0]           # center x
    y = kinematics[i][1]           # center y

    car_x_bottom_left = x - 2.5     # width is 5
    car_y_bottom_left = y - 1       # height is 2
    
    car_x_bottom_right = x + 2.5
    car_y_bottom_right = y - 1
    
    car_x_top_right = x + 2.5
    car_y_top_right = y + 1
    
    car_x_top_left = x - 2.5
    car_y_top_left = y + 1
    
    #rectangle = [[car_x_bottom_left,car_y_bottom_left],[car_x_bottom_right,car_y_bottom_right],[car_x_top_right,car_y_top_right],[car_x_top_left,car_y_top_left]]
    
    #all_cars.append(rectangle)  # all_cars has points of all cars
    
    all_cars = [[[6,1],[12,1],[12,2],[6,2]]]
    
#<------------------------------Draw circle points-------------------------------------->
    


rad = 10
deviation_angle = 45
end_angle = 360
for i in range(0,end_angle,deviation_angle):                    # 45 degree deviation (can be changed)
    x_line.append(rad * math.cos(math.radians(i)) + center[0])
    y_line.append(rad * math.sin(math.radians(i)) + center[1])



endpoints = list(zip(x_line,y_line))

#<----------------------------------Making lines of cars---------------------------------------------------->
#endpoint = (4.58,8)

for car in all_cars:

    L1 = line(car[0],car[1])
    L2 = line(car[1],car[2])
    L3 = line(car[2],car[3])
    L4 = line(car[3],car[0])
    rectangle_line = [L1,L2,L3,L4]  
    sides_of_rectangle.append(rectangle_line)
    
#<--------------------------------------- Main loop-------------------------------------------------------->
for endpoint in endpoints:
    smallest_point = 0  
    intersect = []  # stores intersection point
    L = line(list(center),list(endpoint))         # L is (x,y,z)
    
    # rectangle sides loop
    for i in range(len(sides_of_rectangle)):
        for side in sides_of_rectangle[i]:
            R = intersection(L,side)
            if R:
                if(endpoint == (11.071067811865474, -2.0710678118654773)):
                    print(R)
                    if(R[0] >= all_cars[i][0][0] and R[0] <= all_cars[i][2][0] and R[1] >= all_cars[i][0][1] and R[1] <= all_cars[i][2][1] and np.linalg.norm(R) < np.linalg.norm(endpoint)):
                        if(np.linalg.norm(np.array(R)-np.array(endpoint)) < rad):  # checks if they lie in same quadrant
                            intersect.append(R)

    if len(intersect) > 1:
        dist = np.empty([len(intersect),3])
        dist[:,0:2] = np.array([intersect])
        dist[:,2] = euclidean_distances(np.array([center]),dist[:,0:2])
        row = dist[:,2].argmin()
        smallest_point = tuple(dist[row,0:2])
        #print(intersect)
        #print('first if')

    if(len(intersect) == 1):
        smallest_point = intersect[0]
        #print('second if')

    if(len(intersect) == 0):
        smallest_point = endpoint
        #print('third if')

    distance.append(smallest_point)

#print(distance)


# find distance from each point in list distance and center, store it in a form of dictionary.
distance = np.array(distance)


#np.concatenate((k, distance))






















