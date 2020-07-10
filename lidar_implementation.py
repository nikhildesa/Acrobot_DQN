# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 18:34:03 2020

@author: nikhi
"""


import gym
import math
from math import sqrt
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

# Keep running line 38 , 2-3 times it takes any sample action.
obs = env.step(env.action_space.sample())
kinematics = obs[0]               # kinematics = (row = vehicles)  x  (column = presence,x,y,vx,vy,cosh,sinh)
kinematics = kinematics[kinematics[:,0] == 1]  # consider only those vehicles whose presence is 1

center = (kinematics[0,1],kinematics[0,2])  # My vehicle x,y
xpoints = kinematics[1:,1]                  # Other vehicle x
ypoints = kinematics[1:,2]                  # Other vehicle y




r = 0.20                                    # shooting radius (can be changed)
x_line = []
y_line = []
points = ()
for i in range(0,360,5):                    # 5 degree deviation (can be changed)
    x_line.append(r * math.cos(math.radians(i)) + center[0])
    y_line.append(r * math.sin(math.radians(i)) + center[1])


from matplotlib import pyplot as plt
plt.scatter(xpoints,ypoints)
plt.scatter(center[0],center[1],color = 'green')
plt.scatter(x_line,y_line,color = 'Red')
plt.show()


# ray casting

STEP = 0.06 

a = np.zeros([72,8])

a[:,0] = center[0]      # Initial x
a[:,1] = center[1]      # Initial y

angles = np.arange(0,360,5)
a[:,2] = angles            # Angles
            
a[:,3] = np.cos(np.deg2rad(angles))     # cos
a[:,4] = np.sin(np.deg2rad(angles))     # sin
a[:,5] = r * np.cos(np.deg2rad(angles)) + center[0]     # Final x
a[:,6] = r * np.sin(np.deg2rad(angles)) + center[1]     # Final y
a[:,7] = 0  # flag 0  continue, flag 1 stop you have hit an obstruction


xpoints = xpoints.round(decimals = 2)
ypoints = ypoints.round(decimals = 2)
#center = center.round(decimals = 2)

points = list(zip(xpoints,ypoints))
points

""" Check if points lie inside or outside

positions = []

for i in range(0,len(points)):
    distance = sqrt( (center[0]-points[i][0])**2 + (center[1]-points[i][1])**2 )
    if(distance > r):
        positions.append('outside')
    else:
        positions.append('inside')
        break

if('inside' in positions):
    print('ray scanning start')
    # ray scanning start
    
else:
    print('ray scanning not required')
    # rays scanning not required.
""" 
    
length = 5
width = 2
points[0][]


























