# import numpy as np

# XYZ 2 RGB matrix
# RGB = XYZ * M
# # XYZ = inv(M) * RGB
# M= np.matrix([[2.0416, -0.96924, 0.01344], [-0.56501, 1.876, -0.11836], [-0.34473, 0.04156, 1.0152]])

# MA = np.matrix([[0.8951000, 0.2664000, -0.1614000], [-0.7502000, 1.7135000, 0.0367000], [0.0389000, -0.0685000, 1.0296000]])

# MA = np.matrix([[ 0.9869929, -0.1470543, 0.1599627], [0.4323053, 0.5183603, 0.0492912], [-0.0085287, 0.0400428, 0.9684867]])

# Monitor specific whitepoint measurements
# xyz_whitepoint 	= [113.33, 119.02, 131.07]
xyz_whitepoint_norm  = [95.11, 100.00, 108.43] 

gamma = 2.19
# K_temp = 6563dsa
# lum_cdm2 = 119


# red_max = 59.2
# green_max = 82.8
# blue_max = 34.8