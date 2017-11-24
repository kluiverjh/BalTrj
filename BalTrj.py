''' file = BalTrj.py
Copied from BalTrj06.py on 12Jul17
Numerical simulation of 2D balistic trajectory with
air drag proportional to velocity squared
'''

import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import lagrange

# input partameters
vel = 100.0          # launch velocity in m/s
ad = 45.0            # launch angle above horizion in degrees
tv = 150.0           # projectile terminal velocity in m/s
tli = 30.0           # duration of simulation in seconds
it = 0.5             # calculation time interval in seconds

# input conversion
ar = np.radians(ad)  # launch angle in radians
vx0 = vel*np.cos(ar) # initial X velocity in m/s
vy0 = vel*np.sin(ar) # initial Y velocity in m/s
gc = 9.80665         # acceleration of gravity in m/s2
vg = np.array([0.0, -gc]) # gravity vector
dc = gc/tv**2        # atmospheric drag coefficient

t = np.linspace(0, tli, tli/it + 1) # number of odeint time points

# setup ordinary differential equations (ode's)
def derv(yint, t):   # define 4 differentials
    vv = np.array([yint[0], yint[1]]) # make velocity vector
    vn = np.linalg.norm(vv)  # find velocity magnitude
    vu = vv/vn               # caclulate velocity unit vector
    vd = (-dc*vn**2)*vu      # calculate drag vector
    va = vg + vd     # calculate acceleration vector
    ode0 = va[0]     # acceleration in X direction
    ode1 = va[1]     # acceleration in Y direction
    ode2 = yint[0]   # velocity in X direction
    ode3 = yint[1]   # velocity in Y direction
    return [ode0, ode1, ode2, ode3] # return 4 differentials 

y0 = np.array([vx0, vy0, 0.0, 0.0]) # initial values of differentials

yint = odeint(derv, y0, t) # find solutions to ode's at time points t

# print output header
print('  Time     Vel     Angle   Height Distance')
print('   sec     m/s      deg    meters   meters')

n = 0  # initialize output table line counter
m = 0  # initialize line counter for highest altitude
for x in yint:     # for every solution in yint (every time point in t)
    if yint[n,1]*yint[n+1,1] < 0.0: # if first negative velocity
        m = n      # save line counter as highest altitude
    if x[3] < 0.0: # test for negative altitude Y
        break      # halt if Y position is negative
    vn = (x[0]**2 + x[1]**2)**0.5         # find velocity
    aa = np.degrees(np.arctan(x[1]/x[0])) # angle of flight
    # format line of output table
    sg1 = "{:6.1f} {:9.3f} {:7.2f}".format(t[n], vn, aa)
    sg2 = "{:9.3f} {:8.3f}".format(x[3], x[2])
    print(sg1 + sg2) # print a line of output table
    n += 1           # increment line counter

# determine parameters at maximum altitude of flight
lhv = [yint[m-1,0], yint[m,0], yint[m+1,0]] # 3 X vel near high
lhx = [yint[m-1,2], yint[m,2], yint[m+1,2]] # 3 X pos near high
lhy = [yint[m-1,3], yint[m,3], yint[m+1,3]] # 3 Y pos near high
lht = [t[m-1], t[m], t[m+1]]          # 3 time points near high

# phyx is a polynomial interpolation y = a*x**2 + b*x + c
# find x at which y is maximum where derivative y' = 0.0
# therefore y' = 2*a*x + b = 0.0 and x = -b/2*a
phyx = lagrange(lhx, lhy)    # high point X vs Y poly fit
hx = -phyx[1]/(2.0*phyx[2])  # highest X point
hy = phyx(hx)                # highest Y point

# find time point of maximum altitude
phxt = lagrange(lhx, lht)    # X position vs time poly fit
ht = phxt(hx)                # time of maximum altitude

# find X velocity at maximum altitude ( Y velocity = 0.0 )
phxv = lagrange(lhx, lhv)    # X position vs velocity poly fit
hv = phxv(hx)                # X velocity at maximum altitude

print()
# print summary header
print('  Time     Vel     Angle   Height Distance')
# format and print highest point parameters
sg3 = "{:7.2f} {:8.3f} {:7.2f}".format(ht, hv, 0.0)
sg4 = "{:9.3f} {:8.3f}".format(hy, hx)
print(sg3 + sg4 + ' Highest Point')

# determine flight parameters at impact ( Y position = 0.0 )
# counter n is at line of first negative Y position
lvx = [yint[n-2,0], yint[n-1,0], yint[n,0]] # last 3 X velocities
lvy = [yint[n-2,1], yint[n-1,1], yint[n,1]] # last 3 Y velocities
lgx = [yint[n-2,2], yint[n-1,2], yint[n,2]] # last 3 X positions
lgy = [yint[n-2,3], yint[n-1,3], yint[n,3]] # last 3 Y positions
lgt = [t[n-2], t[n-1], t[n]]                # last 3 time points

pfyx = lagrange(lgy, lgx)   # impact distance polynomial fit
xid = pfyx(0.0)             # impact X position

pfyt = lagrange(lgy, lgt)   # impact time polynomial fit
xit = pfyt(0.0)             # impact time

pfvx = lagrange(lgy, lvx)   # impact X velocity polynomial fit
ivx = pfvx(0.0)             # impact X velocity

pfvy = lagrange(lgy, lvy)   # impact Y velocity polynomial fit
ivy = pfvy(0.0)             # impact Y velocity

iv = (ivx**2 + ivy**2)**0.5 # impact velocity
ia = np.degrees(np.arctan(ivy/ivx)) # impact angle

# format and print impact point parameters
sg5 = "{:7.2f} {:8.3f} {:7.2f}".format(xit, iv, ia)
sg6 = "{:9.3f} {:8.3f}".format(0.0, xid)
print(sg5 + sg6 + ' Impact Point')

'''  EXAMPLE BalTrj.py OUTPUT

  Time     Vel     Angle   Height Distance
   sec     m/s      deg    meters   meters
   0.0   100.000   45.00    0.000    0.000
   0.5    94.534   42.92   33.763   34.980
   1.0    89.424   40.64   64.410   69.246
   1.5    84.660   38.15   92.036  102.851
   2.0    80.242   35.42  116.729  135.841
   2.5    76.172   32.43  138.562  168.259
   3.0    72.457   29.19  157.603  200.143
   3.5    69.106   25.66  173.914  231.528
   4.0    66.133   21.86  187.547  262.443
   4.5    63.554   17.78  198.551  292.915
   5.0    61.383   13.45  206.968  322.968
   5.5    59.635    8.89  212.838  352.622
   6.0    58.320    4.16  216.196  381.892
   6.5    57.443   -0.70  217.075  410.793
   7.0    56.999   -5.60  215.507  439.333
   7.5    56.978  -10.48  211.522  467.522
   8.0    57.360  -15.28  205.149  495.362
   8.5    58.118  -19.91  196.420  522.856
   9.0    59.217  -24.35  185.364  550.004
   9.5    60.620  -28.55  172.014  576.803
  10.0    62.289  -32.49  156.404  603.251
  10.5    64.183  -36.16  138.570  629.342
  11.0    66.267  -39.56  118.549  655.071
  11.5    68.504  -42.70   96.381  680.431
  12.0    70.861  -45.59   72.109  705.415
  12.5    73.311  -48.25   45.776  730.016
  13.0    75.827  -50.70   17.428  754.227

  Time     Vel     Angle   Height Distance
   6.43   57.537    0.00  217.098  406.841 Highest Point
  13.29   77.318  -52.04    0.000  768.180 Impact Point

'''
