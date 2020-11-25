'''
doc: https://matplotlib.org/api/pyplot_api.html
Mathplotlib has been used to make scatterplots to represent the point particles
'''
import matplotlib.pyplot as plt

'''
doc: https://numpy.org/doc/
Numpy has been used to make matrix claculations.
All the arrays used are ndarrys supplied by Numpy and not the ususal python list.
'''
import numpy as np
from numpy import inf

'''
doc: https://docs.python.org/3/library/time.html
Time has been used to get Unix time and to seed Numpy's random number generator
with it.
'''
import time

def getAcceleration(limit, G, M, R):
    '''
    R is the N x 3 position matrix of the form [[x1, y1, z1], [x2, y2, z2], ...]
    M is the N x 1 mass matrix of the form [[m1], [m2], ..]. The reason to
    write masses like this over [m1, m2, ..] is to use the same technique to
    generate the 2d random lists provided by Numpy.
    G is Universal Gravitational constant used for the simulation. I have put it
    inside main to avoid clashes with other code in case the script is imported
    as a module.
    limit is the limiting value of |r2 - r1|.
    A is N x 3 matrix of the acceleration for a particle due to all the other
    particles of the form [[ax1, ay1, az1], [ax2, ay2, az2],..].
    '''
    
    # reference to see how slicing of ndarray's work
    # https://www.w3schools.com/python/numpy_array_slicing.asp
    # X is a ndarray of the form [[x1], [x2], ...]
    X = R[:, 0:1]
    Y = R[:, 1:2]
    Z = R[:, 2:3]

    # ndarray.T returns the tranpose of the ndarray
    # dx is a ndarray that stores the rj - ri values
    # so for a 3x3 case the matrix formed by this operation is of the form
    # [[0, x2 - x1, x3 - x1], [x1 - x2, 0, x3 - x2], [x1 - x3, x2 - x3, 0]]
    Dx = X.T - X
    Dy = Y.T - Y
    Dz = Z.T - Z

    # the limit is very small and hence negligible
    # Dx**2 sqares each of the element
    InvR3 = (Dx**2 + Dy**2 + Dz**2 + limit**2)
    # raising all the elements of InvR3 to the power -3/2
    InvR3 = np.power (InvR3, -1.5)
    # replacing all infinities prodcued by exponentiation process
    InvR3 [InvR3 == inf] = 0
    # For a 3 x 3 case InvR3's final form is
    # [[0, ((x2-x1)^2 + (y2-y1)^2 + (z2 - z1)^2)^-1.5,  ((x3-x1)^2 + (y3-y1)^2 + (z3 - z1)^2)^-1.5]
    #  [((x1-x2)^2 + (y1-y2)^2 + (z1 - z2)^2)^-1.5,   0,((x3-x2)^2 + (y3-y2)^2 + (z3 - z2)^2)^-1.5]
    #  [((x1-x3)^2 + (y1-y3)^2 + (z1 - z3)^2)^-1.5,  ((x2-x3)^2 + (y2-y3)^2 + (z2 - z3)^2)^-1.5, 0]]

    # @ is the matrix multiplication method
    # link explaining it: https://stackoverflow.com/questions/34142485/difference-between-numpy-dot-and-python-3-5-matrix-multiplication
    # The calculation for Ax is of the form
    # G * [[0, (x2-x1)*r21^-1.5*m2, (x3-x1)*r31^-1.5*m3], [(x1-x2)*r12^-1.5*m1, 0, (x3-x2)*r32-1.5*m3]]
    # So the ith-row of Ax is the acceleration of the ith particle due to all the
    #  other particles. The zeroes represent the fact a particle cannot apply
    # force on itself. Hence, the acceleration due to itself is zero.
    Ax = G * (Dx * InvR3) @ M
    Ay = G * (Dy * InvR3) @ M
    Az = G * (Dz * InvR3) @ M
    
    # Creating the final matrix for acceleration of all the particles
    # link for hstack method: https://numpy.org/doc/stable/reference/generated/numpy.hstack.html
    A = np.hstack((Ax, Ay, Az))
    # A is of the form
    # [[ax1, ay1, az1], [ax2, ay2, az2], ...]
    return A
    
def main():
    
    N         = 100    # number of particles
    t         = 0      # current time of the simulation
    endTime   = 3.0   # time at which simulation ends
    dt        = 0.01   # timestep
    limit     = 0.1    # limit length
    G         = 1.0    # value of Gravitational Constant used for the simulation

    # seeding the random number generator with Unix time 
    np.random.seed(int(time.time()))
    
    M   = 100.0*np.ones((N, 1))/N  # total M of particles is 100
    R   = np.random.randn(N, 3)    # randomly selected positions and velocities
    vel = np.random.randn(N, 3)
    
    # Convert to Center-of-M frame
    vel -= np.mean(M * vel,0) / np.mean(M)
    
    # calculate initial gravitational accelerations
    acc = getAcceleration(limit, G, M, R)
    
    # number of timesteps
    Nt = int(np.ceil(endTime/dt))
    
    # save particle orbits for plotting trails
    position_save = np.zeros((N, 3, Nt+1))
    position_save[:,:,0] = R
    t_all = np.arange(Nt + 1)*dt
    
    # figure
    plt.style.use('dark_background')
    fig = plt.figure()
    grid = plt.GridSpec(3, 1, wspace = 0.0, hspace = 0.3)
    
    # simulation loop
    for i in range(Nt):
        # using leapfrog integration. https://en.wikipedia.org/wiki/Leapfrog_integration
        # (1/2) kick
        vel += acc * dt/2.0
        
        # drift
        R += vel * dt
        
        # update accelerations
        acc = getAcceleration(limit, G, M, R)
        
        # (1/2) kick
        vel += acc * dt/2.0
        
        # update time
        t += dt
        
        # save positions for plotting trail
        position_save[:,:,i+1] = R
        
        # clear the current figure
        plt.cla()
        xp = position_save[:, 0, max(i-50,0):i+1]
        yp = position_save[:, 1, max(i-50,0):i+1]
        # plotting the trails
        plt.scatter(xp, yp, s=1, color = 'cornflowerblue')
        # plotting the objects
        plt.scatter(R[:,0], R[:,1], s=10, color ='white')
        plt.pause(0.001)
    
    return 0
    
if __name__== "__main__":
  main()
