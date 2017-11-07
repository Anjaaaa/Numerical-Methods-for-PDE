# Arrary and stuff 
import numpy as np
# Linear algebra solvers from scipy
import scipy.linalg as la
# Basic plotting routines from the matplotlib library 
import matplotlib.pyplot as plt


def uExact(x):
    return np.cos(2*np.pi*x)

################################################
################################################
################################################
# Exercise part a
################################################
# Number of equally spaced subintervals
n = [64,32,16,8,4]
color = ['r', 'b', 'g', 'y', 'm']
c = 0
# Error-Array
a_err = np.zeros((5,2,65))
err = 0
for N in n:
    # Mesh size
    h = 1/N #Important! In Python 2 you needed to write 1.0 to prevent integer divsion
    # Define N+1 grid points via linspace which is part of numpy now aliased as np 
    x = np.linspace(0,1,N+1)
#    print(x)
    # Define a (full) matrix filled with 0s.
    A = np.zeros((N+1, N+1))
    
    # Define tridiagonal part of A by for rows 1 to N-1
    for i in range(1, N):
        A[i, i-1] = -1
        A[i, i+1] = -1
        A[i, i] = 2
    # Define right hand side. Instead of iterating we
    # use a vectorized variant to evaluate f on all grid points
    # Look out for the right h factors! 
    F = h**2*(2*np.pi)**2*np.cos(2*np.pi*x)
    
    # Note that F[0] and F[N] are also filled!
    # Left boundary
    A[0,0] = 1 #h**2?
    F[0] = 1 #h**2?

    # Right boundary
    A[N,N] = 1 #h**2?
    F[N] = 1 #h**2?

#    print(A)

    # Solve
    U = la.solve(A, F)
    plt.plot(x, U, "x-", color = color[c], label = str(N))
    c += 1
    
    # Calculate the error
    # a_err[err,0,j] is the position
    # a_err[err,1,j] is the error
    for j in range(0,N+1):
        a_err[err,0,j] = x[j]
        a_err[err,1,j] = np.abs(U[j] - uExact(x[j]))
    err += 1

# exact solution
xExact = np.linspace(0,1,1000)
plt.plot(xExact, uExact(xExact), 'k', label = 'exact')
plt.legend(loc='best')
plt.show()
plt.clf()

################################################
################################################
################################################
# Exercise part b
################################################

c = 0
# Error-Array
b_err = np.zeros((5,2,65))
err = 0

for N in n:
    # Mesh size
    h = 1/N #Important! In Python 2 you needed to write 1.0 to prevent integer divsion
    # Define N+1 grid points via linspace which is part of numpy now aliased as np 
    x = np.linspace(0,1,N+1)
#    print(x)
    # Define a (full) matrix filled with 0s.
    A = np.zeros((N+1, N+1))
    
    # Define tridiagonal part of A by for rows 1 to N-1
    for i in range(1, N):
        A[i, i-1] = -1
        A[i, i+1] = -1
        A[i, i] = 2
    # Define right hand side. Instead of iterating we
    # use a vectorized variant to evaluate f on all grid points
    # Look out for the right h factors! 
    F = h**2*(2*np.pi)**2*np.cos(2*np.pi*x)
    
    # Note that F[0] and F[N] are also filled!
    # Left boundary
    A[0,0] = 1
    A[0,1] = -1
    F[0] = 0 #h**2?

    # Right boundary
    A[N,N] = 1 #h**2?
    F[N] = 1 #h**2?

#    print(A)

    # Solve
    U = la.solve(A, F)
    plt.plot(x, U, "x-", color = color[c], label = str(N))
    c += 1
    
    # Calculate the error
    # b_err[err,0,j] is the position
    # b_err[err,1,j] is the error
    for j in range(0,N+1):
        b_err[err,0,j] = x[j]
        b_err[err,1,j] = np.abs(U[j] - uExact(x[j]))
    err += 1


# exact solution
xExact = np.linspace(0,1,1000)
plt.plot(xExact, np.cos(2*np.pi*xExact), 'k', label = 'exact')
plt.legend(loc='best')
plt.show()
plt.clf()

# Accuracy

for i in range(0,len(n)):
    plt.plot(a_err[i,0,range(0,n[i]+1)], a_err[i,1,range(0,n[i]+1)], "-", color = color[i], label = str(n[i]))
    plt.plot(b_err[i,0,range(0,n[i]+1)], b_err[i,1,range(0,n[i]+1)], "--", color = color[i])#, label = str(n[i]))
plt.legend(loc='best')
plt.show()
plt.clf()



################################################
################################################
################################################
# Exercise part c
################################################

c = 0
# Error-Array
b_err = np.zeros((5,2,65))
err = 0

for N in n:
    # Mesh size
    h = 1/N #Important! In Python 2 you needed to write 1.0 to prevent integer divsion
    # Define N+1 grid points via linspace which is part of numpy now aliased as np 
    x = np.linspace(0,1,N+1)
#    print(x)
    # Define a (full) matrix filled with 0s.
    A = np.zeros((N+1, N+1))
    
    # Define tridiagonal part of A by for rows 1 to N-1
    for i in range(1, N):
        A[i, i-1] = -1
        A[i, i+1] = -1
        A[i, i] = 2
    # Define right hand side. Instead of iterating we
    # use a vectorized variant to evaluate f on all grid points
    # Look out for the right h factors! 
    F = h**2*(2*np.pi)**2*np.cos(2*np.pi*x)
    
    # Note that F[0] and F[N] are also filled!
    # Left boundary
    A[0,0] = 1
    A[0,1] = -1
    F[0] = 0

    # Right boundary
    A[N,N] = -1
    A[N,N-1] = 1
    F[N] = 0

#    print(A)

    # Solve
    U = la.solve(A, F)
    plt.plot(x, U, "x-", color = color[c], label = str(N))
    c += 1
    
    # Calculate the error
    # b_err[err,0,j] is the position
    # b_err[err,1,j] is the error
    for j in range(0,N+1):
        b_err[err,0,j] = x[j]
        b_err[err,1,j] = np.abs(U[j] - uExact(x[j]))
    err += 1


# exact solution
xExact = np.linspace(0,1,1000)
plt.plot(xExact, np.cos(2*np.pi*xExact), 'k', label = 'exact')
plt.legend(loc='best')
plt.show()
plt.clf()

# Accuracy

for i in range(0,len(n)):
    plt.plot(a_err[i,0,range(0,n[i]+1)], a_err[i,1,range(0,n[i]+1)], "-", color = color[i], label = str(n[i]))
    plt.plot(b_err[i,0,range(0,n[i]+1)], b_err[i,1,range(0,n[i]+1)], "--", color = color[i])#, label = str(n[i]))
plt.legend(loc='best')
plt.show()