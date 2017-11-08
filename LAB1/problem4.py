# Arrary and stuff 
import numpy as np
# Linear algebra solvers from scipy
import scipy.linalg as la
# Basic plotting routines from the matplotlib library 
import matplotlib.pyplot as plt


# package for linear regression
from scipy.optimize import curve_fit


def uExact(x):
    return np.cos(2*np.pi*x)

################################################
################################################
################################################
# Dirichlet boundary conditions
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
#    plt.plot(x, U, "x-", color = color[c], label = str(N))
    c += 1
    
    # Calculate the error
    # a_err[err,0,j] is the position
    # a_err[err,1,j] is the error
    for j in range(0,N+1):
        a_err[err,0,j] = x[j]
        a_err[err,1,j] = np.abs(U[j] - uExact(x[j]))
    err += 1


################################################
################################################
################################################
# Mixed boundary conditions
################################################


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
#    plt.plot(x, U, "x-", color = color[c], label = str(N))
    c += 1
    
    # Calculate the error
    # b_err[err,0,j] is the position
    # b_err[err,1,j] is the error
    for j in range(0,N+1):
        b_err[err,0,j] = x[j]
        b_err[err,1,j] = np.abs(U[j] - uExact(x[j]))
    err += 1

# errors in maximum norm
a_err_max = np.zeros(len(n))
b_err_max = np.zeros(len(n))
for i in range(0,len(n)):
    a_err_max[i] = max(a_err[i,1,:])
    b_err_max[i] = max(b_err[i,1,:])

# create new array because python has a problem with 1/n
h = np.zeros(len(n))
for i in range(0,len(n)):
    h[i] = 1/n[i]

# linear regression to see whether the functions are linear in the loglog plot or rather quadratic
err_log_a = np.log(a_err_max)
err_log_b = np.log(b_err_max)
h_log = np.log(h)

def linear(x, a, b):
    return a*x + b

params_a, cov_a = curve_fit(linear, h_log, err_log_a)
params_b, cov_b = curve_fit(linear, h_log, err_log_b)

# end of linear regression


plt.loglog(h, a_err_max, 'rx', label = 'Dirichlet')
plt.loglog(h, b_err_max, 'bx', label = 'Mixed')
x = np.linspace(0.001,1,1000)
regression_a = np.exp(linear(np.log(x), params_a[0], params_a[1]))
regression_b = np.exp(linear(np.log(x), params_b[0], params_b[1]))

plt.loglog(x, regression_a, 'k--')
plt.loglog(x, regression_b, 'k--')

plt.legend(loc='best')
plt.xlabel('$h$')
plt.ylabel('$\|E(h)\|_\infty$')
plt.xlim(h[0]-0.003, h[len(h)-1]+0.05)
plt.ylim(5*10**(-4), 10)
plt.show()
plt.clf()



################################################
################################################
################################################
# Exercise part c
################################################



# Error-Array
c_err = np.zeros((5,2,65))
err = 0

for N in n:
    # Mesh size
    h_for = 1/N #Important! In Python 2 you needed to write 1.0 to prevent integer divsion
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
    F = h_for**2*(x + np.sin(2*np.pi*x))
    
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
#    plt.plot(x, U, "x-", color = color[c], label = str(N))
    c += 1
    
    # Calculate the error
    # b_err[err,0,j] is the position
    # b_err[err,1,j] is the error
    for j in range(0,N+1):
        c_err[err,0,j] = x[j]
        c_err[err,1,j] = np.abs(U[j] - uExact(x[j]))
    err += 1


# errors in maximum norm
c_err_max = np.zeros(len(n))
for i in range(0,len(n)):
    c_err_max[i] = max(c_err[i,1,:])

# linear regression to see whether the functions are linear in the loglog plot or rather quadratic
err_log_c = np.log(c_err_max)

def linear(x, a, b):
    return a*x + b

params_c, cov_c = curve_fit(linear, h_log, err_log_c)
# end of linear regression


plt.loglog(h, c_err_max, 'rx', label = 'Dirichlet')
x = np.linspace(0.001,1,1000)
regression_c = np.exp(linear(np.log(x), params_c[0], params_c[1]))

plt.loglog(x, regression_c, 'k--')

plt.legend(loc='best')
plt.xlabel('$h$')
plt.ylabel('$\|E(h)\|_\infty$')
plt.xlim(h[0]-0.003, h[len(h)-1]+0.05)
plt.ylim(1,10)
plt.show()
plt.clf()


# Accuracy

for i in range(0,len(n)):
    plt.plot(a_err[i,0,range(0,n[i]+1)], a_err[i,1,range(0,n[i]+1)], "-", color = color[i], label = str(n[i]))
    plt.plot(b_err[i,0,range(0,n[i]+1)], b_err[i,1,range(0,n[i]+1)], "--", color = color[i])#, label = str(n[i]))
plt.legend(loc='best')
plt.show()
