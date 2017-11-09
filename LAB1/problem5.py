# Arrary and stuff 
import numpy as np
# Linear algebra solvers from scipy
import scipy.linalg as la
# Basic plotting routines from the matplotlib library 
import matplotlib.pyplot as plt


def uExact(x, eps):
    return x - (np.exp((x-1)/eps) - np.exp(-1/eps)) / (1-np.exp(-1/eps))

eps = 1
b = 1
################################################
################################################
################################################
# FDM from Exercise 3
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
    A_dd = np.zeros((N+1, N+1))
    A_d = np.zeros((N+1, N+1))
    
    # Define tridiagonal part of A by for rows 1 to N-1
    for i in range(1, N):
        A_dd[i, i-1] = 1
        A_dd[i, i+1] = 1
        A_dd[i, i] = -2
        A_d[i, i-1] = -1
        A_d[i, i+1] = +1
        A_d[i, i] = 0

    A = -eps * A_dd + b * h / 2 * A_d
    # Define right hand side. Instead of iterating we
    # use a vectorized variant to evaluate f on all grid points
    # Look out for the right h factors! 
    F = 2 * h**2 / eps / (np.exp(1/eps)-1) * np.exp(x/eps)
    
    # Note that F[0] and F[N] are also filled!
    # Left boundary
    A[0,0] = 1
    F[0] = 0

    # Right boundary
    A[N,N] = 1 
    F[N] = 0

    # Solve
    U = la.solve(A, F)
    plt.plot(x, U, "x-", color = color[c], label = str(N))
    c += 1
    
    # Calculate the error
    # a_err[err,0,j] is the position
    # a_err[err,1,j] is the error
    for j in range(0,N+1):
        a_err[err,0,j] = x[j]
        a_err[err,1,j] = np.abs(U[j] - uExact(x[j], eps))
    err += 1

# exact solution
xExact = np.linspace(0,1,1000)
plt.plot(xExact, uExact(xExact,eps), 'k', label = 'exact')
plt.legend(loc='best')
plt.show()
plt.clf()

################################################
################################################
################################################
# Manufactured solution Exercise 4
################################################

def uExactMS(x):
    return x + np.cos(2*np.pi*x)
# Number of equally spaced subintervals
n = [64,32,16,8,4]
color = ['r', 'b', 'g', 'y', 'm']
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
    A_dd = np.zeros((N+1, N+1))
    A_d = np.zeros((N+1, N+1))
    
    # Define tridiagonal part of A by for rows 1 to N-1
    for i in range(1, N):
        A_dd[i, i-1] = 1
        A_dd[i, i+1] = 1
        A_dd[i, i] = -2
        A_d[i, i-1] = -1
        A_d[i, i+1] = +1
        A_d[i, i] = 0

    A = -eps * A_dd + b * h / 2 * A_d
    # Define right hand side. Instead of iterating we
    # use a vectorized variant to evaluate f on all grid points
    # Look out for the right h factors! 
    F = 2 * h**2 / eps / (np.exp(1/eps)-1) * np.exp(x/eps)
    
    # Note that F[0] and F[N] are also filled!
    # Left boundary
    A[0,0] = -h
    A[0,1] = h
    F[0] = -1

    # Right boundary
    A[N,N] = 1 
    F[N] = 2

    # Solve
    U = la.solve(A, F)
    plt.plot(x, U, "x-", color = color[c], label = str(N))
    c += 1
    
    # Calculate the error
    # a_err[err,0,j] is the position
    # a_err[err,1,j] is the error
    for j in range(0,N+1):
        b_err[err,0,j] = x[j]
        b_err[err,1,j] = np.abs(U[j] - uExactMS(x[j]))
    err += 1

# exact solution
xExact = np.linspace(0,1,1000)
plt.plot(xExact, uExactMS(xExact), 'k', label = 'exact')
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


# errors in maximum norm
a_err_max = np.zeros(len(n))
b_err_max = np.zeros(len(n))
for i in range(0,len(n)):
    a_err_max[i] = max(a_err[i,1,:])
    b_err_max[i] = max(b_err[i,1,:])


# experimental order of convergence
EOC_a = np.zeros(len(n)-1)
EOC_b = np.zeros(len(n)-1)

for i in range(0,len(EOC_a)):
    EOC_a[i] = np.log(a_err_max[i]/a_err_max[i+1])/np.log(2)
    EOC_b[i] = np.log(b_err_max[i]/b_err_max[i+1])/np.log(2)


# EOC
plt.plot(h[:-1], EOC_a, "rx", label = 'a) Dirichlet b.c. with old u(x)')
plt.plot(h[:-1], EOC_b, "bx", label = 'b) mixed b.c. with old u(x)')
plt.legend(loc='best')
plt.show()
plt.clf()

print("Average EOC a: ", np.mean(EOC_a))
print("Average EOC b: ", np.mean(EOC_b))


