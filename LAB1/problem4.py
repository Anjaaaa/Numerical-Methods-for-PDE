# Arrary and stuff 
import numpy as np
# Linear algebra solvers from scipy
import scipy.linalg as la
# Basic plotting routines from the matplotlib library 
import matplotlib.pyplot as plt

# exact solution
def uExact(x):
    return x + np.cos(2*np.pi*x)

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


plt.loglog(h, a_err_max, 'rx', label = 'Dirichlet')
plt.loglog(h, b_err_max, 'bx', label = 'Mixed')

plt.legend(loc='best')
plt.xlabel('$h$')
plt.ylabel('$\|E(h)\|_\infty$')
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


# experimental order of convergence
EOC_a = np.zeros(len(n)-1)
EOC_b = np.zeros(len(n)-1)
EOC_c = np.zeros(len(n)-1)

for i in range(0,len(EOC_a)):
    EOC_a[i] = np.log(a_err_max[i]/a_err_max[i+1])/np.log(2)
    EOC_b[i] = np.log(b_err_max[i]/b_err_max[i+1])/np.log(2)
    EOC_c[i] = np.log(c_err_max[i]/c_err_max[i+1])/np.log(2)


plt.loglog(h, c_err_max, 'rx', label = 'Dirichlet')
plt.legend(loc='best')
plt.xlabel('$h$')
plt.ylabel('$\|E(h)\|_\infty$')
plt.show()
plt.clf()


# EOC
plt.plot(h[:-1], EOC_a, "rx", label = 'a) Dirichlet b.c. with old u(x)')
plt.plot(h[:-1], EOC_b, "bx", label = 'b) mixed b.c. with old u(x)')
plt.plot(h[:-1], EOC_c, "yx", label = 'c) mixed b.c. with new u(x)')
plt.legend(loc='best')
plt.show()
plt.clf()

print("Average EOC a: ", np.mean(EOC_a))
print("Average EOC b: ", np.mean(EOC_b))
print("Average EOC c: ", np.mean(EOC_c))

