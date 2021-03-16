#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 15:42:34 2021

@author: vejc94
"""

import numpy as np
import matplotlib.pyplot as plt

data = list()

with open('spinbotdata.txt', 'r') as f:
    Lines = f.readlines()
    for line in Lines:
        data.append([float(x) for x in line.split()])
        
q = np.asarray(data[0:3])
n = q.shape[-1]
dq =  np.asarray(data[3:6])
ddq = np.asarray(data[6:9])
u = np.zeros((3*n, 1))
u[0::3,0] =  np.asarray(data[9])
u[1::3,0] =  np.asarray(data[10])
u[2::3,0] =  np.asarray(data[11])


phi = np.zeros((3*n, 3))
phi[0::3, 0] = ddq[0, :]
phi[0::3, -1] = 1
phi[1::3, 1] = 2*dq[2, :]*dq[1, :]*q[2, :] + q[2, :]**2 * ddq[1, :]
phi[2::3, 1] = ddq[2, :] - q[2, :]*dq[1, :]**2

# phi = np.zeros((3*n, 2))
# phi[0::3, 0] = ddq[0, :] + 9.81
# phi[0::3, -1] = 1
# phi[1::3, 1] = 2*dq[2, :]*dq[1, :]*q[2, :] + q[2, :]**2 * ddq[1, :]
# phi[2::3, 1] = ddq[2, :] - q[2, :]*dq[1, :]**2

 
pseudo_inv = np.linalg.inv(phi.T @ phi)
theta = pseudo_inv @ phi.T @ u

u_approx = phi@theta
plt.figure()
plt.title('Least Squares Approximation of $u_1$')
plt.plot(u[0::3])
plt.plot(u_approx[0::3])
plt.legend(['measured', 'estimated'])
plt.xlabel('t')
plt.ylabel('$u_1$ in N')
plt.grid()
plt.savefig("u1.pdf", bbox_inches='tight')


plt.figure()
plt.title('Least Squares Approximation of $u_2$')
plt.plot(u[1::3])
plt.plot(u_approx[1::3])
plt.legend(['measured', 'estimated'])
plt.xlabel('t')
plt.ylabel('$u_2$ in Nm')
plt.grid()
plt.savefig("u2.pdf", bbox_inches='tight')



plt.figure()
plt.title('Least Squares Approximation of $u_3$')
plt.plot(u[2::3])
plt.plot(u_approx[2::3])
plt.legend(['measured', 'estimated'])
plt.grid()
plt.ylabel('$u_3$ in N')
plt.xlabel('t')
plt.savefig("u3.pdf", bbox_inches='tight')
