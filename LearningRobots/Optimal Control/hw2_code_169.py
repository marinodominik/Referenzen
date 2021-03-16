import numpy as np
import matplotlib.pyplot as plt


def reward(s, r, R, a=np.zeros(1), H=0):
    r = -(s - r).T @ R @ (s - r) - a.T*H*a
    return r


# 2.1
np.random.seed(0)
T = 50 + 1
At = np.array([[1., 0.1], [0, 1.]])
Bt = np.array([0, 0.1]).reshape((2, 1))
bt = np.array([5., 0]).reshape((2, 1))
Sigma = np.array([[0.01, 0.], [0, 0.01]])
Kt = np.array([5, 0.3]).reshape((1, 2))
Rt1 = np.array([[100000, 0], [0, 0.1]])
Rt2 = np.array([[0.01, 0], [0, 0.1]])
rt1 = np.array([10, 0]).reshape((2, 1))
rt2 = np.array([20, 0]).reshape((2, 1))

kt = 0.3
Ht = 1
n = 20  # number of samples
rewards = np.zeros((n, T))

#%%
experiments_1 = np.zeros((n, T, 2))
actions_1 = np.zeros((n, T, 1))
for j in range(n):
    s0 = np.random.default_rng().standard_normal((2, 1))
    s = np.zeros((T, 2, 1))
    s[0] = s0
    for i in range(T - 1):
        wt = np.random.default_rng().normal(bt.flatten(), np.diag(Sigma)).reshape((2, 1))
        at = -Kt @ s[i] + kt
        ds = At @ s[i] + Bt @ at + wt
        s[i + 1] = ds

        if i < 15:
            rt = rt1
        else:
            rt = rt2

        if i == 14 or i == 40:
            Rt = Rt1

        else:
            Rt = Rt2
        actions_1[j, i] = at
        rw = reward(s[i], rt, Rt, at, Ht)
        rewards[j, i] = rw.flatten()

    rewards[j, -1] = reward(s[-1], rt2, Rt2)
    experiments_1[j] = s.reshape((-1, 2))

mean_r = np.mean(np.sum(rewards, axis=1))
std_r = np.std(np.sum(rewards, axis=1))
print("CONTROLLER 1\nmean = %d \nstandard deviation = %d\n====================" % (mean_r, std_r))

mean1 = experiments_1.mean(axis=0)
std1 = experiments_1.std(axis=0)

mean_a = actions_1.mean(axis=0)
std_a = actions_1.std(axis=0)

ci = 1.96*std1  # 95% Konfidenzintervall
ci_a = 1.96*std_a

# first state
plt.figure()
plt.title('state 1')
plt.plot(np.arange(T), mean1[:, 0], color='b')
plt.fill_between(np.arange(T), mean1[:, 0] + ci[:, 0], mean1[:, 0] - ci[:, 0], alpha=0.5, color='b')
plt.grid()
plt.show()
# second state
plt.figure()
plt.title('state 2')
plt.plot(np.arange(T), mean1[:, 1], color='r')
plt.fill_between(np.arange(T), mean1[:, 1] + ci[:, 1], mean1[:, 1] - ci[:, 1], alpha=0.5, color='r')
plt.grid()
plt.show()
# actions
plt.plot(np.arange(T), mean_a, color='r')
plt.fill_between(np.arange(T), (mean_a + ci_a)[:,0], (mean_a - ci_a)[:,0], alpha=0.5, color='r')
plt.grid()
plt.title('action')
plt.show()

#%%
plt.figure()
experiments_2 = np.zeros((n, T, 2))

for j in range(n):
    s0 = np.random.default_rng().standard_normal((2, 1))
    s = np.zeros((T, 2, 1))
    s[0] = s0
    for i in range(T-1):
        wt = np.random.default_rng().normal(bt.flatten(), np.diag(Sigma)).reshape((2, 1))

        if i < 15:
            s_des = rt1
            rt = rt1
        else:
            s_des = rt2
            rt = rt2

        if i == 14 or i == 40:
            Rt = Rt1
        else:
            Rt = Rt2
        at = Kt @ (s_des - s[i]) + kt
        ds = At @ s[i] + Bt * at + wt
        s[i + 1] = ds
        rw = reward(s[i], rt, Rt, at, Ht)
        rewards[j, i] = rw.flatten()
    rewards[j, -1] = reward(s[-1], rt2, Rt2)
    experiments_2[j] = s.reshape((-1, 2))

mean2 = experiments_2.mean(axis=0)
std2 = experiments_2.std(axis=0)
ci_2 = 1.96 * std2

mean_r = np.mean(np.sum(rewards, axis=1))
std_r = np.std(np.sum(rewards, axis=1))
print("CONTROLLER 2\nmean = %d \nstandard deviation = %d\n=======================" % (mean_r, std_r))

# first state
plt.figure()
plt.title('state 1')
plt.plot(np.arange(T), mean1[:, 0], color='b')
plt.fill_between(np.arange(T), mean1[:, 0] + ci[:, 0], mean1[:, 0] - ci[:, 0], alpha=0.5, color='b')
plt.legend(['$s^{des}_{t} = r_t$'])
plt.plot(np.arange(T), mean2[:, 0], color='r')
plt.fill_between(np.arange(T), mean2[:, 0] + ci_2[:, 0], mean2[:, 0] - ci_2[:, 0], alpha=0.5, color='r')
plt.legend(['$s^{des}_{t} = r_t$', '$s^{des}_{t} = 0$'])
plt.grid()
plt.show()

#%%
# computation of the optimal action for all time steps
# begin at the last time step t=T
Vt = np.zeros((T, Rt1.shape[0], Rt1.shape[1]))
Vt[-1] = Rt2
vt = np.zeros((T, rt1.shape[0], 1))
vt[-1] = Rt2@rt2
# calculating for Value function
for i in range(T-2, 0, -1):
    Mt = Bt@np.linalg.inv(Ht + Bt.T@Vt[i+1]@Bt)@Bt.T@Vt[i+1]@At
    if i < 15:
        rt = rt1
    else:
        rt = rt2

    if i == 14 or i == 40:
        Rt = Rt1
    else:
        Rt = Rt2
    Vt[i] = Rt + (At - Mt).T@Vt[i+1]@At
    vt[i] = Rt@rt + (At - Mt).T@(vt[i+1] - Vt[i+1]@bt)

# calculating states

states = list()
actions = np.zeros((n, T, 1))

for j in range(n):
    s0 = np.random.default_rng().standard_normal((2, 1))
    s = np.zeros((T, 2, 1))
    s[0] = s0
    for i in range(T - 1):
        wt = np.random.default_rng().normal(bt.flatten(), np.diag(Sigma)).reshape((2, 1))
        at = -np.linalg.inv(Ht + Bt.T@Vt[i+1]@Bt)@Bt.T@(Vt[i+1]@(At@s[i] + bt) - vt[i+1])
        ds = At @ s[i] + Bt*at + wt.reshape((2, 1))
        s[i + 1] = ds
        actions[j, i] = at
        if i < 15:
            rt = rt1
        else:
            rt = rt2
        if i == 14 or i == 40:
            Rt = Rt1
        else:
            Rt = Rt2
        rw = reward(s[i], rt, Rt, at, Ht)
        rewards[j, i] = rw.flatten()
    rewards[j, -1] = reward(s[-1], rt2, Rt2)
    states.append(s)


mean_r = np.mean(np.sum(rewards, axis=1))
std_r = np.std(np.sum(rewards, axis=1))
print("CONTROLLER 3\nmean = %d \nstandard deviation = %d\n==============" % (mean_r, std_r))

states = np.asarray(states)
s_mean = states.mean(axis=0).reshape(T, 2)
s_std = states.std(axis=0).reshape(T, 2)
a_mean = actions.mean(axis=0).flatten()
a_std = actions.std(axis=0).flatten()

ci_s = 1.96 * s_std
ci_a = 1.96 * a_std

fig, ax = plt.subplots(3)

# first state
ax[0].plot(np.arange(T), s_mean[:, 0], color='b')
ax[0].fill_between(np.arange(T), s_mean[:, 0] + ci_s[:, 0], s_mean[:, 0] - ci_s[:, 0], alpha=0.5, color='b')
ax[0].grid()
ax[0].set_title('state 1')

# second state
ax[1].plot(np.arange(T), s_mean[:, 1], color='g')
ax[1].fill_between(np.arange(T), s_mean[:, 1] + ci_s[:, 1], s_mean[:, 1] - ci_s[:, 1], alpha=0.5, color='g')
ax[1].grid()
ax[1].set_title('state 2')

# actions
ax[2].plot(np.arange(T), a_mean, color='r')
ax[2].fill_between(np.arange(T), a_mean + ci_a, a_mean - ci_a, alpha=0.5, color='r')
ax[2].grid()
ax[2].set_title('action')
plt.show()



