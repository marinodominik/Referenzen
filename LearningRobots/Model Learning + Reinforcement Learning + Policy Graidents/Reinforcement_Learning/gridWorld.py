import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

maxIters = 15


##
def genGridWorld():
    O = -1e5  # Dangerous places to avoid
    D = 35  # Dirt
    W = -100  # Water
    C = -3000  # Cat
    T = 1000  # Toy
    grid_list = {0: '', O: 'O', D: 'D', W: 'W', C: 'C', T: 'T'}
    grid_world = np.array([[0, O, O, 0, 0, O, O, 0, 0, 0],
                           [0, 0, 0, 0, D, O, 0, 0, D, 0],
                           [0, D, 0, 0, 0, O, 0, 0, O, 0],
                           [O, O, O, O, 0, O, 0, O, O, O],
                           [D, 0, 0, D, 0, O, T, D, 0, 0],
                           [0, O, D, D, 0, O, W, 0, 0, 0],
                           [W, O, 0, O, 0, O, D, O, O, 0],
                           [W, 0, 0, O, D, 0, 0, O, D, 0],
                           [0, 0, 0, D, C, O, 0, 0, D, 0]])
    return grid_world, grid_list


##
def showWorld(grid_world, tlt):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(tlt)
    ax.set_xticks(np.arange(0.5, 10.5, 1))
    ax.set_yticks(np.arange(0.5, 9.5, 1))
    ax.grid(color='b', linestyle='-', linewidth=1)
    ax.imshow(grid_world, interpolation='nearest', cmap='copper')
    return ax


##
def showTextState(grid_world, grid_list, ax):
    for x in range(grid_world.shape[0]):
        for y in range(grid_world.shape[1]):
            if grid_world[x, y] >= -3000:
                ax.annotate(grid_list.get(grid_world[x, y]), xy=(y, x), horizontalalignment='center')


##
def showPolicy(policy, ax):
    for x in range(policy.shape[0]):
        for y in range(policy.shape[1]):
            if policy[x, y] == 0:
                ax.annotate('$\downarrow$', xy=(y, x), horizontalalignment='center')
            elif policy[x, y] == 1:
                ax.annotate(r'$\rightarrow$', xy=(y, x), horizontalalignment='center')
            elif policy[x, y] == 2:
                ax.annotate('$\u2191$', xy=(y, x), horizontalalignment='center')
            elif policy[x, y] == 3:
                ax.annotate('$\leftarrow$', xy=(y, x), horizontalalignment='center')
            elif policy[x, y] == 4:
                ax.annotate('$\perp$', xy=(y, x), horizontalalignment='center')


##
def ValIter(R, discount=None, maxSteps=None, infHor=False, probModel=False):
    T = maxSteps
    rows, columns = R.shape

    robot_actions = {
        "down": 0,
        "right": 1,
        "up": 2,
        "left": 3,
        "stay": 4
    }

    Q = np.zeros((rows, columns, len(robot_actions)))  # Basis Function

    if not infHor:
        V = np.zeros((rows, columns, T))                #Value Function
        V[:, :, -1] = np.copy(R)

        for t in reversed(range(T - 1)):
            for row in range(rows):
                for col in range(columns):
                    for action in robot_actions.items():
                        if not probModel:
                            new_pos = calcNextPosition(action[0], np.asarray([row, col]))
                            Q[row, col, action[1]] = R[row, col] + V[new_pos[0], new_pos[1], t + 1]
                        else:
                            #Task d)
                            new_pos, prob = calcNextPosition(action[0], np.asarray([row, col]), True)
                            Q[row, col, action[1]] = R[row, col]
                            for pos, p in zip(new_pos, prob):
                                Q[row, col, action[1]] = Q[row, col, action[1]] + p * V[pos[0], pos[1], t + 1]
                    V[row, col, t] = max(Q[row, col, :])
    else:
        improvement = np.inf
        V = np.zeros(R.shape)
        V_last = np.zeros(R.shape)

        while improvement > 1e-5:
            for row in range(rows):
                for col in range(columns):
                    for action in robot_actions.items():
                        new_pos = calcNextPosition(action[0], np.asarray([row, col]))
                        Q[row, col, action[1]] = R[row, col] + discount * V[new_pos[0], new_pos[1]]
            V = np.max(Q, axis=2)
            improvement = abs(np.sum(V.flatten() - V_last.flatten()))
            V_last = V

    return V, Q



##
def maxAction(V, R, discount, probModel=None):
    # YOUR CODE HERE
    pass        #TODO ?????


##
def findPolicy(Q, probModel=False):
    if not probModel:
        return np.argmax(Q, axis=2)
    else:
        pass    #TODO ?????


def calcNextPosition(action, pos, prob_model=False):
    new_pos = np.zeros((1, 2))
    prob = 0
    if not prob_model:
        if action == "down":
            new_pos = pos + np.array([1, 0])
            if new_pos[0] > 8:
                new_pos = pos

        elif action == "right":
            new_pos = pos + np.array([0, 1])
            if new_pos[1] > 9:
                new_pos = pos

        elif action == "up":
            new_pos = pos + np.array([-1, 0])
            if new_pos[0] < 0:
                new_pos = pos

        elif action == "left":
            new_pos = pos + np.array([0, -1])

            if new_pos[1] < 0:
                new_pos = pos

        elif action == "stay":
            new_pos = pos
    else:
        if action == "down" or action == "up":
            #also possible action left OR right OR failing
            new_pos = calcNextPosition(action, pos)
            new_pos = np.vstack((new_pos, calcNextPosition("left", pos)))
            new_pos = np.vstack((new_pos, calcNextPosition("right", pos)))
            new_pos = np.vstack((new_pos, calcNextPosition("stay", pos)))
            prob = np.array([0.7, 0.1, 0.1, 0.1]).reshape((4, 1))
        elif action == "right" or action == "left":
            #possible actions up OR down OR failing
            new_pos = calcNextPosition(action, pos)
            new_pos = np.vstack((new_pos, calcNextPosition("down", pos)))
            new_pos = np.vstack((new_pos, calcNextPosition("up", pos)))
            new_pos = np.vstack((new_pos, calcNextPosition("stay", pos)))
            prob = np.array([0.7, 0.1, 0.1, 0.1]).reshape((4, 1))
        else:
            new_pos = pos.reshape((1, 2))
            prob = np.array([1])

    if prob_model == False:
        return np.asarray(new_pos)
    else:
        return np.asarray(new_pos), np.asarray(prob)


############################

saveFigures = True

data = genGridWorld()
grid_world = data[0]
grid_list = data[1]

ax = showWorld(grid_world, 'Environment')
showTextState(grid_world, grid_list, ax)
if saveFigures:
    plt.savefig('gridworld.pdf')
#
# # Finite Horizon
V, Q = ValIter(grid_world, maxSteps=15, infHor=False)
V = V[:, :, 0]
showWorld(np.maximum(V, 0), 'Value Function - Finite Horizon')
if saveFigures:
    plt.savefig('value_Fin_15.pdf')

policy = findPolicy(Q)
ax = showWorld(grid_world, 'Policy - Finite Horizon')
showPolicy(policy, ax)
if saveFigures:
    plt.savefig('policy_Fin_15.pdf')
#
# # # Infinite Horizon
V, Q = ValIter(grid_world, discount=0.8, infHor=True)
showWorld(np.maximum(V, 0), 'Value Function - Infinite Horizon')
if saveFigures:
   plt.savefig('value_Inf_08.pdf')

policy = findPolicy(Q)
ax = showWorld(grid_world, 'Policy - Infinite Horizon')
showPolicy(policy, ax)
if saveFigures:
    plt.savefig('policy_Inf_08.pdf')

# # Finite Horizon with Probabilistic Transition
V, Q = ValIter(grid_world, maxSteps=15, probModel=True)
V = V[:, :, 0]
showWorld(np.maximum(V, 0), 'Value Function - Finite Horizon with Probabilistic Transition')
if saveFigures:
    plt.savefig('value_Fin_15_prob.pdf')

policy = findPolicy(Q)
ax = showWorld(grid_world, 'Policy - Finite Horizon with Probabilistic Transition')
showPolicy(policy, ax)
if saveFigures:
   plt.savefig('policy_Fin_15_prob.pdf')
