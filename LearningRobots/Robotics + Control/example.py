# Robot Learning - Group 169
# Victor Jimenez 2491031
# Daniel Piendl 2586991
# Dominik Marino 2468378

from jointCtlComp import *
from taskCtlComp import *

# Controller in the joint space. The robot has to reach a fixed position.
jointCtlComp(['P', 'PD', 'PID', 'PD_Grav', 'ModelBased'], True)

# Same controller, but this time the robot has to follow a fixed trajectory.
# Run again with high Gains. Change Parameter in my_ctl and change plot behaviour in jointCtlComp
jointCtlComp(['P', 'PD', 'PID', 'PD_Grav', 'ModelBased'], False)

# Controller in the task space.
taskCtlComp(['JacTrans'],resting_pos=np.mat([0, np.pi]).T)
taskCtlComp(['JacPseudo'],resting_pos=np.mat([0, np.pi]).T)
taskCtlComp(['JacDPseudo'],resting_pos=np.mat([0, np.pi]).T)

taskCtlComp(['JacNullSpace'],resting_pos=np.mat([0, np.pi]).T)
taskCtlComp(['JacNullSpace'],resting_pos=np.mat([0, -np.pi]).T)

input('Press Enter to close')