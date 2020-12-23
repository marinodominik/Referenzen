import numpy as np
import matplotlib.pyplot as plt

def mutation (n = 20):
	P = np.array([[0.45, 0.023], [0.55, 0.977]])
	start_vector = np.array ([1,  0]).T
	mut = []
	no_mut = []
	y = []
	for i in range(1, n):
		p = np.dot(np.linalg.matrix_power(P, i), start_vector)
		mut.append(p[0])
		no_mut.append(p[1])
		y.append(i)

	plt.plot(y, mut, label='mutation')
	plt.xlabel("iteration")
	plt.ylabel( 'Probability')

	plt.plot(y, no_mut , label='no mutation' )
	plt.legend()
	plt.show()
	print(np.linalg.matrix_power(P, 20))
	print(mut[-1], no_mut[-1])


if __name__ == "__main__":
	mutation()
