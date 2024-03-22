import numpy as np
import matplotlib.pyplot as plt

from sarsa import forward_sarsa, backward_sarsa, q_learning, double_q_learning, epsilon_greedy_policy
from wind_env import WindEnv

def run_by_Q(Q, eps):
	print('start running')
	env = WindEnv()
	action_space = np.arange(len(env.action_space))
	env.reset()
	# env.state = env.coord_to_idx([3, 0])
	s = env.state
	terminated = False
	path = [env.idx_to_coord(s)]
	while not terminated:
		a, _ = epsilon_greedy_policy(Q, s, action_space, eps)
		s, _, terminated = env.step(a, return_pair=False)
		path.append(env.idx_to_coord(s))
	print('running finished')
	return np.array(path)

def plot_board(ax):
	env = WindEnv()
	n_plot = 100

	# plot horizontal lines
	xs = np.linspace(0, env.w, n_plot)
	for i in range(env.h + 1):
		ax.plot(xs, np.ones([n_plot]) * i, 'b-')

	# plot vertical lines
	ys = np.linspace(0, env.h, n_plot)
	for i in range(env.w + 1):
		ax.plot(np.ones([n_plot]) * i, ys, 'b-')

def plot_states(ax, states):
	env = WindEnv()
	Is, Js = states[:, 0], states[:, 1]
	ax.plot(Js + 0.5, env.h - Is - 0.5, 'r-')

def plot_steps(ax, num_steps, label=''):
	acc = 0
	plots = []
	for i, n in enumerate(num_steps):
		l = [acc for _ in range(n)]
		acc += 1
		plots += l
	ax.plot(plots, label=label)
	ax.set_xlabel('time steps')
	ax.set_ylabel('episodes')

if __name__ == '__main__':
	fig, (ax1, ax2) = plt.subplots(1, 2)
	Q, num_steps = forward_sarsa(epsilon=0.1, gamma=0.99, alpha=1., num_episodes_max=400)
	path1 = run_by_Q(Q, 0.)
	plot_steps(ax2, num_steps, label='forward sarsa')

	Q, num_steps = backward_sarsa(epsilon=0.1, gamma=0.99, alpha=1., lamb=0.7, num_episodes_max=400)
	path2 = run_by_Q(Q, 0.)
	plot_steps(ax2, num_steps, label='backward sarsa')

	Q, num_steps = q_learning(epsilon=0.1, gamma=0.99, alpha=1., num_episodes_max=400)
	path3 = run_by_Q(Q, 0.)
	plot_steps(ax2, num_steps, label='q learning')

	Q, num_steps = double_q_learning(epsilon=0.1, gamma=0.99, alpha=1., num_episodes_max=400)
	path4 = run_by_Q(Q, 0.)
	plot_steps(ax2, num_steps, label='double q learning')



	plot_board(ax1)
	plot_states(ax1, path4)
	plt.legend()
	plt.show()