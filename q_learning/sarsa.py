import numpy as np
from wind_env import WindEnv

def epsilon_greedy_policy(Q, state, action_space, eps):
	"""

	:param Q: Q function table, [num_states, num_actions]
	:param state: int, indicating the state
	:param action_space: list of actions
	:param eps: float
	:return:
	"""
	A = len(action_space)
	probs = np.ones([A]) * eps / A
	Qs = Q[state, :]
	a = np.argmax(Qs)
	probs[a] += 1 - eps
	a = np.random.choice(action_space, p=probs)
	return a, Qs[a]


def forward_sarsa(epsilon=0.1, gamma=0.99, alpha=0.5, num_episodes_max=1500):
	"""
	:param epsilon: float, param for epsilon greedy policy
	:param gamma: discount factor
	:param alpha: learning rate
	:return:
	"""
	env = WindEnv()
	action_space = np.arange(len(env.action_space))
	Q = np.zeros([len(env.state_space), len(action_space)])#  + 0.1


	# num_episodes = 0
	num_steps = []
	for e in range(num_episodes_max):
		# num_episodes += 1
		num_steps_epi = 0

		s = env.reset()
		a, _ = epsilon_greedy_policy(Q, s, action_space, epsilon)
		terminated = False
		while not terminated:
			sp, reward, terminated = env.step(a, return_pair=False)
			ap, Qp = epsilon_greedy_policy(Q, sp, action_space, epsilon/ np.sqrt(e/5+1)) # epsilon/ np.sqrt(e/5+1)
			Q[s, a] += alpha / np.sqrt(np.sqrt(e+1)) * (reward + gamma * Qp * (1 - terminated) - Q[s, a])
			# print(s, a, sp, ap)
			s, a = sp, ap
			num_steps_epi += 1
		num_steps.append(num_steps_epi)
	print('training finished')
	return Q, num_steps

def backward_sarsa(epsilon=0.1, gamma=0.99, alpha=0.5, lamb=0.5, num_episodes_max=1500):
	"""
	:param epsilon: float, param for epsilon greedy policy
	:param gamma: discount factor
	:param alpha: learning rate
	:param lamb: float, lambda-return parameter
	:return:
	"""
	env = WindEnv()
	action_space = np.arange(len(env.action_space))
	Q = np.zeros([len(env.state_space), len(action_space)])


	# num_episodes = 0
	num_steps = []
	for e in range(num_episodes_max):
		num_steps_epi = 0

		s = env.reset()
		a, _ = epsilon_greedy_policy(Q, s, action_space, epsilon)
		E = np.zeros([len(env.state_space), len(action_space)])
		terminated = False
		while not terminated:
			E[s,a] += 1
			sp, reward, terminated = env.step(a, return_pair=False)
			ap, Qp = epsilon_greedy_policy(Q, sp, action_space, epsilon/ np.sqrt(e/5+1)) # epsilon/ np.sqrt(e/5+1)
			delta = reward + gamma * Qp * (1 - terminated) - Q[s, a]
			Q += alpha / np.sqrt(np.sqrt(e+1)) * delta * E
			E *= gamma * lamb
			s, a = sp, ap
			num_steps_epi += 1
		num_steps.append(num_steps_epi)
	print('training finished')
	return Q, num_steps

def q_learning(epsilon=0.1, gamma=0.99, alpha=0.5, num_episodes_max=1500):
	"""
	:param epsilon: float, param for epsilon greedy policy
	:param gamma: discount factor
	:param alpha: learning rate
	:return:
	"""
	env = WindEnv()
	action_space = np.arange(len(env.action_space))
	Q = np.zeros([len(env.state_space), len(action_space)])#  + 0.1


	# num_episodes = 0
	num_steps = []
	for e in range(num_episodes_max):
		# num_episodes += 1
		num_steps_epi = 0

		s = env.reset()
		terminated = False
		while not terminated:
			a, _ = epsilon_greedy_policy(Q, s, action_space, epsilon)
			sp, reward, terminated = env.step(a, return_pair=False)
			_, Qp = epsilon_greedy_policy(Q, sp, action_space, 0.)
			Q[s, a] += alpha / np.sqrt(np.sqrt(e+1)) * (reward + gamma * Qp * (1 - terminated) - Q[s, a])
			s = sp
			num_steps_epi += 1
		num_steps.append(num_steps_epi)
	print('training finished')
	return Q, num_steps

def double_q_learning(epsilon=0.1, gamma=0.99, alpha=0.5, num_episodes_max=1500):
	"""
	:param epsilon: float, param for epsilon greedy policy
	:param gamma: discount factor
	:param alpha: learning rate
	:return:
	"""
	env = WindEnv()
	action_space = np.arange(len(env.action_space))
	Qs = [
		np.zeros([len(env.state_space), len(action_space)]),
		np.zeros([len(env.state_space), len(action_space)])
	]

	num_steps = []
	for e in range(num_episodes_max):
		num_step_epi = 0
		s = env.reset()
		terminated = False

		while not terminated:
			a, _ = epsilon_greedy_policy(Qs[0]+Qs[1], s, action_space, epsilon)
			sp, reward, terminated = env.step(a, return_pair=False)

			if np.random.randn() < 0.:
				update, target = 0, 1
			else:
				update, target = 1, 0

			# # wrong version
			# ap, _ = epsilon_greedy_policy(Qs[target], sp, action_space, 0.)
			# Qp = Qs[update][sp, ap]

			# correct version
			ap, _ = epsilon_greedy_policy(Qs[update], sp, action_space, 0.)
			Qp = Qs[target][sp, ap]

			Qs[update][s, a] += alpha / np.sqrt(np.sqrt(e + 1)) * (reward + gamma * Qp * (1 - terminated) - Qs[update][s, a])
			s = sp
			num_step_epi += 1
		num_steps.append(num_step_epi)
	print('training finished')
	return (Qs[0]+Qs[1])/2, num_steps



