import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import gym

import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage
import imageio


global_env = gym.make("CartPole-v1", render_mode='human', max_episode_steps=1000)
# global_env = gym.make("CartPole-v1", render_mode='rgb_array', max_episode_steps=1000)

def array_to_tensor(arr):
	return torch.Tensor(arr)

def tensor_to_array(ten):
	return ten.cpu().detach().numpy()

def pipe(x, *funcs):
	for f in funcs:
		x = f(x)
	return x


def epsilon_greedy_policy(Qs, eps):
	"""
	:param Qs: np.array, [batch, num_actions]
	:param eps:
	:return:
	"""
	probs = np.ones_like(Qs) * eps / Qs.shape[1]
	a = np.argmax(Qs, axis=1)
	probs[np.arange(Qs.shape[0]), a] += 1-eps

	actions = [
		np.random.choice(np.arange(Qs.shape[1]), p=p) for p in probs
	]
	return np.array(actions)



class DQN(nn.Module):
	def __init__(self):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(4, 16), nn.ELU(),
			nn.Linear(16, 32), nn.ELU(),
			nn.Linear(32, 16), nn.ELU(),
			nn.Linear(16, 2)
		)

	def forward(self, x):
		return self.net(x)


	def run_model(self, max_steps=1000):
		state, _ = global_env.reset()
		steps = 0
		rewards = 0


		for _ in range(max_steps):
			Qs = pipe(
				state[None, :], array_to_tensor, self.net, tensor_to_array
			)
			actions = epsilon_greedy_policy(Qs, 0)
			state, reward, terminal, truncated, info = global_env.step(actions[0])
			steps += 1
			rewards += reward

			if terminal:
				break
		return steps, rewards


class DQNSolver():
	def __init__(self):
		self.env = gym.make("CartPole-v1", render_mode='rgb_array', max_episode_steps=1000)
		self.qnet = DQN()
		self.qnet_target = DQN()
		self.buffer = []
		self.buffer_size = 100000
		self.gamma = 0.99

		self.state, _ = self.env.reset()
		self.qnet_target.load_state_dict(self.qnet.state_dict())

	@torch.no_grad()
	def sample(self, steps=100, random=False, eps=0.1):

		for _ in range(steps):
			if random:
				action = self.env.action_space.sample()
			else:
				Qs = pipe(
					self.state[None, :], array_to_tensor, self.qnet, tensor_to_array
				)
				action = epsilon_greedy_policy(Qs, eps)[0]

			state_next, reward, terminal, truncated, info = self.env.step(action)

			self.buffer.append(
				(self.state, action, reward * 0.1, terminal, state_next)
			)

			self.state = state_next

			if terminal:
				self.state, _ = self.env.reset()

		if len(self.buffer) > self.buffer_size:
			self.buffer = self.buffer[:self.buffer_size]

	def sample_from_buffer(self, batch_size):
		idx = np.random.randint(0, len(self.buffer), (batch_size,))
		batch = [self.buffer[i] for i in idx]
		states = np.array([r[0] for r in batch])
		actions = np.array([r[1] for r in batch])
		rewards = np.array([r[2] for r in batch])
		terminals = np.array([r[3] for r in batch])
		states_next = np.array([r[4] for r in batch])
		return torch.from_numpy(states).float(), \
				torch.from_numpy(actions), \
				torch.from_numpy(rewards).float(), \
				torch.from_numpy(terminals).float(), \
				torch.from_numpy(states_next).float()

	def train(self):
		batch_size = 32
		self.sample(256, random=True)
		optimizer = torch.optim.Adam(params=self.qnet.parameters(), lr=1e-4)
		num_iter = 0

		while True:
			num_iter += 1
			states, actions, rewards, terminals, states_next = solver.sample_from_buffer(batch_size)

			with torch.no_grad():
				Qs_next = self.qnet(states_next)
				astar = torch.argmax(Qs_next, dim=1)
				Q_target = self.qnet_target(states_next)[torch.arange(batch_size), astar] # double DQN
				Q_target = rewards + self.gamma * Q_target * (1 - terminals)

			Qs = self.qnet(states)[torch.arange(batch_size), actions]
			loss = F.mse_loss(Qs, Q_target)

			optimizer.zero_grad()
			loss.backward()
			nn.utils.clip_grad_norm_(self.qnet.parameters(), 0.5)
			optimizer.step()


			# polyak update
			for param, target_param in zip(self.qnet.parameters(), self.qnet_target.parameters()):
				target_param.data.copy_((1-0.005)*target_param.data + 0.005*param.data)

			if num_iter % 20 == 0:
				# hard update
				# with torch.no_grad():
				# 	self.qnet_target.load_state_dict(self.qnet.state_dict())
				self.sample(16, eps=0.1)

			if num_iter % 2000 == 0:
				steps, rewards = self.qnet.run_model(1000)
				print("%d, %.2f, %.4f" % (num_iter, rewards, loss.item()))

if __name__ == '__main__':
	solver = DQNSolver()
	solver.train()
	# solver.sample(100)
	# state, actions, rewards, terminals, state_next = solver.sample_from_buffer(32)
	# print(state.shape, actions.shape, rewards.shape, terminals.shape,  state_next.shape)