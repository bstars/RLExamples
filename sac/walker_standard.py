import sys
sys.path.append('..')

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import namedtuple
import imageio

from dm_wrapper import DeepMindWrapper

def array_to_tensor(arr):
	return torch.from_numpy(arr).float()

def tensor_to_array(ten):
	return ten.cpu().detach().numpy()


def pipe(x, *funcs):
	for f in funcs:
		x = f(x)
	return x

class QNetwork(nn.Module):
	def __init__(self, state_dim, action_dim):
		super().__init__()

		self.net = nn.Sequential(
			nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
			nn.Linear(256, 256), nn.ReLU(),
			nn.Linear(256, 1)
		)

	def forward(self, states, actions):
		"""

		:param states: [batch, state_dim]
		:param actions: [batch, action_dim]
		:return:
		:rtype:
		"""
		sa = torch.cat([states, actions], dim=-1)
		return self.net(sa).squeeze(-1)

class VNetwork(nn.Module):
	def __init__(self, state_dim):
		super().__init__()

		self.net = nn.Sequential(
			nn.Linear(state_dim, 256), nn.ReLU(),
			nn.Linear(256, 256), nn.ReLU(),
			nn.Linear(256, 1)
		)

	def forward(self, states):
		"""
		:param states: [batch, state_dim]
		:return:
		:rtype:
		"""
		return self.net(states).squeeze(-1)

class PiNetwork(nn.Module):
	def __init__(self, state_dim, action_dim):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(state_dim, 256), nn.ReLU(),
			nn.Linear(256, 256), nn.ReLU(),
			nn.Linear(256, action_dim * 2)
		)

	def forward(self, states):
		"""
		:param states: [batch, state_dim]
		:return:
			u_transformed: [n_samples, action_dim]
			log_prob_transformed: [n_samples, batch]
		"""
		mu_std = self.net(states)
		mu, std = torch.chunk(mu_std, 2, -1)
		std = std.clamp(min=-20, max=2).exp()
		dist = Normal(mu, std)

		u = dist.rsample()
		u_transformed = torch.tanh(u)

		# y = T(x)
		# P(y) = P(x) / |T'(x)|
		log_prob = dist.log_prob(u)
		log_prob_transformed = log_prob - torch.log(1 - u_transformed**2 + 1e-6)
		log_prob_transformed = log_prob_transformed.sum(-1)
		return u_transformed, log_prob_transformed

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'terminal', 'state_next'))


class Memory:
	def __init__(self, size):
		self.buffer = []
		self.size = size

	def add(self, state, action, reward, terminal, state_next):
		self.buffer.append(
			Transition(state, action, reward, terminal, state_next)
		)
		if len(self.buffer) > self.size:
			self.buffer.pop(0)

	def sample(self, batch_size):
		idx = np.random.randint(0, len(self.buffer), (batch_size,))
		state = np.array([self.buffer[i].state for i in idx])
		action = np.array([self.buffer[i].action for i in idx])
		reward = np.array([self.buffer[i].reward for i in idx])
		terminal = np.array([self.buffer[i].terminal for i in idx])
		state_next = np.array([self.buffer[i].state_next for i in idx])
		return torch.from_numpy(state).float(), \
			torch.from_numpy(action).float(), \
			torch.from_numpy(reward).float(), \
			torch.from_numpy(terminal).float(), \
			torch.from_numpy(state_next).float()

class SACStandard:
	def __init__(self):
		self.gamma = 0.99
		self.env = DeepMindWrapper('walker', 'walk')
		self.state, _ = self.env.reset()
		self.buffer_size = int(1e+6)
		self.state_dim = 24
		self.action_dim = 6
		self.batch_size = 256
		self.episodes = 0
		self.episode_reward = 0

		self.buffer = Memory(self.buffer_size)
		self.Pi = PiNetwork(self.state_dim, self.action_dim)
		self.Q1 = QNetwork(self.state_dim, self.action_dim)
		self.Q2 = QNetwork(self.state_dim, self.action_dim)
		self.V = VNetwork(self.state_dim)
		self.V_target = VNetwork(self.state_dim)
		self.V_target.load_state_dict(self.V.state_dict())
		self.V_target.eval()

		self.log_alpha = torch.zeros(1, requires_grad=True)

	@torch.no_grad()
	def sample_step(self, random=True):
		if random:
			action = self.env.random_action()
		else:
			action, _ = pipe(self.state[None,:], array_to_tensor, self.Pi)
			action = tensor_to_array(action[0])

		state_next, reward, terminal, _, _ = self.env.step(action)
		self.episode_reward += reward

		self.buffer.add(self.state.copy(), action, reward, terminal, state_next)

		self.state = state_next.copy()

		if terminal:
			self.episodes += 1
			print(self.episodes, self.episode_reward)
			self.episode_reward = 0
			self.state, _ = self.env.reset()

	def train(self):
		for _ in range(256):
			self.sample_step(random=False)
		Q1_opt = torch.optim.Adam(self.Q1.parameters(), lr=1e-3)
		Q2_opt = torch.optim.Adam(self.Q2.parameters(), lr=1e-3)
		V_opt = torch.optim.Adam(self.V.parameters(), lr=1e-3)
		Pi_opt = torch.optim.Adam(self.Pi.parameters(), lr=1e-3)
		A_opt = torch.optim.Adam([self.log_alpha], lr=1e-3)
		num_iter = 0
		num_ckpt = 0

		while True:
			num_iter += 1
			states, actions, rewards, terminals, states_next = self.buffer.sample(self.batch_size)

			with torch.no_grad():
				q_target = rewards + self.gamma * self.V_target(states_next)

			# Train Q network
			q1 = self.Q1(states, actions)
			q2 = self.Q2(states, actions)
			q1_loss = F.mse_loss(q1, q_target)
			q2_loss = F.mse_loss(q2, q_target)

			Q1_opt.zero_grad()
			q1_loss.backward()
			Q1_opt.step()

			Q2_opt.zero_grad()
			q2_loss.backward()
			Q2_opt.step()

			# sample actions
			actions_new, log_probs_new = self.Pi(states)

			# Train V network
			q1 = self.Q1(states, actions_new)
			q2 = self.Q2(states, actions_new)
			q = torch.min(q1, q2)
			v_target = q - self.log_alpha.exp() * log_probs_new
			v = self.V(states)
			v_loss = F.mse_loss(v, v_target.detach())

			V_opt.zero_grad()
			v_loss.backward()
			V_opt.step()

			# Train Pi network
			Pi_loss = (self.log_alpha.exp().detach() * log_probs_new - q).mean()
			Pi_opt.zero_grad()
			Pi_loss.backward()
			Pi_opt.step()

			# dual update, very similar to Adaptive KL coefficient in PPO
			A_loss = - self.log_alpha.exp() * (log_probs_new - self.action_dim).detach().mean()

			A_opt.zero_grad()
			A_loss.backward()
			A_opt.step()

			with torch.no_grad():
				for param, target_param in zip(self.V.parameters(), self.V_target.parameters()):
					target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)

			self.sample_step(random=False)

			if num_iter % 50000 == 0:
				torch.save(
					{
						'Q1':self.Q1.state_dict(),
						'Q2': self.Q2.state_dict(),
						'Pi':self.Pi.state_dict(),
						'V': self.V.state_dict(),
						'log_alpha':self.log_alpha
					},
					'standard_%d.pth' % (num_ckpt)
				)
				num_ckpt += 1

def play():
	env = DeepMindWrapper('walker', 'walk')
	Pi = PiNetwork(24, 6)
	ckpt = torch.load('./standard_3.pth')
	Pi.load_state_dict(ckpt['Pi'])

	state, _ = env.reset()
	gifs = []
	while True:
		gifs.append(env.render())
		action, _ = pipe(state[None, :], array_to_tensor, Pi)
		action = tensor_to_array(action[0])
		state, reward, terminal, _, _ = env.step(action)
		if terminal:
			break
	imageio.mimsave('walker_standard.gif', gifs, duration=50, loop=1000)

if __name__ == '__main__':
	sac = SACStandard()
	sac.train()




