import sys
sys.path.append('..')

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import namedtuple
import imageio
import gym


def array_to_tensor(arr):
	return torch.from_numpy(arr).float()

def tensor_to_array(ten):
	return ten.cpu().detach().numpy()


def pipe(x, *funcs):
	for f in funcs:
		x = f(x)
	return x


class QNetwork(nn.Module):
	def __init__(self, state_dim, action_dim, n_skills):
		super().__init__()
		self.n_skills = n_skills
		self.net = nn.Sequential(
			nn.Linear(state_dim + action_dim + n_skills, 256), nn.ReLU(),
			nn.Linear(256, 256), nn.ReLU(),
			nn.Linear(256, 1)
		)

	def forward(self, states, actions, skills):
		"""
		:param states: [batch, state_dim]
		:param actions: [batch, action_dim]
		:param skills: [batch,]
		:return:
		"""
		z = F.one_hot(skills, num_classes=self.n_skills).detach()
		saz = torch.cat([states, actions, z], dim=-1)
		return self.net(saz).squeeze(-1)


class PiNetwork(nn.Module):
	def __init__(self, state_dim, action_dim, n_skills):
		super().__init__()
		self.n_skills = n_skills
		self.net = nn.Sequential(
			nn.Linear(state_dim + n_skills, 256), nn.ReLU(),
			nn.Linear(256, 256), nn.ReLU(),
			nn.Linear(256, action_dim * 2)
		)

	def forward(self, states, skills):
		"""
		:param states: [batch, state_dim]
		:param skills: [batch, ]
		:return:
		"""
		z = F.one_hot(skills, num_classes=self.n_skills).detach()
		sz = torch.cat([states, z], dim=-1)
		mu_std = self.net(sz)
		mu, std = torch.chunk(mu_std, 2, -1)
		std = std.clamp(min=-20, max=2).exp()
		dist = Normal(mu, std)

		u = dist.rsample()
		u_transformed = torch.tanh(u)

		# y = T(x)
		# P(y) = P(x) / |T'(x)|
		log_prob = dist.log_prob(u)
		log_prob_transformed = log_prob - torch.log(1 - u_transformed ** 2 + 1e-6)
		log_prob_transformed = log_prob_transformed.sum(-1)
		return u_transformed, log_prob_transformed

class DNetwork(nn.Module):
	def __init__(self, state_dim, n_skills):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(state_dim, 128), nn.ReLU(),
			nn.Linear(128, 128), nn.ReLU(),
			nn.Linear(128, n_skills)
		)

	def forward(self, states):
		"""
		:param states: [batch, state_dim]
		:return:
		"""
		return self.net(states)

Transition = namedtuple('Transition',
                        ('state', 'action', 'skill', 'terminal', 'state_next'))

class Memory:
	def __init__(self, size):
		self.buffer = []
		self.size = size

	def add(self, state, action, skill, terminal, state_next):
		self.buffer.append(
			Transition(state, action, skill, terminal, state_next)
		)
		if len(self.buffer) > self.size:
			self.buffer.pop(0)

	def sample(self, batch_size):
		idx = np.random.randint(0, len(self.buffer), (batch_size,))
		state = np.array([self.buffer[i].state for i in idx])
		action = np.array([self.buffer[i].action for i in idx])
		skill = np.array([self.buffer[i].skill for i in idx])
		terminal = np.array([self.buffer[i].terminal for i in idx])
		state_next = np.array([self.buffer[i].state_next for i in idx])
		return torch.from_numpy(state).float(), \
			torch.from_numpy(action).float(), \
			torch.from_numpy(skill).long()[:,0], \
			torch.from_numpy(terminal).float(), \
			torch.from_numpy(state_next).float()


class DIAYN:
	def __init__(self):
		self.gamma = 0.99
		self.n_skills = 50
		self.state_dim = 24
		self.action_dim = 4
		self.batch_size = 256
		self.episodes = 0
		self.alpha = 0.2
		self.env = gym.make("BipedalWalker-v3", render_mode='rgb_array', max_episode_steps=1000)

		self.buffer = Memory(int(1e6))
		self.D = DNetwork(self.state_dim, self.n_skills)
		self.Pi = PiNetwork(self.state_dim, self.action_dim, self.n_skills)
		self.Q1 = QNetwork(self.state_dim, self.action_dim, self.n_skills)
		self.Q2 = QNetwork(self.state_dim, self.action_dim, self.n_skills)
		self.Q1_target = QNetwork(self.state_dim, self.action_dim, self.n_skills)
		self.Q2_target = QNetwork(self.state_dim, self.action_dim, self.n_skills)
		self.Q1_target.load_state_dict(self.Q1.state_dict())
		self.Q2_target.load_state_dict(self.Q2.state_dict())

		self.skill = np.random.randint(self.n_skills, size=(1,), dtype=int)
		self.state, _ = self.env.reset()

	@torch.no_grad()
	def sample_step(self):
		action, _ = self.Pi(
			array_to_tensor(self.state[None, :]),
			array_to_tensor(self.skill).long()
		)

		action = tensor_to_array(action[0])
		state_next, _, terminal, _, _ = self.env.step(action)

		self.buffer.add(self.state.copy(), action, self.skill, terminal, state_next)
		self.state = state_next.copy()

		if terminal:
			self.episodes += 1
			self.state, _ = self.env.reset()
			self.skill = np.random.randint(self.n_skills, size=(1,), dtype=int)

	def train(self):

		while self.episodes < 1:
			self.sample_step()

		D_opt = torch.optim.Adam(self.D.parameters(), lr=3e-4)
		Q1_opt = torch.optim.Adam(self.Q1.parameters(), lr=3e-4)
		Q2_opt = torch.optim.Adam(self.Q2.parameters(), lr=3e-4)
		Pi_opt = torch.optim.Adam(self.Pi.parameters(), lr=3e-4)

		num_iter = 0
		num_ckpt = 0

		while True:
			num_iter += 1
			states, actions, skills, terminal, states_next = self.buffer.sample(self.batch_size)

			# Compute reward
			# According to the paper, q should be q(s_{t+1}|z)
			# However, according to the code https://github.com/ben-eysenbach/sac/blob/master/sac/algos/diayn.py#L180 , it's q(s_t}|z)
			# I guess this does not make much difference since the whole episode is generated by \pi(z)
			with torch.no_grad():
				skills_logit = self.D(states)
				qz = F.log_softmax(skills_logit, dim=-1)[np.arange(self.batch_size), skills].detach()
				reward = qz - np.log(1/self.n_skills + 1e-6)

			with torch.no_grad():
				actions_next, log_probs_next = self.Pi(states, skills)
				q1_next = self.Q1(states_next, actions_next, skills)
				q2_next = self.Q2(states_next, actions_next, skills)
				q_next = torch.min(q1_next, q2_next)
				q_target = reward + self.gamma * (1 - terminal) * (q_next - self.alpha * log_probs_next)

			# Train Q Network
			Q1_opt.zero_grad()
			Q2_opt.zero_grad()

			q1 = self.Q1(states, actions, skills)
			q2 = self.Q2(states, actions, skills)
			q1_loss = F.mse_loss(q1, q_target)
			q2_loss = F.mse_loss(q2, q_target)

			q1_loss.backward()
			Q1_opt.step()

			q2_loss.backward()
			Q2_opt.step()


			# Train Policy
			actions_new, log_probs_new = self.Pi(states, skills)
			q1 = self.Q1(states, actions_new, skills)
			q2 = self.Q2(states, actions_new, skills)
			q = torch.min(q1, q2)
			Pi_loss = (self.alpha * log_probs_new - q).mean()

			Pi_opt.zero_grad()
			Pi_loss.backward()
			Pi_opt.step()

			# Train discriminator
			D_opt.zero_grad()
			skills_logit = self.D(states)
			d_loss = F.cross_entropy(skills_logit, skills)
			d_loss.backward()
			D_opt.step()

			with torch.no_grad():
				for param, target_param in zip(self.Q1.parameters(), self.Q1_target.parameters()):
					target_param.data.copy_(0.995*target_param.data + 0.005*param.data)

				for param, target_param in zip(self.Q2.parameters(), self.Q2_target.parameters()):
					target_param.data.copy_(0.995*target_param.data + 0.005*param.data)
			self.Q1_target.eval()
			self.Q2_target.eval()

			self.sample_step()

			if self.episodes > 2000:
				break

			if num_iter % 1000 == 0:
				print("%d, %.3f, %.3f" % (num_iter, qz.mean().item(), -log_probs_new.mean().item()))

			if num_iter % 50000 == 0:
				torch.save(
					{
						'Q1':self.Q1.state_dict(),
						'Q2': self.Q2.state_dict(),
						'Pi':self.Pi.state_dict(),
						'D':self.D.state_dict()
					},
					'modern_%d_64.pth' % (num_ckpt)
				)
				num_ckpt += 1

def play():
	env = gym.make("BipedalWalker-v3", render_mode='rgb_array')
	Pi = PiNetwork(24, 4, 50)
	ckpt = torch.load('./modern_11_64.pth')
	Pi.load_state_dict(ckpt['Pi'])

	for skill in range(50):
		z = torch.ones(1, dtype=int) * skill
		state, _ = env.reset()
		gifs = []
		steps = 0
		while True:
			steps += 1
			gifs.append(env.render())
			action, _ = Pi(
				array_to_tensor(state[None,:]), z
			)
			action = tensor_to_array(action[0])
			state, reward, terminal, _, _ = env.step(action)
			if terminal or steps > 1000:
				break
		imageio.mimsave('skill_%d.gif' % skill, gifs, duration=20, loop=1000)

if __name__ == '__main__':
	# agent = DIAYN()
	# agent.train()

	play()
