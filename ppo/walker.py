import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import gym


import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage
import imageio

device = 'cpu'
learning_rate = 1e-4
gamma = 0.99
lamb = 0.95
clip_ratio = 0.1
max_batch_length = 1024
batch_size = 32


vf_coef = 1.
ent_coef = 1e-2


# global_env = gym.make("BipedalWalker-v3", render_mode='human', max_episode_steps=1000)# .env
global_env = gym.make("BipedalWalker-v3", render_mode='rgb_array', max_episode_steps=1000)# .env

# global_env.reset()
# while True:
# 	action = global_env.action_space.sample()
# 	state_next, reward, terminal, truncated, info = global_env.step(action)
# 	global_env.render()
# 	if terminal or truncated:
# 		break

def array_to_tensor(arr):
	return torch.Tensor(arr)

def tensor_to_array(ten):
	return ten.cpu().detach().numpy()

def pipe(x, *funcs):
	for f in funcs:
		x = f(x)
	return x

class ActorCritic(nn.Module):
	def __init__(self):
		super().__init__()
		self.actor = nn.Sequential(
			nn.Linear(24, 32), nn.ELU(),
			nn.Linear(32, 32), nn.ELU(),
			nn.Linear(32, 16), nn.ELU(),
			nn.Linear(16, 8)
		)
		self.critic = nn.Sequential(
			nn.Linear(24, 32), nn.ELU(),
			nn.Linear(32, 32), nn.ELU(),
			nn.Linear(32, 16), nn.ELU(),
			nn.Linear(16, 1)
		)

	def pi(self, states):
		"""
		:param states: [batch, 24]
		:return:
		:rtype:
		"""
		mean_std = self.actor(states)
		mean = F.tanh(mean_std[:, :4])
		std = F.softplus(mean_std[:, 4:]) + 1e-4

		normal = Normal(mean, std)
		actions = normal.sample()
		log_prob = normal.log_prob(actions).sum(-1)
		return actions, mean, std, log_prob

	def v(self, states):
		return self.critic(states).squeeze(-1)

	def run_model(self, max_steps=500):
		state, _ = global_env.reset()
		steps = 0
		rewards = 0

		gifs = []
		while True:

			img = global_env.render()
			gifs.append(img)

			steps += 1
			global_env.render()
			action, _, _, _ = self.pi(array_to_tensor(state[None, :]))
			action = tensor_to_array(action[0])
			state, reward, terminal, truncated, info = global_env.step(action)
			rewards += reward

			# if terminal or truncated or (steps >= max_steps):
			if terminal or (steps >= max_steps):
				imageio.mimsave('walker.gif', gifs, duration=50, loop=1000)
				return steps, rewards


class PPOClip():
	def __init__(self):
		self.env = gym.make("BipedalWalker-v3", render_mode='rgb_array', max_episode_steps=1000)# .env
		self.state, _ = self.env.reset()
		self.ac = ActorCritic()
		self.gamma = gamma
		self.lamb = lamb
		self.episodes = 0

	@torch.no_grad()
	def sample(self, max_iteration=500):
		states = []
		actions = []
		action_log_probs = []
		rewards = []
		terminals = []
		values = []

		for _ in range(max_iteration):
			value = pipe(
				self.state[None,:], array_to_tensor, self.ac.v
			)

			action, mean, std, log_prob = pipe(
				self.state[None, :], array_to_tensor, self.ac.pi
			)

			state_next, reward, terminal, truncated, info = pipe(
				action[0], tensor_to_array, self.env.step
			)
			reward = -10 if terminal else reward

			states.append(self.state)
			actions.append(action[0])
			action_log_probs.append(log_prob[0])
			rewards.append(reward)
			terminals.append(terminal)
			values.append(value)

			self.state = state_next
			# if terminal or truncated:
			if terminal:
				self.state, _ = self.env.reset()

		value = pipe(
			self.state[None, :], array_to_tensor, self.ac.v
		)
		values.append(value)

		return torch.from_numpy(np.stack(states, axis=0)).float(), \
			torch.stack(actions, dim=0), \
			torch.stack(action_log_probs, dim=0), \
			torch.from_numpy(np.array(rewards)).float(), \
			torch.from_numpy(np.array(terminals)).float(), \
			torch.cat(values, dim=0)

	def lambda_return(self, values, rewards, terminals):
		"""
		:param values: torch.Tensor, [T+1]
		:param reward: torch.Tensor, np.array, [T]
		:param terminals: torch.Tensor, [T]
		:return:
		"""
		diffs = rewards + self.gamma * (1-terminals) * values[1:] - values[:-1]
		advantages = [0]
		for t in pipe(diffs, len, range, reversed):
			advantages.append(
				diffs[t] + self.gamma * self.lamb * (1-terminals[t]) * advantages[-1]
			)
		advantages = torch.stack(advantages[1:][::-1])
		return advantages, advantages + values[:-1]


	def train(self):
		optimizer = torch.optim.Adam(self.ac.parameters(), lr=learning_rate)
		num_iter = 0
		num_ckpt = 11
		self.ac.load_state_dict(
			torch.load('./10.pth')
		)

		while True:
			num_iter += 1
			states, actions, action_log_probs, rewards, terminals, values = ppo.sample(max_iteration=1024)
			advantages, targets = self.lambda_return(values, rewards, terminals)


			for _ in range(30):
				idx = np.random.randint(0, len(actions), batch_size)
				states_b = states[idx]
				actions_b = actions[idx]
				log_probs_b = action_log_probs[idx]
				advantages_b = advantages[idx]
				targets_b = targets[idx]

				# minibatch advantage normalization
				advantages_b = (advantages_b - advantages_b.mean()) / (advantages_b.std() + 1e-8)

				# actor loss
				_, mean_new, std_new, _ = self.ac.pi(states_b)
				log_probs_new = Normal(mean_new, std_new).log_prob(actions_b).sum(-1)
				ratio = torch.exp(log_probs_new - log_probs_b)
				ratio_clipped = torch.clamp(ratio, 1-clip_ratio, 1 + clip_ratio)
				a_loss = - torch.mean(
					torch.minimum(
						ratio * advantages_b, ratio_clipped * advantages_b
					)
				)

				# critic loss
				v = self.ac.v(states_b)
				v_loss = F.mse_loss(v, targets_b)

				# entropy loss
				e_loss = - torch.mean(
					torch.log(std_new).sum(-1)
				)

				# update
				loss = a_loss + vf_coef * v_loss + ent_coef * e_loss
				optimizer.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
				optimizer.step()

			if num_iter % 200 == 0:
				steps, rewards = self.ac.run_model(max_steps=1000)
				print("%d, %d, %.4f, %.4f" % (steps, rewards, a_loss.item(), v_loss.item()))
			if num_iter % 1000 == 0:
				torch.save(
					self.ac.state_dict(),
					'./%d.pth' % num_ckpt
				)
				num_ckpt += 1


if __name__ == '__main__':
	# ac = ActorCritic()
	# ac.run_model()
	ppo = PPOClip()
	# ppo.train()

	ppo.ac.load_state_dict(torch.load('./15.pth'))
	ppo.ac.run_model(max_steps=1000)
