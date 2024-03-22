import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import gym



def array_to_tensor(arr):
	return torch.Tensor(arr)

def tensor_to_array(ten):
	return ten.cpu().detach().numpy()

class ActorCritic(nn.Module):
	def __init__(self):
		super().__init__()
		self.actor = nn.Sequential(
			nn.Linear(4, 16), nn.Tanh(),
			nn.Linear(16, 16), nn.Tanh(),
			nn.Linear(16, 16), nn.Tanh(),
			nn.Linear(16, 2), nn.Softmax(dim=1)
		)

		self.critic = nn.Sequential(
			nn.Linear(4, 16), nn.Tanh(),
			nn.Linear(16, 16), nn.Tanh(),
			nn.Linear(16, 16), nn.Tanh(),
			nn.Linear(16, 1)
		)

		self.critic_target = nn.Sequential(
			nn.Linear(4, 16), nn.Tanh(),
			nn.Linear(16, 16), nn.Tanh(),
			nn.Linear(16, 16), nn.Tanh(),
			nn.Linear(16, 1)
		)
		self.update_target()

	def update_target(self):
		self.critic_target.load_state_dict(self.critic.state_dict())

	def forward(self, x):
		return self.critic_target(x)[:,0], self.actor(x)

	def run_model(self):
		env = gym.make("CartPole-v1", render_mode='human')
		state, _ = env.reset()
		steps = 0
		rewards = 0.

		while True:
			steps += 1
			env.render()
			_, actor_prob = self(array_to_tensor(state[None, :]))
			actor_prob = tensor_to_array(actor_prob)[0]
			action = np.argmax(actor_prob)
			state, reward, terminal, truncated, info = env.step(action)
			rewards += reward
			if terminal or truncated:
				print(steps, rewards)
				return steps

class ActorCriticSolver():
	def __init__(self):
		self.env = gym.make("CartPole-v1", render_mode='rgb_array')
		self.state, _ = self.env.reset()
		self.ac = ActorCritic()
		self.gamma = 0.99


	def sample_episode(self, max_iteration=150):
		states = []
		actions = []
		action_probs = []
		rewards = []
		terminals = []
		values = []

		for _ in range(max_iteration):
			value = self.ac.critic_target(array_to_tensor(self.state[None, :]))[:,0]
			action_prob = self.ac.actor(array_to_tensor(self.state[None, :]))
			action = np.random.choice(
				np.arange(2), p=tensor_to_array(action_prob[0])
			)
			state_next, reward, terminal, truncated, info = self.env.step(action)

			states.append(self.state)
			actions.append(action)
			action_probs.append(action_prob[0])
			rewards.append(reward)
			terminals.append(terminal)
			values.append(value)

			self.state = state_next
			if terminal or truncated:
				self.state, _ = self.env.reset()

		value, action_prob = self.ac(array_to_tensor(self.state[None, :]))
		values.append(value)

		return torch.from_numpy(np.stack(states, axis=0)), \
				torch.from_numpy(np.array(actions)), \
				torch.stack(action_probs, dim=0), \
				torch.from_numpy(np.array(rewards)), \
				torch.from_numpy(np.array(terminals)).float(), \
				torch.cat(values, dim=0)


	def solve(self):
		optim = torch.optim.Adam(
			list(self.ac.actor.parameters()) + list(self.ac.critic.parameters()), lr=1e-3
		)
		iterations = 0
		while True:
			iterations += 1
			states, actions, action_probs, rewards, terminals, values = self.sample_episode(max_iteration=20)
			batch = states.shape[0]
			TD_error = rewards + self.gamma * (1-terminals) * values[1:].detach() - self.ac.critic(states)[:,0]
			closs = torch.mean(TD_error ** 2)
			aloss = torch.mean(
				-torch.log(action_probs[np.arange(batch), actions]) * TD_error.detach()
			)
			loss = aloss + closs

			optim.zero_grad()
			loss.backward()
			optim.step()

			if iterations % 50 == 0:
				self.ac.update_target()

			if iterations % 50 == 0:
				steps = self.ac.run_model()
if __name__ == '__main__':
	# ac = ActorCritic()
	# ac.run_model()
	solver = ActorCriticSolver()
	solver.solve()
	# states, actions, action_probs, rewards, terminals, values = solver.sample_episode(max_iteration=20)
	#
	# print(
	# 	states.shape,
	# 	actions.shape,
	# 	action_probs.shape,
	# 	rewards.shape,
	# 	terminals.shape,
	# 	values.shape
	# )