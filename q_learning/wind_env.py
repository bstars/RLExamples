import numpy as np

class WindEnv():
	def __init__(self):
		"""
		Keep the state as an integer
		"""
		self.h = 7
		self.w = 10
		self.moveN = np.array([-1, 0])
		self.moveS = np.array([1, 0])
		self.moveE = np.array([0, 1])
		self.moveW = np.array([0, -1])

		self.action_space = [self.moveN, self.moveS, self.moveW, self.moveE]
		self.state_space = np.arange(0, self.h * self.w)
		self.terminal_state = np.array([3, 7])

	def idx_to_coord(self, idx):
		return np.array([idx // self.w, idx % self.w])

	def coord_to_idx(self, coord):
		return int(coord[0] * self.w + coord[1])

	def reset(self):
		temp = self.coord_to_idx(self.terminal_state)
		state = np.random.choice(
			[i for i in range(temp)] + [i for i in range(temp + 1, len(self.state_space))]
		)
		self.state = state
		self.state = self.coord_to_idx([3, 0])
		self.terminated = False
		return self.state

	def step(self, action, return_pair=False):
		"""

		:param action: int
		:param return_pair: bool, if True, return the coordinate of the state
		:return:
		:rtype:
		"""
		coord = self.idx_to_coord(self.state)
		if np.all(coord == self.terminal_state):
			self.terminated = True
			return coord if return_pair else self.coord_to_idx(coord), 0, True

		end_coord = coord + self.action_space[action]
		if coord[1] in [3,4,5,8]:
			end_coord[0] -= 1
		elif coord[1] in [6,7]:
			end_coord[0] -= 2

		coord = end_coord
		coord[0] = int(np.clip(coord[0], 0, self.h - 1))
		coord[1] = int(np.clip(coord[1], 0, self.w - 1))

		self.terminated = np.all(coord == self.terminal_state)
		self.state = self.coord_to_idx(coord)


		state = coord if return_pair else self.state
		reward = -1 if not self.terminated else 0
		return state, reward, self.terminated





def test_model():
	env = WindEnv()
	env.reset()
	action_space = env.action_space
	i = 0
	while True:
		i += 1

		idx = np.random.choice(
			np.arange(len(action_space))
		)
		# print(env.idx_to_coord(env.state))
		a = action_space[idx]
		state, reward, terminated = env.step(a, return_pair=True)

		print(i, a, state, reward, terminated)
		print()
		if terminated:
			return

if __name__ == '__main__':
	test_model()