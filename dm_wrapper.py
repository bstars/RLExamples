import cv2
import numpy as np
from dm_control import suite


class DeepMindWrapper():
	def __init__(self, domain, task):
		self.env = suite.load(domain, task)
		self.action_spec = self.env.action_spec()
		self.camera = dict(quadruped=2).get(domain, 0)

	@staticmethod
	def flatten_observation(obs:dict):
		state = []
		for key, value in obs.items():
			if isinstance(value, np.ndarray):
				state.extend(value.tolist())
			else:
				state.append(value)
		return np.array(state)

	def reset(self):
		time_step = self.env.reset()
		return self.flatten_observation(time_step.observation), None

	def step(self, action):
		time_step = self.env.step(action)
		return self.flatten_observation(time_step.observation), time_step.reward, time_step.last(), None, None

	def render(self, waitkey=30):
		img = self.env.physics.render(camera_id=self.camera)
		img_ = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		cv2.imshow('image', img_)
		cv2.waitKey(waitkey)
		return img

	def random_action(self):
		return np.random.uniform(self.action_spec.minimum,
		                  self.action_spec.maximum,
		                  size=self.action_spec.shape)


if __name__ == '__main__':
	env = DeepMindWrapper('walker', 'walk')
	# print(env.action_spec.maximum, env.action_spec.minimum)
	state, _ = env.reset()
	while True:
		env.render()
		action = env.random_action()
		state, reward, terminal, _, _ = env.step(action)
		print(state.shape, action.shape, reward, terminal)

		if terminal:
			break
