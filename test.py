import gym
import rl_envs
import numpy as np
import yaml

def load_yaml(path):
    config = yaml.load(open(path, 'r'), Loader=yaml.FullLoader)
    return config

if __name__ == '__main__':
    env = gym.make("intersection-env-v0")
    env.configure(load_yaml('./configs/default.yaml'))
    done = False
    for i in range(20):
        obs = env.reset()
        ep_reward = 0
        while True:
            acc = (np.random.random() - 0.5) * 10.0
            steer = (np.random.random() - 0.5) * np.pi
            action = {"acceleration": acc, "steering": steer}
            next_obs, reward, done, info = env.step(action)
            env.render()
            obs = next_obs
            ep_reward += reward
            if done:
                break
        print("EpReward: {}, Steps: {}.".format(ep_reward, env.steps))
