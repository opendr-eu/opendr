import sys

from human_detection_env import Env

env = Env(args = sys.argv)

env.reset()
while True:
    obs, reward, dones, _ = env.step(0)
    if dones:
        obs = env.reset()
        print("DONE")
