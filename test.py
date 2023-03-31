import gym
import procgen

import gym
import procgen

for env_name in procgen.env.ENV_NAMES:
    env_name = f"procgen-{env_name}-v0"
    print("Environment:", env_name)
    env = gym.make(env_name)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("-" * 50)
    env.close()

# d={}
# i=0
# for env_name in procgen.env.ENV_NAMES:
#     env_name = f"procgen-{env_name}-v0"
#     d[i] = env_name
#     i+=1
# print(d)
