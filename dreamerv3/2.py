
from dm_control import suite
from dm_control import viewer
import numpy as np
import time
import os
os.environ['DISPLAY'] = ':100'
print("Display set to:", os.environ.get('DISPLAY'))

print("Loading environment...")
env = suite.load(domain_name="cartpole", task_name="balance")

def random_policy(time_step):
    return np.random.uniform(-1, 1, size=env.action_spec().shape)

print("Launching viewer... (waiting 5 seconds)")
viewer.launch(env, policy=random_policy)
time.sleep(5)  # Give the window time to appear
print("Viewer should be visible now")
