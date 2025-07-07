from dm_control import suite
from dm_control import viewer

env = suite.load(domain_name="cartpole", task_name="balance")

def random_policy(time_step):
    return np.random.uniform(-1, 1, size=env.action_spec().shape)

viewer.launch(env, policy=random_policy)

# AE
