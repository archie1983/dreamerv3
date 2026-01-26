import time

import cloudpickle
import elements
import numpy as np
import portal
import traceback


class Driver:

  def __init__(self, make_env_fns, parallel=True, **kwargs):
    assert len(make_env_fns) >= 1
    self.parallel = parallel
    self.kwargs = kwargs
    self.length = len(make_env_fns)
    if parallel:
      import multiprocessing as mp
      context = mp.get_context()
      print("AE pipes count: ", self.length)
      self.pipes, pipes = zip(*[context.Pipe() for _ in range(self.length)])
      self.stop = context.Event()
      fns = [cloudpickle.dumps(fn) for fn in make_env_fns]
      self.procs = [
          portal.Process(self._env_server, self.stop, i, pipe, fn, start=True)
          for i, (fn, pipe) in enumerate(zip(fns, pipes))]
      self.pipes[0].send(('act_space',))
      self.act_space = self._receive(self.pipes[0])
    else:
      self.envs = [fn() for fn in make_env_fns]
      self.act_space = self.envs[0].act_space
    self.callbacks = []
    self.acts = None
    self.carry = None
    self.driver_retired = False
    self.reset()

  def reset(self, init_policy=None):
    self.acts = {
        k: np.zeros((self.length,) + v.shape, v.dtype)
        for k, v in self.act_space.items()}
    self.acts['reset'] = np.ones(self.length, bool)

    # AE: Debug
    #print("AE: ACTS after reset: ", self.acts)
    #for key, value in self.acts.items():
    #  print("AE: ACTS after reset: key: ", key, " val: ", value)

    # AE: init_policy is a function that lives inside Agent class. Why are we assigning here the function and the boolean AND-ing of its results?
    self.carry = init_policy and init_policy(self.length)
    #print("AE, driver.py: self.carry: ", self.carry)

  def close(self):
    print("AE: closing pipes")
    #traceback.print_stack()
    if self.parallel:
      [proc.kill() for proc in self.procs]
    else:
      [env.close() for env in self.envs]

  def on_step(self, callback):
    self.callbacks.append(callback)

  def __call__(self, policy, steps=0, episodes=0):
    step, episode = 0, 0
    while step < steps or episode < episodes:
      step, episode = self._step(policy, step, episode)

  def _step(self, policy, step, episode):
    # AE: Here we have a collection of actions in something like this: {'action': array([0, 2], dtype=int32), 'reset': array([False, False])}
    # The size of the action array in the dictionary is 2 here because in this case we had 2 AI2-Thor environments
    # running- each providing an observation which lead to an action. This collection of actions was acquired in this
    # same step function- just in the previous iteration.
    acts = self.acts
    #print("AE, driver.py: acts: ", acts)
    #print("AE, driver.py: acts['action']: ", acts['action'])
    # AE: Here we assert that the number of actions in the action list matches the environment count. Same for the reset flags.
    assert all(len(x) == self.length for x in acts.values())
    # AE: And that all of them are lists (even if there's only 1 environment)
    assert all(isinstance(v, np.ndarray) for v in acts.values())
    # AE: Here we now start separating actions and reset states in sets meant for each environment. So the above example
    # will turn into: [{'action': np.int32(0), 'reset': np.False_}, {'action': np.int32(2), 'reset': np.False_}]
    acts = [{k: v[i] for k, v in acts.items()} for i in range(self.length)]
    #print("AE, driver.py: acts2: ", acts)
    #print("AE, driver.py: acts2[0]: ", acts[0])
    #print("AE, driver.py: acts2[0]['action']: ", acts[0]['action'])
    # AE: Now we pass each action to the environment that it is meant to go to. The enviornment will process it and
    # provide an observation as a feedback.
    if self.parallel:
      [pipe.send(('step', act)) for pipe, act in zip(self.pipes, acts)]
      obs = [self._receive(pipe) for pipe in self.pipes]
    else:
      obs = [env.step(act) for env, act in zip(self.envs, acts)]
      ## Now let's look at all the envs and if all of them are retired, then we want to mark this driver as retired too
      self.driver_retired = True
      for i in range(self.length):
        #print("ENV ", i, " retired = ", self.envs[i].env_retired)
        self.driver_retired = self.driver_retired and self.envs[i].env_retired
    #print("AE, driver.py: obs[0].keys(): ", obs[0].keys())
    # AE: As a reminder, the keys of observation for our case are: ['image', 'reward', 'is_first', 'is_last', 'is_terminal']
    # Now we
    obs = {k: np.stack([x[k] for x in obs]) for k in obs[0].keys()}
    logs = {k: v for k, v in obs.items() if k.startswith('log/')}
    obs = {k: v for k, v in obs.items() if not k.startswith('log/')}
    assert all(len(x) == self.length for x in obs.values()), obs
    #print("AE, driver.py: self.carry: ", self.carry) # jaxlib._jax.XlaRuntimeError: INVALID_ARGUMENT: Disallowed device-to-host transfer: shape=(8192), dtype=BF16, device=cuda:0
    # This is where we get the new actions from the policy, the observation (obs) and previous state (self.carry)
    #self.carry, acts, outs, cont = policy(self.carry, obs, **self.kwargs)
    self.carry, acts, outs = policy(self.carry, obs, **self.kwargs)
    #print("AE, driver.py: acts3: ", acts)
    assert all(k not in acts for k in outs), (
        list(outs.keys()), list(acts.keys()))
    if obs['is_last'].any():
      mask = ~obs['is_last']
      acts = {k: self._mask(v, mask) for k, v in acts.items()}
    self.acts = {**acts, 'reset': obs['is_last'].copy()}
    #print("AE, driver.py: self.acts4: ", self.acts)
    #TODO: Inspect trans = {**obs, **acts, **outs, **logs} to understand where each field comes from.
    # Specifically continue flag, is_first, is_last and steps left to do, etc.
    trans = {**obs, **acts, **outs, **logs}
    #print("AE outs: ", outs, " logs: ", logs, " acts: ", acts, " cont: ", cont)
    #print("AE outs: ", outs, " logs: ", logs, " acts: ", acts)
    #print("AE obs: ", obs)
    #print("AE term comp: ", outs['cont'], " :: ", obs['is_last'])
    #print("AE  term comp8: ", (outs['cont'] > 0.8) == obs['is_terminal'], " @ (", outs['cont'], " > 0.8) == ", obs['is_terminal'])
    #print("AE  term comp3: ", (outs['cont'] < 0.3) == obs['is_terminal'], " @ (", outs['cont'], " < 0.3) == ", obs['is_terminal'])
    #if ((outs['cont'] > 0.8) == obs['is_terminal']):
    #if ((outs['cont'] < 0.3) == obs['is_terminal']):
    #  print("1", outs['cont'], ' # ', outs['distanceleft'], end=' + ')
    #else:
    #  print("0", outs['cont'], ' # ', outs['distanceleft'], end=' +')
    #print("AE outs: ", outs)
    #if obs['is_terminal']:
    #  print(outs['cont'], ' # ', outs['distanceleft'], end='')
    #print(obs['distanceleft'], ' . ', outs['distanceleft'], " ", outs['cont'], " ", obs['is_terminal'])
    for i in range(self.length):
      trn = elements.tree.map(lambda x: x[i], trans)
      [fn(trn, i, **self.kwargs) for fn in self.callbacks]
    step += len(obs['is_first'])
    episode += obs['is_last'].sum()
    return step, episode

  def _mask(self, value, mask):
    while mask.ndim < value.ndim:
      mask = mask[..., None]
    return value * mask.astype(value.dtype)

  def _receive(self, pipe):
    try:
      msg, arg = pipe.recv()
      if msg == 'error':
        raise RuntimeError(arg)
      assert msg == 'result'
      return arg
    except Exception:
      print('Terminating workers due to an exception.')
      [proc.kill() for proc in self.procs]
      raise

  @staticmethod
  def _env_server(stop, envid, pipe, ctor):
    '''
    In the original code we were swallowing exceptions and being quiet as a member of resistance,
    but if we do that, then it's of course hard to debug when it ends abruptly. So, e.g. if you
    set run.debug = False in the configs.yaml file, then it will bomb out in evaluation mode with
    a cryptic message that has little to do with the real error. Therefore we should squeak here
    about what errors happened.
    :param stop:
    :param envid:
    :param pipe:
    :param ctor:
    :return:
    '''
    try:
      ctor = cloudpickle.loads(ctor)
      env = ctor()
      while not stop.is_set():
        if not pipe.poll(0.1):
          time.sleep(0.1)
          continue
        try:
          msg, *args = pipe.recv()
        except EOFError:
          print("AE: driver.py, EOFError")
          return
        if msg == 'step':
          assert len(args) == 1
          act = args[0]
          obs = env.step(act)
          pipe.send(('result', obs))
        elif msg == 'obs_space':
          assert len(args) == 0
          pipe.send(('result', env.obs_space))
        elif msg == 'act_space':
          assert len(args) == 0
          pipe.send(('result', env.act_space))
        else:
          raise ValueError(f'Invalid message {msg}')
    except ConnectionResetError:
      print('Connection to driver lost')
    except Exception as e:
      print("AE: driver.py, some other error", e)
      pipe.send(('error', e))
      raise
    finally:
      print("AE: driver.py, env server terminating")
      try:
        env.close()
      except Exception:
        pass
      pipe.close()
