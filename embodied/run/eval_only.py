from collections import defaultdict
from functools import partial as bind

import elements
import embodied
import numpy as np


def eval_only(make_agent, make_env, make_logger, args):
  assert args.from_checkpoint

  # AE: This make_agent function is defined in main.py and it basically constructs an Agent defined in jax/agent.py
  # Once we have it, we can then load stored weights from a checkpoint. That happens towards the end of this function.
  agent = make_agent()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  logdir.mkdir()
  print('Logdir', logdir)
  step = logger.step
  usage = elements.Usage(**args.usage)
  agg = elements.Agg()
  epstats = elements.Agg()
  episodes = defaultdict(elements.Agg)
  should_log = elements.when.Clock(args.log_every)
  policy_fps = elements.FPS()

  @elements.timer.section('logfn')
  def logfn(tran, worker):
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')
    for key, value in tran.items():
      isimage = (value.dtype == np.uint8) and (value.ndim == 3)
      if isimage and worker == 0:
        episode.add(f'policy_{key}', value, agg='stack')
      elif key.startswith('log/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')
    if tran['is_last']:
      result = episode.result()
      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length'),
      }, prefix='episode')
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      epstats.add(result)

  fns = [bind(make_env, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=(not args.debug))
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(logfn)
  print("AE: args.from_checkpoint: ", args.from_checkpoint)
  cp = elements.Checkpoint()
  print("AE: agent: ", agent)

  # AE: elements.Checkpoint object has a __setattr__(self, name, value) function which allows to set
  # arbitrary attributes with arbitrary values. These values need to implement a load or save method.
  # If, e.g., the value is an object that implements load method, then we can later call cp.load and get
  # the passed object loaded up with the checkpoint weights.
  cp.agent = agent
  # AE: In this case, what we're loading is an Agent object defined in jax/agent.py which is a superclass to
  # dreamerv3/agent.py. And this agent here is actually constructed using mage_agent function which is passed
  # from main.py and therefore this agent is the one from dreamerv3/agent.py. Ultimately we construct it by passing
  # observation space, action space and configuration.
  cp.load(args.from_checkpoint, keys=['agent'])

  print('Start evaluation')
  # AE: Are we here passing args one by one to agent's policy net and collecting results into policy variable?
  # AE: Here we simply define a function (using lambda) that will take in *args and return agent.policy called
  # on those args. We only want to create this function, not call it yet. This function will be passed to driver
  # function as a function pointer and will be called from there.
  policy = lambda *args: agent.policy(*args, mode='eval')
  driver.reset(agent.init_policy)
  while step < args.steps:
    driver(policy, steps=10)
    if should_log(step):
      logger.add(agg.result())
      logger.add(epstats.result(), prefix='epstats')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

  logger.close()
