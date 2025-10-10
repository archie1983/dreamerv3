import importlib
import os
import pathlib
import sys
from functools import partial as bind

folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))
__package__ = folder.name

import elements
import embodied
import numpy as np
import portal
import ruamel.yaml as yaml


def main(argv=None):
  from .agent import Agent
  [elements.print(line) for line in Agent.banner]

  configs = elements.Path(folder / 'configs.yaml').read()
  configs = yaml.YAML(typ='safe').load(configs)
  parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
  #print("AE parsed, other, argv: ", parsed, other, argv)
  config = elements.Config(configs['defaults'])

  for name in parsed.configs:
    config = config.update(configs[name])
  config = elements.Flags(config).parse(other)
  config = config.update(logdir=(
      config.logdir.format(timestamp=elements.timestamp())))
  print("AE conf: ", config)
  if 'JOB_COMPLETION_INDEX' in os.environ:
    config = config.update(replica=int(os.environ['JOB_COMPLETION_INDEX']))
  print('Replica:', config.replica, '/', config.replicas)

  logdir = elements.Path(config.logdir)
  print('Logdir:', logdir)
  print('Run script:', config.script)
  if not config.script.endswith(('_env', '_replay')):
    logdir.mkdir()
    config.save(logdir / 'config.yaml')

  def init():
    elements.timer.global_timer.enabled = config.logger.timer

  portal.setup(
      errfile=config.errfile and logdir / 'error',
      clientkw=dict(logging_color='cyan'),
      serverkw=dict(logging_color='cyan'),
      initfns=[init],
      ipv6=config.ipv6,
  )

  args = elements.Config(
      **config.run,
      replica=config.replica,
      replicas=config.replicas,
      logdir=config.logdir,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      report_length=config.report_length,
      consec_train=config.consec_train,
      consec_report=config.consec_report,
      replay_context=config.replay_context,
  )

  print("AE: config.script: ", config.script)
  if config.script == 'train':
    # Now we will call the "train" function from embodied.run
    # and we will pass function pointers to it as parameters. A function pointer for making agent, for replaying,
    # for making environment, etc.
    print("AE: calling TRAIN")
    embodied.run.train(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args)
    print("AE: TRAIN Exited")

  elif config.script == 'train_eval':
    embodied.run.train_eval(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'eval_replay', 'eval'),
        bind(make_env, config),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args)

  elif config.script == 'eval_only':
    embodied.run.eval_only(
        bind(make_agent, config),
        bind(make_env, config),
        bind(make_logger, config),
        args)

  elif config.script == 'parallel':
    embodied.run.parallel.combined(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'replay_eval', 'eval'),
        bind(make_env, config),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args)

  elif config.script == 'parallel_env':
    is_eval = config.replica >= args.envs
    embodied.run.parallel.parallel_env(
        bind(make_env, config), config.replica, args, is_eval)

  elif config.script == 'parallel_envs':
    is_eval = config.replica >= args.envs
    embodied.run.parallel.parallel_envs(
        bind(make_env, config), bind(make_env, config), args)

  elif config.script == 'parallel_replay':
    embodied.run.parallel.parallel_replay(
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'replay_eval', 'eval'),
        bind(make_stream, config),
        args)

  else:
    raise NotImplementedError(config.script)

##
# Crucially, make_agent calls make_env and invokes the environment.
##
def make_agent(config):
  from .agent import Agent
  env = make_env(config, 0)
  notlog = lambda k: not k.startswith('log/')
  obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
  act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
  env.close()
  print("AE: obs_space: ", obs_space)
  print("AE: act_space: ", act_space)
  if config.random_agent:
    return embodied.RandomAgent(obs_space, act_space)
  cpdir = elements.Path(config.logdir)
  cpdir = cpdir.parent if config.replicas > 1 else cpdir
  print("AE: cpdir: ", cpdir)
  # AE: At this point we have action space, observation space and full configuration.
  # We will not instantiate an Agent based on this information. This agent will later
  # be used to load pretrained Agent's weight in run/eval_only.py
  return Agent(obs_space, act_space, elements.Config(
      **config.agent,
      logdir=config.logdir,
      seed=config.seed,
      jax=config.jax,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      replay_context=config.replay_context,
      report_length=config.report_length,
      replica=config.replica,
      replicas=config.replicas,
  ))


def make_logger(config):
  step = elements.Counter()
  logdir = config.logdir
  multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
  outputs = []
  outputs.append(elements.logger.TerminalOutput(config.logger.filter, 'Agent'))
  for output in config.logger.outputs:
    if output == 'jsonl':
      outputs.append(elements.logger.JSONLOutput(logdir, 'metrics.jsonl'))
      outputs.append(elements.logger.JSONLOutput(
          logdir, 'scores.jsonl', 'episode/score'))
    elif output == 'tensorboard':
      outputs.append(elements.logger.TensorBoardOutput(
          logdir, config.logger.fps))
    elif output == 'expa':
      exp = logdir.split('/')[-4]
      run = '/'.join(logdir.split('/')[-3:])
      proj = 'embodied' if logdir.startswith(('/cns/', 'gs://')) else 'debug'
      outputs.append(elements.logger.ExpaOutput(
          exp, run, proj, config.logger.user, config.flat))
    elif output == 'wandb':
      name = '/'.join(logdir.split('/')[-4:])
      outputs.append(elements.logger.WandBOutput(name))
    elif output == 'scope':
      outputs.append(elements.logger.ScopeOutput(elements.Path(logdir)))
    else:
      raise NotImplementedError(output)
  logger = elements.Logger(step, outputs, multiplier)
  return logger


def make_replay(config, folder, mode='train'):
  batlen = config.batch_length if mode == 'train' else config.report_length
  consec = config.consec_train if mode == 'train' else config.consec_report
  capacity = config.replay.size if mode == 'train' else config.replay.size / 10
  length = consec * batlen + config.replay_context
  assert config.batch_size * length <= capacity

  directory = elements.Path(config.logdir) / folder
  if config.replicas > 1:
    directory /= f'{config.replica:05}'
  kwargs = dict(
      length=length, capacity=int(capacity), online=config.replay.online,
      chunksize=config.replay.chunksize, directory=directory)

  if config.replay.fracs.uniform < 1 and mode == 'train':
    assert config.jax.compute_dtype in ('bfloat16', 'float32'), (
        'Gradient scaling for low-precision training can produce invalid loss '
        'outputs that are incompatible with prioritized replay.')
    recency = 1.0 / np.arange(1, capacity + 1) ** config.replay.recexp
    selectors = embodied.replay.selectors
    kwargs['selector'] = selectors.Mixture(dict(
        uniform=selectors.Uniform(),
        priority=selectors.Prioritized(**config.replay.prio),
        recency=selectors.Recency(recency),
    ), config.replay.fracs)

  return embodied.replay.Replay(**kwargs)


def make_env(config, index, **overrides):
  suite, task = config.task.split('_', 1)
  #print("AE suite, task: ", suite, task)
  if suite == 'memmaze':
    from embodied.envs import from_gym
    import memory_maze  # noqa
  # Create a dictionaty called ctor and immediately select the environment
  # from that dictionary as defined in suite. E.g., dmlab. This is so reminiscent of JAVA structure, is Danijar Hafner
  # a JAVA guy?
  ctor = {
      'dummy': 'embodied.envs.dummy:Dummy',
      'gym': 'embodied.envs.from_gym:FromGym',
      'dm': 'embodied.envs.from_dmenv:FromDM',
      'crafter': 'embodied.envs.crafter:Crafter',
      'dmc': 'embodied.envs.dmc:DMC',
      'atari': 'embodied.envs.atari:Atari',
      'atari100k': 'embodied.envs.atari:Atari',
      'dmlab': 'embodied.envs.dmlab:DMLab',
      'minecraft': 'embodied.envs.minecraft:Minecraft',
      'loconav': 'embodied.envs.loconav:LocoNav',
      'pinpad': 'embodied.envs.pinpad:PinPad',
      'langroom': 'embodied.envs.langroom:LangRoom',
      'procgen': 'embodied.envs.procgen:ProcGen',
      'bsuite': 'embodied.envs.bsuite:BSuite',
      'memmaze': lambda task, **kw: from_gym.FromGym(
          f'MemoryMaze-{task}-v0', **kw),
      'ai2thorae': 'embodied.envs.ai2thor_ae:AI2ThorEnv',
      'indoors': 'embodied.envs.indoors:Indoors',
      'indoorseval': 'embodied.envs.indoorseval:IndoorsEval',
  }[suite]
  if isinstance(ctor, str):
    # Split the selected env string into components, e.g. it could be 'embodied.envs.dmlab:DMLab'
    module, cls = ctor.split(':')
    # Load the selected module, e.g., embodied.envs.dmlab
    module = importlib.import_module(module)
    # Now find the required class in that module, e.g., DMLab
    ctor = getattr(module, cls)
    #print("AE module, cls, ctor: ", module, cls, ctor)
  kwargs = config.env.get(suite, {})
  kwargs.update(overrides)
  if kwargs.pop('use_seed', False):
    kwargs['seed'] = hash((config.seed, index)) % (2 ** 32 - 1)
  if kwargs.pop('use_logdir', False):
    kwargs['logdir'] = elements.Path(config.logdir) / f'env{index}'

  # AE: Specially for our env, we want to pass to it what script this is: eval, train or something else
  if (suite == "ai2thorae"):
      kwargs["mode"] = config.script

  # Now call that class, instantiate it (e.g., DMLab class from embodied.envs.dmlab).
  env = ctor(task, **kwargs)
  # Now use that instance of the environment class- wrap_env it and then return whatever it returns
  print("AE: ENV:: ", env)
  return wrap_env(env, config)

##
# Essentially unifying data types and checking spaces (the building block data type of environments)
# Not entirely sure the purpose of it, but it's probably important... and boring.
##
def wrap_env(env, config):
  # Go through action space
  for name, space in env.act_space.items():
    print("AE: ENV:: ", name, space)
    # If action space is not discrete, then normalize it. Not sure what it means in practice.
    if not space.discrete:
      env = embodied.wrappers.NormalizeAction(env, name)
      print("AE: Normalized ENV:: ", env, name)

  env = embodied.wrappers.UnifyDtypes(env)
  print("AE: Post Unify Dtypes ENV:: ", env, name)
  env = embodied.wrappers.CheckSpaces(env)
  print("AE: Post check spaces ENV:: ", env, name)

  for name, space in env.act_space.items():
    if not space.discrete:
      env = embodied.wrappers.ClipAction(env, name)
  return env


def make_stream(config, replay, mode):
  fn = bind(replay.sample, config.batch_size, mode)
  stream = embodied.streams.Stateless(fn)
  stream = embodied.streams.Consec(
      stream,
      length=config.batch_length if mode == 'train' else config.report_length,
      consec=config.consec_train if mode == 'train' else config.consec_report,
      prefix=config.replay_context,
      strict=(mode == 'train'),
      contiguous=True)

  return stream


if __name__ == '__main__':
  main()
