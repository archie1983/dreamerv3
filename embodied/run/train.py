import collections
from functools import partial as bind

import elements
import embodied
import numpy as np


def train(make_agent, make_replay, make_env, make_stream, make_logger, args):
  print("AE: Making Agent")
  agent = make_agent()
  print("AE: Making Replay")
  replay = make_replay()
  print("AE: Making Logger")
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  print("AE: logdir: ", logdir)
  step = logger.step
  print("AE: step: ", step)
  usage = elements.Usage(**args.usage)
  print("AE: usage: ", usage)
  train_agg = elements.Agg()
  print("AE: train_agg: ", train_agg)
  epstats = elements.Agg()
  print("AE: epstats: ", epstats)
  episodes = collections.defaultdict(elements.Agg)
  print("AE: episodes: ", episodes)
  policy_fps = elements.FPS()
  print("AE: policy_fps: ", policy_fps)
  train_fps = elements.FPS()
  print("AE: train_fps: ", train_fps)

  batch_steps = args.batch_size * args.batch_length
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  should_log = embodied.LocalClock(args.log_every)
  should_report = embodied.LocalClock(args.report_every)
  should_save = embodied.LocalClock(args.save_every)
  should_keepmodel = embodied.LocalClock(args.keepmodel_every)

  @elements.timer.section('logfn')
  def logfn(tran, worker):
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')
    for key, value in tran.items():
      if value.dtype == np.uint8 and value.ndim == 3:
        if worker == 0:
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

  print("AE: args: ", args)
  fns = [bind(make_env, i) for i in range(args.envs)]
  print("AE: len(fns): ", len(fns))
  driver = embodied.Driver(fns, parallel=not args.debug)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(replay.add)
  driver.on_step(logfn)

  # We will not just take a sample and train on that sample with reward. What we will do instead
  # is get many training samples and put them into a stream. Then when we've got enough samples,
  # we will construct a batch and train on that batch. This will help us set up a replay pipeline
  # that we will be constructing batches from.
  stream_train = iter(agent.stream(make_stream(replay, 'train')))
  stream_report = iter(agent.stream(make_stream(replay, 'report')))

  carry_train = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)

  def trainfn(tran, worker):
    if len(replay) < args.batch_size * args.batch_length:
      return
    # We have some number of steps done at this point. So we go through them and construct batches from them
    # according to our configuration, then we feed those batches to the network for training.
    for _ in range(should_train(step)):
      with elements.timer.section('stream_next'):
        # And this is where we construct a batch from the accumulated steps so that we can train on it.
        batch = next(stream_train)
      print('b', sep='', end='')
      carry_train[0], outs, mets = agent.train(carry_train[0], batch)
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
      train_agg.add(mets, prefix='train')
  driver.on_step(trainfn)

  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = agent
  cp.replay = replay
  if args.from_checkpoint:
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=args.from_checkpoint_regex)))
  cp.load_or_save()

  print('Start training loop')
  policy = lambda *args: agent.policy(*args, mode='train')
  driver.reset(agent.init_policy)
  while step < args.steps:

    driver(policy, steps=10)

    if should_report(step) and len(replay):
      agg = elements.Agg()
      for _ in range(args.consec_report * args.report_batches):
        carry_report, mets = agent.report(carry_report, next(stream_report))
        agg.add(mets)
      logger.add(agg.result(), prefix='report')

    if should_log(step):
      print("AE: Writing LOG")
      logger.add(train_agg.result())
      logger.add(epstats.result(), prefix='epstats')
      logger.add(replay.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

    if should_save(step):
      cp.save()

    # We want to store the model at the given intervals for evaluation
    if should_keepmodel(step):
      keep_cp = elements.Checkpoint(logdir / 'kept_models', keep=20, step=step)
      keep_cp.agent = cp.agent
      keep_cp.replay = cp.replay
      keep_cp.step = cp.step
      keep_cp.save()

  logger.close()
