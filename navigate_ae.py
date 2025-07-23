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

    configs = elements.Path(folder / 'dreamerv3/configs.yaml').read()
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
    print('Run script:', config.script)

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

    embodied.run.navigate_AI2_thor(
        bind(make_agent, config),
        bind(make_env, config),
        args)

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

def make_env(config, index, **overrides):
    #suite, task = config.task.split('_', 1)
    #ai2thorae_10
    task = 10 # Choose an AI2-Thor training habitat here. E.g, 10 will end up being "train_10"
    ctor = 'embodied.envs.ai2thor_ae:AI2ThorEnv'
    if isinstance(ctor, str):
        # Split the selected env string into components, e.g. it could be 'embodied.envs.dmlab:DMLab'
        module, cls = ctor.split(':')
        # Load the selected module, e.g., embodied.envs.dmlab
        module = importlib.import_module(module)
        # Now find the required class in that module, e.g., DMLab
        ctor = getattr(module, cls)
        #print("AE module, cls, ctor: ", module, cls, ctor)
    kwargs = config.env.get("ai2thorae", {})
    kwargs.update(overrides)
    if kwargs.pop('use_seed', False):
        kwargs['seed'] = hash((config.seed, index)) % (2 ** 32 - 1)
    if kwargs.pop('use_logdir', False):
        kwargs['logdir'] = elements.Path(config.logdir) / f'env{index}'
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

if __name__ == '__main__':
    main()
