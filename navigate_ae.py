import importlib
import os, time
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

class Navigator():
    def __init__(self, agent, env, config):
        self.agent = agent
        self.env = env
        self.config = config

    ##
    # Here we will look at the action space and create a vector for all actions for each entry in a batch.
    # Since we are trying to navigate, then batch length = 1 and our actions are of course just simple
    # numbers themselves rather than vectors, so it will all be a simple collection of likeness to:
    # [[0, 1, 2, 3]]
    # Reset flag is just a boolen (set to 1 at first)
    # And importantly we will get another tensor which we call self.carry. It will contain initial states
    # for Encoder, Decoder and our RSSM latent space model. Encoder and Decoder intial states are just empty
    # sets, but RSSM (which we call dyn here) has tensors representing stochastic and deterministic state (initialized to
    # something of course, because we haven't yet started representing the world with them.
    ##
    def reset(self):
        length = 1 # Change this to some other number if you want batches of other sizes for some reason
        self.acts = {
            k: np.zeros((length,) + v.shape, v.dtype)
            for k, v in self.env.act_space.items()}
        self.acts['reset'] = np.ones(length, bool)
        # AE: init_policy is a function that lives inside Agent class. Why are we assigning here the function and the boolean AND-ing of its results?
        self.carry = self.agent.init_policy(length)

        # AE: Debug
        print("AE: ENV ACTS: ", self.env.act_space)
        print("AE: ACTS: ", self.acts)

        for key, value in self.acts.items():
            print("AE: ACTS: key: ", key, " val: ", value)

        ## turning list values into discrete values, e.g.:
        # key:  action  val:  [0]
        # key:  reset  val:  [ True]
        # INTO:
        # key:  action  val:  0
        # key:  reset  val:  True
        acts = [{k: v[i] for k, v in self.acts.items()} for i in range(1)]

        # AE: Debug
        print("AE: ENV ACTS2: ", self.env.act_space)
        print("AE: ACTS2: ", acts)

        for key, value in acts[0].items():
            print("AE: ACTS2[0]: key: ", key, " val: ", value)

        # AE: Do the intitial step with the action['reset'] == True to get the first observation
        print("AE: reset::acts[0] : ", acts[0])
        # at this point acts[0] ==  {'action': np.int32(0), 'reset': np.True_}
        self.obs = self.env.step(acts[0])
        #print("AE, driver.py: self.carry: ", self.carry)
        # Now that we have self.carry, we can start inferencing from our world model - feed it observations and get navigation decisions.

    ##
    # Look at current observation and RSSM state and infer the next best action
    ##
    def navigation_step(self):
        #obs = {k: np.stack([x[k] for x in self.obs]) for k in self.obs[0].keys()}
        #print("AE: self.obs: ", self.obs)
        obs = {k: np.stack([self.obs[k]]) for k in self.obs.keys()}
        obs = {k: v for k, v in obs.items() if not k.startswith('log/')}
        # print("AE, driver.py: self.carry: ", self.carry) # jaxlib._jax.XlaRuntimeError: INVALID_ARGUMENT: Disallowed device-to-host transfer: shape=(8192), dtype=BF16, device=cuda:0
        #self.carry, acts, outs = self.agent.policy(self.carry, obs, **self.config)
        self.carry, acts, outs = self.agent.policy(self.carry, obs, mode="eval_only")
        print("AE, driver.py: acts3: ", acts)
        assert all(k not in acts for k in outs), (
            list(outs.keys()), list(acts.keys()))
        if obs['is_last'].any():
            mask = ~obs['is_last']
            acts = {k: self._mask(v, mask) for k, v in acts.items()}
        self.acts = {**acts, 'reset': obs['is_last'].copy()}
        print("AE, driver.py: self.acts4: ", self.acts)
        trans = {**obs, **acts, **outs}

        # Now that we have the best action according to Actor, we can execute that action in the environment
        print("AE: navigation_step::self.acts ", self.acts)
        #self.acts = {k: self.acts[k].item() for k in self.acts}
        acts = [{k: v[i] for k, v in self.acts.items()} for i in range(1)]
        print("AE: navigation_step::self.acts2 ", acts)
        self.obs = self.env.step(acts[0])

DEBUG = False
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

    if DEBUG:
        embodied.run.navigate_AI2_thor(
            bind(make_agent, config),
            bind(make_env, config),
            args)
    else:
        agent = make_agent(config)
        cp = elements.Checkpoint()
        cp.agent = agent
        # We can also load checkpoints or parts of a checkpoint from a different directory.
        cp.load("/home/hp20024/robotics/latent_planning/dreamer_models/room_centre_1", keys=['agent'])
        #print(cp.agent)

        # Now make environment. TODO: We really should re-use the environment that was created inside make_agent.
        env = make_env(config, 0)
        # Now we have an agent and the environment. Let's load the Navigator so that it starts its job
        navig = Navigator(cp.agent, env, config)
        navig.reset()
        for i in range(100):
            navig.navigation_step()
            time.sleep(0.1)

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
