import logging, threading, elements, random, socket, traceback
import numpy as np
from ai2_thor_model_training.training_data_extraction import RobotNavigationControl
from ai2_thor_model_training.ae_utils import (NavigationUtils, action_mapping,
                                              action_to_index, index_to_action, inverted_action_mapping,
                                              AI2THORUtils, get_path_length, get_centre_of_the_room,
                                              room_this_point_belongs_to, get_rooms_ground_truth,
                                              get_all_objects_of_type, is_point_inside_room_ground_truth,
                                              create_full_grid_from_room_layout, add_buffer_to_unreachable, RoomType)
from ai2_thor_model_training.connection import recv_data, send_data

import thortils as tt
from thortils import launch_controller
from thortils.agent import thor_reachable_positions
from thortils.utils import roundany, getch
from thortils.utils.math import sep_spatial_sample

from shapely.geometry import Point
import pickle, json

np.float = float
np.int = int
np.bool = bool

class Env:
  def __repr__(self):
    return (
        f'{self.__class__.__name__}('
        f'obs_space={self.obs_space}, '
        f'act_space={self.act_space})')
  @property
  def obs_space(self):
    # The observation space must contain the keys is_first, is_last, and
    # is_terminal. Commonly, it also contains the keys reward and image. By
    # convention, keys starting with 'log/' are not consumed by the agent.
    raise NotImplementedError('Returns: dict of spaces')
  @property
  def act_space(self):
    # The action space must contain the reset key as well as any actions.
    raise NotImplementedError('Returns: dict of spaces')
  def step(self, action):
    raise NotImplementedError('Returns: dict')
  def close(self):
    pass

class Wrapper:
  def __init__(self, env):
    self.env = env
  def __len__(self):
    return len(self.env)
  def __bool__(self):
    return bool(self.env)
  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self.env, name)
    except AttributeError:
      raise ValueError(name)

class TimeLimit(Wrapper):
  def __init__(self, env, duration, reset=True):
    super().__init__(env)
    self._duration = duration
    self._reset = reset
    self._step = 0
    self._done = False
  def step(self, action):
    #print(self._step, " ", end='', sep='')
    if action['reset'] or self._done:
      self._step = 0
      self._done = False
      if self._reset:
        action.update(reset=True)
        return self.env.step(action)
      else:
        action.update(reset=False)
        obs = self.env.step(action)
        obs['is_first'] = True
        return obs
    self._step += 1
    obs = self.env.step(action)
    if self._duration and self._step >= self._duration:
      print("AE: Episode Duration Exceeded -- Terminating Episode : ", self._step, " / ", self._duration)
      obs['is_last'] = True
    self._done = obs['is_last']
    return obs

class Roomcentre(Wrapper):

    def __init__(self, *args, **kwargs):
        self.logdir = kwargs["logdir"]
        actions = action_mapping

        # Actions
        actions = actions.copy()
        if "STOP" in actions:
            actions.pop("STOP")  # remove STOP action because that will be treated differently

        self.rewards = [
            DistanceReductionReward(),
            TargetAchievedRewardRoomCentre(),
        ]
        length = kwargs.pop('length', 36000)
        env = RoomCentreFinder(actions, *args, **kwargs)
        self.unwrapped_env = env
        env = TimeLimit(env, length)
        super().__init__(env)

    def step(self, action):
        obs = self.env.step(action)
        reward = sum([fn(obs) for fn in self.rewards])
        obs['reward'] = np.float32(reward)

        if obs['is_last'] and not self.unwrapped_env.env_retired and self.unwrapped_env.hab_set != "train":
            episode_stats = {
                "final_reward": str(obs['reward']),
            }
            with open(self.logdir + "/episode_data.jsonl", "a") as f:
                f.write(json.dumps(episode_stats) + "\n")

        # we may not want to train on distance_left parameter, but if we pop it, then wrappers complain,
        # so perhaps it can stay for now.
        #obs.pop("distance_left")
        return obs

class Door(Wrapper):

    def __init__(self, *args, **kwargs):
        self.logdir = kwargs["logdir"]
        #print("DI1")
        actions = action_mapping
        #print("*args: ", args, " **kwargs: ", kwargs)
        reward_close_enough = kwargs["reward_close_enough"]

        # Actions
        actions = actions.copy()
        if "STOP" in actions:
            actions.pop("STOP")  # remove STOP action because that will be treated differently

        self.rewards = [
            DistanceReductionReward(scale=1.0),
            TargetAchievedRewardForDoor(epsilon=reward_close_enough)
        ]
        length = kwargs.pop('length', 36000)
        #print("AE: len", length)
        env = DoorFinder(actions, *args, **kwargs)
        self.unwrapped_env = env
        env = TimeLimit(env, length)
        super().__init__(env)
        #print("DI2")

    def step(self, action):
        #print("A1")
        obs = self.env.step(action)
        reward = sum([fn(obs) for fn in self.rewards])
        obs['reward'] = np.float32(reward)
        #print("A2")

        if obs['is_last'] and not self.unwrapped_env.env_retired and self.unwrapped_env.hab_set != "train":
            episode_stats = {
                "final_reward": str(obs['reward']),
            }
            with open(self.logdir + "/episode_data.jsonl", "a") as f:
                f.write(json.dumps(episode_stats) + "\n")

        # we may not want to train on distance_left parameter, but if we pop it, then wrappers complain,
        # so perhaps it can stay for now.
        #obs.pop("distance_left")
        return obs

        # Introduce a marker on the image that points towards the door that we want to go to. That would allow input and
        # training guidance to navigate to a specific door, not just a random door. Introduce room field in observation so that we can
        # classify target achieved when we change rooms.

##
# Using this class, we can stack objectives of the agent behaviour. E.g., to first achieve the
# middle of the room and only then look for the doors. Or even find all doors in order.
##
class DistanceReductionReward:
    def __init__(self, scale=1.0):
        self.scale = scale
        self.prev_distance = None
        self.best_distance_so_far = None

    def __call__(self, obs, inventory=None):
        #print("D1")
        reward = 0.0
        #distance_left = obs['distance_left']
        distance_left = obs['distanceleft']

        if obs['is_first']:
            self.best_distance_so_far = distance_left
        else:
            if self.best_distance_so_far > distance_left:
                '''
                if we improved best distance, then reward is the improvement factor
                '''
                #reward = self.scale * (self.best_distance_so_far - distance_left)
                reward = 1
                self.best_distance_so_far = distance_left
                #print("BIG REW: ", reward)
                #print("r", reward, end="", sep="")
                print("r", end="", sep="")
            elif self.best_distance_so_far == distance_left:
                '''
                if no improvement, then bigger penalty. No movement needs to be discouraged
                '''
                reward = -0.25
            elif self.best_distance_so_far < distance_left and self.prev_distance < distance_left:
                '''
                if we have moved away from the target, then penalty by the reduction
                '''
                #reward = self.scale * (self.prev_distance - distance_left)
                reward = -0.5
            elif self.best_distance_so_far < distance_left and self.prev_distance > distance_left:
                '''
                if we have improved our position from last time, but not yet the best path, then small reward
                '''
                reward = 0.25
            elif self.best_distance_so_far < distance_left and self.prev_distance == distance_left:
                '''
                if no improvement since last time, then penalty to discourage not moving
                '''
                reward = -0.25
            else:
                '''
                shouldn't happen. If it does, then the above code has error.
                '''
                print("CHECK DistanceReductionReward CODE!!!")
                exit()

        self.prev_distance = distance_left
        #print("D2")

        return np.float32(reward)

##
# Issue a reward for achieving the target - once per scene
##
class TargetAchievedRewardForDoor:
    def __init__(self, epsilon = 0.0, steps_in_new_room = 3):
        '''
        :param epsilon: How close is close enough to issue the reward
        '''
        self.reward_issued = False
        self.steps_in_new_room = steps_in_new_room
        self.epsilon = epsilon

    def __call__(self, obs, inventory=None):
        #print("T1")
        reward = 0
        if obs['is_first']:
            self.reward_issued = False
        #elif not self.reward_issued and (obs['distance_left'] <= self.epsilon or obs['steps_after_room_change'] >= self.steps_in_new_room):
        elif not self.reward_issued and (obs['distanceleft'] <= self.epsilon or obs['stepsafterroomchange'] >= self.steps_in_new_room):
            reward = 20
            self.reward_issued = True
        #print("T2")
        return np.float32(reward)

##
# Issue a reward for achieving the target - once per scene
##
class TargetAchievedRewardRoomCentre:
    def __init__(self, epsilon = 0.0, steps_in_new_room = 3):
        '''
        :param epsilon: How close is close enough to issue the reward
        '''
        self.reward_issued = False
        self.steps_in_new_room = steps_in_new_room
        self.epsilon = epsilon

    def __call__(self, obs, inventory=None):
        #print("T1")
        reward = 0
        if obs['is_first']:
            self.reward_issued = False
        #elif not self.reward_issued and (obs['distance_left'] <= self.epsilon or obs['steps_after_room_change'] >= self.steps_in_new_room):
        elif not self.reward_issued and obs['distanceleft'] <= self.epsilon:
            reward = 20
            self.reward_issued = True
        #print("T2")
        return np.float32(reward)

class CollectReward:

    def __init__(self, item, once=0, repeated=0):
        self.item = item
        self.once = once
        self.repeated = repeated
        self.previous = 0
        self.maximum = 0

    def __call__(self, obs, inventory):
        current = inventory[self.item]
        if obs['is_first']:
            self.previous = current
            self.maximum = current
            return 0
        reward = self.repeated * max(0, current - self.previous)
        if self.maximum == 0 and current > 0:
            reward += self.once
        self.previous = current
        self.maximum = max(self.maximum, current)
        return reward

class AI2ThorBase(Env):

    LOCK = threading.Lock()

    hab_exploration_stats_collection = []

    def __init__(self,
                 actions,
                 logdir="not_set",
                 repeat=1,
                 size=(64, 64),
                 logs=False,
                 hab_space=(100, 600),
                 hab_set="train",
                 places_per_hab=20,
                 grid_size=0.125,
                 reward_close_enough=0.125,
                 plan_close_enough=0.25,
                 env_index=-1,
                 server_ip='0.0.0.0',
                 port=9999,
                 encoding='utf-8'
                 ):
        '''

        :param actions:
        :param repeat:
        :param size:
        :param logs:
        :param hab_space:
        :param hab_set:
        :param places_per_hab:
        :param grid_size:
        :param reward_close_enough:
        :param plan_close_enough:
        :param env_index: If this is anything other than -1, then we are evaluating with 3 envs and we want to split
            the hab_space into three and only use one portion per env. This could be improved by also specifying the
            number of envs, not just the index, but for now we will work with the assumption that the number of envs is 3.
        '''
        #print("C1")
        if logs:
            logging.basicConfig(level=logging.DEBUG)

        # if we have an env_index, then we assume that there are 3 envs and we will split hab_space between those
        # 3 envs and assign a portion to the current env according to its index.
        if env_index > -1:
            (hab_min, hab_max) = hab_space
            hab_diff = hab_max - hab_min
            hab_step = int(hab_diff / 3)
            hab_starts = range(hab_min, hab_max, hab_step)
            new_hab_min = hab_starts[env_index]
            new_hab_max = (new_hab_min + hab_step - 1) if env_index < 2 else (new_hab_min + hab_step)
            hab_space = (new_hab_min, new_hab_max)

        # AE: AI2-Thor simulation stuff
        self.atu = AI2THORUtils()
        self.rooms_in_habitat = None
        self.current_path_length = 1000
        self.reachable_positions = None
        self.grid_size = grid_size # how fine do we want the 2D grid to be.
        self.reward_close_enough = reward_close_enough # how close to the target is close enough for the purposes of reward. If we're this close or closer in simulation to the target, then consider it done
        self.plan_close_enough = plan_close_enough # how close to the target is close enough for the purposes of path planning. We may end up planning path to a point anywhere near the actual target by this much
        self.nu = NavigationUtils(step=self.grid_size)
        # If we get into a bad spot from which for whatever reason we can't plan a path out, then we'll set this to
        # True and based on it will teleport to a new place when we see this set.
        self._bad_spot = False
        self._bad_spot_cnt = 0
        self._total_reward_for_this_run = 0
        self.step_count_in_current_episode = 0
        self.step_count_since_start = 0
        self.distance_left = np.float32(0.0)
        self.room_type = -1 # current room type
        self.starting_room = None # which room we end up in when we spawn
        self.target_room = None # which room we want to end up in
        self.current_room = None # which room are we in now
        self.steps_in_new_room = 0 # how many steps have we made inside the new room since we first stepped into the target room (resets if we leave target room)
        self.env_retired = False # in some cases we want to be able to signal to driver.py that this env does not need driving anymore. This will help with that.
        self.prev_obs = None

        # When we store the statistics of each test run, we will want to capture these variables
        self.astar_path = []
        self.path_start = None
        self.path_dest = None
        self.travelled_path = []
        self.chosen_actions = []
        self.logdir = logdir

        # Remote connection stuff
        self.server_ip = server_ip
        self.port = port
        self.encoding = encoding
        self.client_socket = None

        print("AE hab_space:", hab_space, " logdir: ", logdir)
        #traceback.print_stack()
        # AE: based on whether we're training or evaluating, we will want to use different subsets of the habitat set
        (self.hab_min, self.hab_max) = hab_space
        self.hab_set = hab_set
        self.places_per_hab = places_per_hab

        self.choose_habitats_randomly_or_sequentially = True
        if (hab_set == "train"):
            print(f"AE Training on : {type(self).__name__}")
            self.choose_habitats_randomly_or_sequentially = True
        elif (hab_set == "test"):
            print(f"AE Testing on : {type(self).__name__}")
            self.choose_habitats_randomly_or_sequentially = False
        else:
            print(f"AE ?Validation? on : {type(self).__name__}")
            self.choose_habitats_randomly_or_sequentially = False

        # when we select a random position and plan path to the room centre, we will assign a value to this parameter
        # with the A* path length from that random position to the desired point. This will help calculate reward from all
        # further points.
        self.initial_path_length = 0

        # If we use reward that tracks the best length of the path left, then we will need this variable
        self.best_path_length = 0

        # We will need to keep track of the target point that we want to reach because we will be re-planning path to it
        # from all sorts of different points.
        self.current_target_point = None

        # upon beginning we don't have any habitat loaded yet, but we will check this variable to determine if we have
        self.habitat_id = None
        self.explored_placements_in_current_habitat = []

        # Dreamer stuff
        self._size = size
        self._repeat = repeat
        self.isFirst = False

        # Here we connect remotely to an environment elsewhere, because we can't run AI2-Thor on a Jetson
        with self.LOCK:
            self.client_socket = self.connect_to_server()

        self._step = 0
        self._obs_space = self.obs_space

        self._action_names = tuple(actions.keys())
        self._action_values = tuple(actions.values())
        message = f'Indoor Navigation action space ({len(self._action_values)}):'
        print(message, ', '.join(self._action_names))
        #print("C2")

    def connect_to_server(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            print(f"Attempting to connect to server at {self.server_ip}:{self.port}...")
            self.client_socket.connect((self.server_ip, self.port))
            print("✅ Connected to server.")

            self.load_next_start_point_remotely()

            ## 1. SEND INITIAL COMMAND
            #initial_command = {"command": "INIT", "hab_id": "83", "hab_set": "test"}
            #send_data(self.client_socket, json.dumps(initial_command).encode(self.encoding))

            ## Await READY response
            #init_response_bytes = recv_data(self.client_socket)
            #if not init_response_bytes:
            #    raise Exception("Server failed to send initialization response.")

            #init_response = json.loads(init_response_bytes.decode(self.encoding))
            #if init_response.get("status") == "READY":
            #    print(f"Server initialized scene: {init_response.get('scene')}")
            #    self.reachable_positions = init_response.get("reachable_positions")
            #    self.unreachable_postions = init_response.get("unreachable_postions")
            #    self.full_grid = init_response.get("full_grid")

            ## print("self.reachable_positions: ", self.reachable_positions)
            #else:
            #    raise Exception("Server reported initialization failure.")

        # Example of how to access other data:
        # print(f"Agent Position: {metadata['agent']['position']}")
        except ConnectionRefusedError:
            print(f"❌ Connection Refused. Ensure server is running at {self.server_ip}:{self.port} and firewall is open.")
            self.close_client_socket()
        except Exception as e:
            print(f"An error occurred: {e}")
            self.close_client_socket()

        return self.client_socket

    def load_next_start_point_remotely(self):
        if self.client_socket:
            try:
                # 1. SEND INITIAL COMMAND
                initial_command = {"command": "NEXT_POINT"}
                send_data(self.client_socket, json.dumps(initial_command).encode(self.encoding))

                # Await READY response
                init_response_bytes = recv_data(self.client_socket)
                if not init_response_bytes:
                    raise Exception("Server failed to send initialization response.")

                init_response = json.loads(init_response_bytes.decode(self.encoding))
                if init_response.get("msg") == "HAB_AND_POS":
                    self.hab_id = init_response.get("hab_id")
                    self.cur_pos = init_response.get("cur_pos")
                    self.current_target_point = init_response.get("current_target_point")
                    self.initial_path_length = init_response.get("initial_path_length")
                    self.astar_path = init_response.get("astar_path")
                    self.path_start = init_response.get("path_start")
                    self.path_dest = init_response.get("path_dest")
                    self.starting_room = init_response.get("starting_room")
                    self.current_path_length = init_response.get("current_path_length")
                    self.best_path_length = init_response.get("best_path_length")
                else:
                    raise Exception("HAB_AND_POS failed to return.")

            # Example of how to access other data:
            # print(f"Agent Position: {metadata['agent']['position']}")
            except Exception as e:
                print(f"An error occurred: {e}")
                self.close_client_socket()

    def close_client_socket(self):
        self.client_socket.close()
        print("Client disconnected.")

    @property
    def obs_space(self):
        return {
            'image': elements.Space(np.uint8, self._size + (3,)),
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
            #'distance_left': elements.Space(np.float32),
            #'steps_after_room_change': elements.Space(np.float32),
            #'room_type': elements.Space(np.float32),
            'distanceleft': elements.Space(np.float32),
            'stepsafterroomchange': elements.Space(np.float32),
            'roomtype': elements.Space(np.float32),
        }

    @property
    def act_space(self):
        return {
            'action': elements.Space(np.int32, (), 0, len(self._action_values)),
            'reset': elements.Space(bool),
        }

    def step(self, action):
        # If this env has been retired (in evaluation mode we have evaluated everything already), then
        # don't actually do any stepping, but just return the previous obs
        if self.env_retired:
            return self.prev_obs

        #print("S1")
        action = action.copy()
        #index = action.pop('action')
        #print("action: ", action, " self._action_values: ", self._action_values, " inder:", index)
        #action.update(self._action_values[index])
        #print("action: ", action)

        if action['reset']:
            print('R', end='', sep='')
            # STORE EPISODE STATS:
            # A* path length, A* path, travelled path length, travelled path, habitat id, actions taken.
            if self.hab_set != "train":
                episode_stats = {
                    "local_step": self.step_count_since_start,
                    "steps_used": self.step_count_in_current_episode,
                    "habitat_id": self.habitat_id,
                    "bad_spot": self._bad_spot,
                    "have_arrived": str(self.have_we_arrived(self.reward_close_enough)),
                    "path_start": self.path_start,
                    "path_dest": self.path_dest,
                    "astar_path": self.astar_path,
                    "travelled_path": self.travelled_path,
                    "chosen_actions": self.chosen_actions,
                }
                #print(hab_exploration_stats)

                with open(self.logdir + "/episode_data.jsonl", "a") as f:
                    f.write(json.dumps(episode_stats) + "\n")

            obs = self._reset()
        else:
            raw_action = index_to_action(int(action['action']))
            self.rnc.execute_action(raw_action, moveMagnitude=self.grid_size, grid_size=self.grid_size, adhere_to_grid=True)
            self.chosen_actions.append(int(action['action']))
            # This is slightly ugly, but we need to calculate distance_left variable right after rnc.execute_action
            # to allow observation to be up to date. In time this should be moved to some function instead of relying
            # on global variables.
            try:
                self.distance_left, self.room_type, cur_pos_xy = self.get_current_path_and_pose_state()
                self.travelled_path.append(cur_pos_xy)
                self._done = self.have_we_arrived(self.reward_close_enough)
            except ValueError as e:
                self.distance_left = np.float32(0.0)
                self._bad_spot = True

            if self._bad_spot:
                #print("FORCED SCENE CHANGE!!!", self.step_count_in_current_episode)
                #STORE EPISODE STATS
                print('O', end='', sep='')
                ##
                # This must be self._done = True instead of direct reset. We will reset in the next loop
                ##
                #obs = self._reset()
                self._done = True
            #else:
            obs = self.current_ai2thor_observation()

        # Now we turn the obs that was returned by the environment into obs that we use for training,
        # and to not confuse the two, make sure that 'pov' field is not there, because it should be 'image'.
        obs = self._obs(obs)
        self._step += 1
        self.step_count_in_current_episode += 1
        self.step_count_since_start += 1
        assert 'pov' not in obs, list(obs.keys())
        #print("S2")
        self.prev_obs = obs
        return obs

    ##
    # Returns current observation of the state (image mostly)
    ##
    def current_ai2thor_observation(self):
        #print("O1")
        event = self.controller.last_event
        self._current_image = event.cv2img

        #print("self.current_room == self.target_room", self.current_room, self.target_room)
        # if we're in the target room, then count how many steps we've done in the target room
        self.steps_in_new_room = self.steps_in_new_room + 1 if self.current_room == self.target_room else 0

        obs = dict(
            reward = 0.0,
            pov = self._current_image,
            is_first = np.bool(self.isFirst),
            is_last = np.bool(self._done),
            is_terminal = np.bool(self._done),
            #distance_left = np.float32(self.distance_left),
            #steps_after_room_change = np.float32(self.steps_in_new_room),
            #room_type = np.float32(self.room_type),
            distanceleft=np.float32(self.distance_left),
            stepsafterroomchange=np.float32(self.steps_in_new_room),
            roomtype=np.float32(self.room_type),
        )
        if self._done:
            print('D', sep='', end='')

        self.isFirst = False # this will have to be set to True when we reset the env
        #print("O2")
        return obs

    def _reset(self):
        #print("R1")
        self.astar_path = []
        self.path_start = None
        self.path_dest = None
        self.travelled_path = []
        self.chosen_actions = []
        # Load new point or even a habitat, set reward to 0 and is_first = True and is_last = False and self._done = False
        with self.LOCK:
            self.load_next_start_point_remotely()

        self.step_count_in_current_episode = 0
        self._step = 0
        self._done = False
        self._bad_spot = False

        self.distance_left = 0
        self.steps_in_new_room = 0
        self.room_type = -1
        self.starting_room = None

        obs = self.current_ai2thor_observation()
        #print("R2")
        return obs

    def _obs(self, obs):
        #print("_O1")
        obs = {
            'image': obs['pov'],
            'reward': np.float32(0.0), # reward will be calculated later
            'is_first': obs['is_first'],
            'is_last': obs['is_last'],
            'is_terminal': obs['is_terminal'],
            #'distance_left': obs['distance_left'],
            #'steps_after_room_change': obs['steps_after_room_change'],
            #'room_type': obs['room_type'],
            'distanceleft': obs['distanceleft'],
            'stepsafterroomchange': obs['stepsafterroomchange'],
            'roomtype': obs['roomtype'],
            # 'log/player_pos': np.array([player_x, player_y, player_z], np.float32),
        }
        #print("obs: ", obs)
        for key, value in obs.items():
            space = self._obs_space[key]
            if not isinstance(value, np.ndarray):
                value = np.array(value)
            #print("val: ", value, " space: ", space, " key: ", key, " (key, value, @dtype@, value.shape, space): ", (key, value, value.shape, space))
            assert value in space, (key, value, value.dtype, value.shape, space)
        #print("obs: ", obs)
        #print("_O2")
        return obs

    def create_rnd_object(self):
        seed = 1983
        if not hasattr(self, "rnd"):
            self.rnd = random.Random(seed)
        return self.rnd

    # Determines if we have little enough left to call it an achieved goal
    def have_we_arrived(self, epsilon = 0.0):
        pass

    def close(self):
        if (self.controller != None):
            self.controller.stop()

    ##
    # Chooses the target point that we want to navigate to, given current position.
    # This will have to be implemented in derived classes.
    ##
    def choose_target_point(self, place_with_rtn = None, place_with_no_rtn = None):
        pass
        # if (self.doors_or_centre):
        #     self.current_target_point = self.choose_door_target(place_with_rtn)
        #     # print("self.current_target_point: ", self.current_target_point)
        # else:
        #     point_for_room_search = (p[0], "", p[1])
        #     # print("point_for_room_search: ", point_for_room_search)
        #     self.current_target_point = self.find_room_centre_target(point_for_room_search)

    def find_room_type_of_this_point(self, point_for_room_search):
        '''
        We want to find out an ID for the room type that this point belongs to
        :param point_for_room_search:
        :return:
        '''
        #print("FR1")
        room_of_placement = room_this_point_belongs_to(self.rooms_in_habitat, point_for_room_search)
        if room_of_placement == None:
            return None
        room_type = room_of_placement[0]
        room_type = np.uint8(RoomType.interpret_label(room_type).value)
        #print("FR2")
        return room_type

##
# Room centre finding task
##
class RoomCentreFinder(AI2ThorBase):
    def __init__(self, actions, *args, **kwargs):
        super().__init__(actions, *args, **kwargs)

    def choose_target_point(self, place_with_rtn = None, place_with_no_rtn = None):
        return self.find_room_centre_target(place_with_no_rtn)

    ##
    # Finds the centre of the current room given the current position and the rooms in habitat.
    ##
    def find_room_centre_target(self, point_for_room_search):
        #print("FRC1")
        # We've just been put in a random place in a habitat. We want to move now to where we want to go,
        # e.g., middle of the room, a door, etc.. For that we need to plan a path to there.
        room_of_placement = room_this_point_belongs_to(self.rooms_in_habitat, point_for_room_search)

        if (room_of_placement == None): raise ValueError("Room of placement not identifiable")

        room_centre = room_of_placement[2]
        #print("FRC2")
        return room_centre

    # Determines if we have little enough left to call it an achieved goal
    def have_we_arrived(self, epsilon = 0.0):
        return (self.current_path_length <= epsilon)

##
# Door Finding Task
##
class DoorFinder(AI2ThorBase):
    def __init__(self, actions, *args, **kwargs):
        super().__init__(actions, *args, **kwargs)

    def choose_target_point(self, place_with_rtn = None, place_with_no_rtn = None):
        return self.choose_door_target(place_with_rtn)

    ##
    # Go through all the doors and find the most appropriate as a target, then add a little bit extra so that
    # we end up going through the door.
    # place_with_rtn: Place with rotation, e.g.: (6.62, 6.25, 180)
    ##
    def choose_door_target(self, place_with_rtn):
        #print("CD1")
        current_target_point = None
        try:
            current_target_point = self.nu.find_door_target(place_with_rtn,
                                                            self.rooms_in_habitat,
                                                            self.reachable_positions,
                                                            self.habitat,
                                                            self.controller, close_enough=self.plan_close_enough,
                                                            step=self.grid_size, extend_path=True)

            # t1 = time.time()
            pose = ((place_with_rtn[0], 0.0, place_with_rtn[1]),
                    (0.0, place_with_rtn[2], 0.0))  # place_with_rtn in AI2-Thor format
            path_length = self.nu.get_path_cost_to_target_point(pose,
                                                                current_target_point,
                                                                self.reachable_positions,
                                                                close_enough=self.plan_close_enough,
                                                                step=self.grid_size)

            # if we've been successful so far, then we can now look up room type
            trg_pos_xy = (current_target_point.x, "", current_target_point.y)
            self.target_room = room_this_point_belongs_to(self.rooms_in_habitat, trg_pos_xy)
            # print("AE: path plan time: ", (time.time() - t1))
        except ValueError as e:
            path_length = 0
            # print("AE: No Path Found", e)
            print('£', end='', sep='')

        if (self.target_room == None):
            raise ValueError("Target room not identifiable")

        if current_target_point == None:
            raise ValueError("No suitable doors were found")

        # print("AE: Path Length: ", path_length)
        #(cur_path, reachable_positions, start, dest) = self.nu.get_last_path_and_params()
        # print("AE: Path: ", cur_path)
        # atu.visualise_path2(cur_path, reachable_positions, unreachable_postions, rooms_in_habitat, start, dest,
        #                    show_unreachable_pos=False,
        #                    show_reachable_pos=False)
        # atu.visualise_path2(cur_path, reachable_positions, buf_unreachable_pos, rooms_in_habitat, start, dest, show_unreachable_pos=True)
        #print("CD2")
        return current_target_point

    # Determines if we have little enough left to call it an achieved goal
    def have_we_arrived(self, epsilon = 0.0):
        return (self.current_path_length <= epsilon or self.steps_in_new_room >= 3)
