import logging, threading, elements, random, embodied, traceback
import numpy as np
from ai2_thor_model_training.training_data_extraction import RobotNavigationControl
from ai2_thor_model_training.ae_utils import (NavigationUtils, action_mapping,
                                              action_to_index, index_to_action, inverted_action_mapping,
                                              AI2THORUtils, get_path_length, get_centre_of_the_room,
                                              room_this_point_belongs_to, get_rooms_ground_truth,
                                              get_all_objects_of_type, is_point_inside_room_ground_truth,
                                              create_full_grid_from_room_layout, add_buffer_to_unreachable, RoomType)

import thortils as tt
from thortils import launch_controller
from thortils.agent import thor_reachable_positions
from thortils.utils import roundany, getch
from thortils.utils.math import sep_spatial_sample

from shapely.geometry import Point
import pickle

np.float = float
np.int = int
np.bool = bool

class Roomcentre(embodied.Wrapper):

    def __init__(self, *args, **kwargs):
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
        env = embodied.wrappers.TimeLimit(env, length)
        super().__init__(env)

    def step(self, action):
        obs = self.env.step(action)
        reward = sum([fn(obs) for fn in self.rewards])
        obs['reward'] = np.float32(reward)

        # we may not want to train on distance_left parameter, but if we pop it, then wrappers complain,
        # so perhaps it can stay for now.
        #obs.pop("distance_left")
        return obs

class Door(embodied.Wrapper):

    def __init__(self, *args, **kwargs):
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
        env = embodied.wrappers.TimeLimit(env, length)
        super().__init__(env)
        #print("DI2")

    def step(self, action):
        #print("A1")
        obs = self.env.step(action)
        reward = sum([fn(obs) for fn in self.rewards])
        obs['reward'] = np.float32(reward)
        #print("A2")

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

class AI2ThorBase(embodied.Env):

    LOCK = threading.Lock()

    hab_exploration_stats_collection = []

    def __init__(self,
                 actions,
                 repeat=1,
                 size=(64, 64),
                 logs=False,
                 hab_space=(100, 600),
                 hab_set="train",
                 places_per_hab=20,
                 grid_size=0.125,
                 reward_close_enough=0.125,
                 plan_close_enough=0.25
                 ):
        #print("C1")
        if logs:
            logging.basicConfig(level=logging.DEBUG)

        # AE: AI2-Thor simulation stuff
        self.rnc = RobotNavigationControl()
        self.controller = None
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

        print("AE hab_space:", hab_space)
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

        # Here we create the first AI2Thor environment (controller)
        with self.LOCK:
            self.load_next_start_point()

        self._step = 0
        self._obs_space = self.obs_space

        self._action_names = tuple(actions.keys())
        self._action_values = tuple(actions.values())
        message = f'Indoor Navigation action space ({len(self._action_values)}):'
        print(message, ', '.join(self._action_names))
        #print("C2")

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
        #print("S1")
        action = action.copy()
        #index = action.pop('action')
        #print("action: ", action, " self._action_values: ", self._action_values, " inder:", index)
        #action.update(self._action_values[index])
        #print("action: ", action)

        if action['reset']:
            print('R', end='', sep='')
            obs = self._reset()
        else:
            raw_action = index_to_action(int(action['action']))
            self.rnc.execute_action(raw_action, grid_size=self.grid_size, adhere_to_grid=True)

            # This is slightly ugly, but we need to calculate distance_left variable right after rnc.execute_action
            # to allow observation to be up to date. In time this should be moved to some function instead of relying
            # on global variables.
            try:
                self.distance_left, self.room_type = self.get_current_path_and_pose_state()
                self._done = self.have_we_arrived(self.reward_close_enough)
            except ValueError as e:
                self.distance_left = np.float32(0.0)
                self._bad_spot = True

            if self._bad_spot:
                #print("FORCED SCENE CHANGE!!!", self.step_count_in_current_episode)
                obs = self._reset()
            else:
                obs = self.current_ai2thor_observation()

        # Now we turn the obs that was returned by the environment into obs that we use for training,
        # and to not confuse the two, make sure that 'pov' field is not there, because it should be 'image'.
        obs = self._obs(obs)
        self._step += 1
        self.step_count_in_current_episode += 1
        self.step_count_since_start += 1
        assert 'pov' not in obs, list(obs.keys())
        #print("S2")
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
        # Load new point or even a habitat, set reward to 0 and is_first = True and is_last = False and self._done = False
        with self.LOCK:
            self.load_next_start_point()
            #obs = self._env.step({'reset': True})

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

    def load_random_habitat(self):
        #print("LRH1")
        # choose a random habitat from a space of given habitats by self.hab_max and self.hab_min
        loaded = False

        # we are going to choose a completely new habitat now. Before we do that, we want to register somewhere
        # what habitat was being explored up until now and what placements were looked at in there.
        if len(self.explored_placements_in_current_habitat) > 0:
            hab_exploration_stats = {
                "local_step": self.step_count_since_start,
                "habitat_id": self.habitat_id,
                "explored_placements_in_current_habitat": self.explored_placements_in_current_habitat
            }
            #print(hab_exploration_stats)
            AI2ThorBase.hab_exploration_stats_collection.append(hab_exploration_stats)
            with open("stat_store", "wb") as stat_store:
                pickle.dump(AI2ThorBase.hab_exploration_stats_collection, stat_store)
        # now that we've saved previous habitat exploration stats, we can carry on with a new habitat

        while not loaded:
            try:
                if (self.choose_habitats_randomly_or_sequentially): # if we want a random habitat (e.g. we're training)
                    sp = elements.Space(np.int32, (), self.hab_min, self.hab_max)
                    self.habitat_id = sp.sample()
                else:
                    # if we want a sequential habitat (e.g. we're evaluating or testing)
                    if (self.habitat_id == None):
                        self.habitat_id = self.hab_min
                    elif (self.habitat_id < self.hab_max):
                        self.habitat_id += 1
                    else:
                        # we're done, we need to terminate the evaluation process now
                        exit()

                # load_habitat will also call self.choose_random_placement_in_habitat(), which will in turn calculate
                # current distance cost to the target
                self.load_habitat(self.habitat_id)
                # enfore at least 2 rooms in a habitat
                if len(self.rooms_in_habitat) >= 2:
                    loaded = True
            except ValueError as e:
                continue
        #print("LRH2")
    ##
    # This kind of combines 2 functions: load_random_habitat and choose_random_placement_in_habitat.
    # The idea is that usually we only want to load a different starting point within the same habitat,
    # but sometimes we will want to load a new habitat entirely. Also, if we have exhausted all usable random
    # places in the given habitat, then we want to load a new habitat. This is all best handled in one place-
    # this function.
    ##
    def load_next_start_point(self):
        #print("L1")
        # if nothing has been loaded, then we just load a brand new habitat - Simple
        if self.habitat_id is None:
            self.load_random_habitat()
        else:
            # otherwise, we want to look at what have we explored and what is available
            # if we have already explored 20 random locations in this habitat, then it's time to move on
            if len(self.explored_placements_in_current_habitat) > self.places_per_hab:
                self.load_random_habitat()
            else:
                # otherwise try to load the next random placement (it will attempt a few times, currently 10).
                # If that fails, then we load new habitat.
                try:
                    self.choose_random_placement_in_habitat()
                except ValueError as e:
                    self.load_random_habitat()

        self.isFirst = True # we just loaded a new scene or habitat. The next observation will be first
        #print("L2")

    ##
    # Load the given habitat- load it, and put agent in a random place
    ##
    def load_habitat(self, habitat_id):
        #print("LH1")
        # load required habitat
        #print("AE: haba: ", habitat_id)
        self.habitat = self.atu.load_proctor_habitat(int(habitat_id), self.hab_set)
        self.explored_placements_in_current_habitat = []

        # Launch a controller for the loaded habitat. If we already have a controller,
        # then reset it instead of loading a new one.
        if (self.controller == None):
            self.controller = launch_controller({"scene": self.habitat,
                                                 "VISIBILITY_DISTANCE": 3.0,
                                                 "headless": True,
                                                 "IMAGE_WIDTH": 64,
                                                 "IMAGE_HEIGHT": 64,
                                                 "GRID_SIZE": self.grid_size,
                                                 "GPU_DEVICE": 1,
                                                 # "RENDER_DEPTH": False,
                                                 # "RENDER_INSTANCE_SEGMENTATION": False,
                                                 # "RENDER_IMAGE": True
                                                 # "IMAGE_WIDTH": 64,
                                                 # "IMAGE_HEIGHT": 64
                                                 })
            self.rnc.set_controller(self.controller)  # This allows our control scripts to interact with AI2-THOR environment
            self.atu.set_controller(self.controller)
        else:
            self.controller.reset(self.habitat)
            # self.reset_state()
            self.rnc.reset_state()
            # self.rnc.set_controller(self.controller)

        # Take a snapshot of all available positions- these won't change while we're in this habitat,
        # so no need to re-do them everytime we plan a path.
        #self.grid_size = self.controller.initialization_parameters["gridSize"]
        self.reachable_positions, self.unreachable_postions, self.full_grid, self.rooms_in_habitat = self.update_navigation_artifacts(self.habitat)

        # Now place the robot in a random position and figure out the target from there.
        self.choose_random_placement_in_habitat()
        #self.choose_specific_placement_in_habitat()
        #print("LH2")

    def choose_specific_placement_in_habitat(self):
        #params["position"] = dict(x=7.0, y=0.9009997844696045, z=5.625)
        #params["rotation"] = dict(x=0.0, y=270, z=0.0)
        # self.controller.step(action="Teleport", **pos_navigate_to)
        place_with_rtn = (5.62, 3.5, 270)
        self.rnc.teleport_to(place_with_rtn)
        try:
            self.current_target_point = self.choose_door_target(place_with_rtn)
        except ValueError as err:
            print("O", err)

        #print("AE: Path Length: ", path_length)
        (cur_path, reachable_positions, start, dest) = self.nu.get_last_path_and_params()
        print("AE: Path: ", cur_path)
        self.atu.visualise_path2(cur_path, self.reachable_positions, self.unreachable_postions, self.rooms_in_habitat, start, dest,
                            show_unreachable_pos = False,
                            show_reachable_pos = True)
        k = getch()

        cur_pos = self.rnc.get_agent_pos_and_rotation()
        target_point = Point(5.12, 4.0)#((3.12, 0.88, 4.75), (0.0, 180, 0.0))
        door_path_length = self.nu.get_path_cost_to_target_point(cur_pos,
                                                                         target_point,
                                                                         self.reachable_positions,
                                                                         close_enough=self.plan_close_enough,
                                                                         step=self.grid_size,
                                                                         debug=True)
        print("AE: door_path_length: ", door_path_length)
        k = getch()

    ##
    # Here we will select a number of random placements and then choose one to navigate from it
    # to some goal.
    ##
    def choose_random_placement_in_habitat(self):
        #print("CH1")
        ## All we need is a set of random positions and we get them like this:
        # params for the random teleportation part
        seed = 1983
        num_stops = 20
        num_rotates = 4
        sep = 1.0
        v_angles = [30]
        h_angles = [0, 45, 90, 135, 180, 225, 270, 315]

        """
        num_stops: Number of places the agent will be placed
        num_rotates: Number of random rotations at each place
        sep: the minimum separation the sampled agent locations should have
    
        kwargs: See thortils.vision.projection.open3d_pcd_from_rgbd;
        """
        ## If we are training (i.e., loading habitats randomly), then don't use a seed
        if (self.choose_habitats_randomly_or_sequentially):
            rnd = random.Random()
        else: # if, on the other hand, we are evaluating or testing (loading habitats sequentially), then test everything the same way- use a seed
            rnd = random.Random(seed)

        initial_agent_pose = tt.thor_agent_pose(self.controller)
        initial_horizon = tt.thor_camera_horizon(self.controller.last_event)

        # reachable_positions = tt.thor_reachable_positions(self.controller)
        # self.reachable_positions
        placements = sep_spatial_sample(self.reachable_positions, sep, num_stops, rnd=rnd)

        # print(placements)

        # Choose one placement in the set of placements and then plan path from that placement to
        # the middle of the room. If planning path is not possible, then choose another one.
        path_planned = False
        placement_attempts = 0
        while not path_planned:
            placement_attempts += 1
            #els = elements.Space(np.int32, (), 0, len(placements))
            #p = list(placements)[int(els.sample())]
            el_ndx = rnd.randrange(0, len(placements))
            p = list(placements)[el_ndx]
            # p = placements.pop()

            # append a rotation to the place.
            yaw = rnd.sample(h_angles, 1)[0]
            place_with_rtn = p + (yaw,)
            #print("Placement: ", place_with_rtn)
            self.explored_placements_in_current_habitat.append(place_with_rtn)
            ## Teleport, then start new exploration. Achieve goal. Then repeat.
            self.rnc.teleport_to(place_with_rtn)

            # We've just been put in a random place in a habitat. We want to move now to where we want to go,
            # e.g., middle of the room, a door, etc.. For that we need to plan a path to there.
            try:
                point_for_room_search = (p[0], "", p[1])
                self.current_target_point = self.choose_target_point(place_with_rtn, point_for_room_search) # self.target_room will be set in this function

                cur_pos = self.rnc.get_agent_pos_and_rotation()
                self.initial_path_length = self.nu.get_path_cost_to_target_point(cur_pos,
                                                                                 self.current_target_point,
                                                                                 self.reachable_positions,
                                                                                 close_enough=self.plan_close_enough,
                                                                                 step=self.grid_size)

                if isinstance(self, DoorFinder):
                    # what is the room we start in
                    self.starting_room = room_this_point_belongs_to(self.rooms_in_habitat, point_for_room_search)

                    # We must ensure that we navigate from one room to another
                    if self.target_room == self.starting_room: raise ValueError("start and end points in same room")
            except ValueError as e:
                # If the path could not be planned, then drop it and carry on with the next one
                #print(f"ERROR: {e}")
                if placement_attempts <= 10:
                    print('.', sep='', end='')
                    continue
                else:
                    # If we have tried for 10 times already, then give up with this habitat
                    print("next_hab", sep="", end="")
                    raise e

            # print("PATH & PLAN: ", path_and_plan)
            #            path = path_and_plan[0]
            #            plan = path_and_plan[1]
            # self.prev_pose = thor_agent_pose(self.controller)  # This is where we are before the plan started
            # place_with_rtn
            # thor_pose_as_tuple(self.prev_pose)
            #print("AE poses: place_with_rtn: ", place_with_rtn, " p: ", p, " self.rnc.get_agent_pos_and_rotation(): ",
            #      self.rnc.get_agent_pos_and_rotation())
            #            cur_pos = self.rnc.get_agent_pos_and_rotation()
            #            self.initial_path_length = get_path_length(path, cur_pos)

            # at this point current path length is the initial path length. We will re-calculate current path length
            # many times and reward will be calculated using it.
            self.current_path_length = self.initial_path_length
            self.best_path_length = self.initial_path_length

            #print("CH2")
            path_planned = True

    # Get all reachable positions and store them in a variable.
    def update_navigation_artifacts(self, house):
        #print("U1")
        reachable_positions = [
            tuple(map(lambda x: round(roundany(x, self.grid_size), 2), pos))
            for pos in thor_reachable_positions(self.controller)]
        # print(reachable_positions, self.grid_size)

        # In this habitat we have these rooms
        rooms_in_habitat = get_rooms_ground_truth(house)
        # print(house["rooms"])
        # print("reachable_positions: ", reachable_positions)
        # AE: Path length infra set up
        # pos_ba = thor_reachable_positions(controller, by_axes = True)
        # print("AE, by axes: ", pos_ba)
        full_grid = create_full_grid_from_room_layout(rooms_in_habitat, step=self.grid_size)
        full_grid = [tuple(map(lambda x: round(x, 2), pos)) for pos in full_grid]
        unreachable_postions = set(full_grid) - set(reachable_positions)
        #(safe_pos, buf_unreachable_pos) = add_buffer_to_unreachable(set(reachable_positions), set(full_grid), step=self.grid_size)

        #print("U2")
        return reachable_positions, unreachable_postions, full_grid, rooms_in_habitat

    # This function will calculate path length to the desired point from the current position.
    # Also- what room type we're in
    def get_current_path_and_pose_state(self):
        #print("G1")
        try:
            cur_pos = self.rnc.get_agent_pos_and_rotation()
            self.current_path_length = self.nu.get_path_cost_to_target_point(cur_pos,
                                                                             self.current_target_point,
                                                                             self.reachable_positions,
                                                                             close_enough=self.plan_close_enough,
                                                                             step=self.grid_size)
            self.current_path_length = np.float32(self.current_path_length)
        except ValueError as e:
            #print(f"ERROR: {e}")
            #print("Using previous current_path_length: ", self.current_path_length)
            print('!', end='', sep='')
            self._bad_spot = True
            self._bad_spot_cnt += 1
            raise e # pass it on because reward calculation also needs to know

        # if we've been successful so far, then we can now look up room type
        cur_pos_xy = (cur_pos[0][0], "", cur_pos[0][2])
        room_type = self.find_room_type_of_this_point(cur_pos_xy)
        if room_type == None:
            print('i', end='', sep='')
            raise ValueError("Bad room picked, type can't be determined.")
        # and the actual room
        self.current_room = room_this_point_belongs_to(self.rooms_in_habitat, cur_pos_xy)
        #print("G2")
        return self.current_path_length, room_type

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
