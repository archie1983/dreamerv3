#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from enum import Enum
import functools, re, zlib, time

#import sys
#print('\n'.join(sys.path))

#import deepmind_lab
import elements
import numpy as np

#import matplotlib.pyplot as plt

## AE: From my AI2-Thor simulation packages and for their purposes
from ai2_thor_model_training.training_data_extraction import RobotNavigationControl
from ai2_thor_model_training.ae_utils import (NavigationUtils, action_mapping,
                                              action_to_index, index_to_action, inverted_action_mapping,
                                              AI2THORUtils, get_path_length, get_centre_of_the_room,
                                              room_this_point_belongs_to, get_rooms_ground_truth,
                                              get_all_objects_of_type, is_point_inside_room_ground_truth,
                                              create_full_grid_from_room_layout, add_buffer_to_unreachable)

from thortils import launch_controller
from thortils.utils.math import sep_spatial_sample
import thortils as tt
import random, cv2
from thortils.agent import thor_agent_pose, thor_pose_as_tuple
from thortils.controller import _resolve
from thortils.navigation import get_shortest_path_to_object, get_navigation_actions, _round_pose, _same_pose, transform_pose, _valid_pose, _cost

from thortils.agent import thor_reachable_positions, thor_agent_position, thor_agent_pose
from thortils.utils import roundany, PriorityQueue, normalize_angles, euclidean_dist, getch
from thortils.constants import MOVEMENT_PARAMS
from shapely.geometry import Point
import embodied

#from PIL import Image

class AI2ThorEnv(embodied.Env):
    TOKENIZER = re.compile(r'([A-Za-z_]+|[^A-Za-z_ ]+)')

    def __init__(
            self, level, repeat=4, size=(64, 64), mode='train',
            episodic=True, seed=None):

        # AE: AI2-Thor simulation stuff
        self.rnc = RobotNavigationControl()
        self.controller = None
        self.atu = AI2THORUtils()
        self.rooms_in_habitat = None
        self.current_path_length = 1000
        self.reachable_positions = None
        self.grid_size = 0.125 # how fine do we want the 2D grid to be.
        self.reward_close_enough = 0.125 # how close to the target is close enough for the purposes of reward.
        self.plan_close_enough = 0.25  # how close to the target is close enough for the purposes of path planning.
        self.nu = NavigationUtils(step=self.grid_size)
        self.step_time = 0.0
        # If we get into a bad spot from which for whatever reason we can't plan a path out, then we'll set this to
        # True and based on it will teleport to a new place when we see this set.
        self._bad_spot = False
        self._bad_spot_cnt = 0
        self._total_reward_for_this_run = 0

        # when we select a random position and plan path to the room centre, we will assign a value to this parameter
        # with the A* path length from that random position to the desired point. This will help calculate reward from all
        # further points.
        self.initial_path_length = 0

        # If we use reward that tracks the best length of the path left, then we will need this variable
        self.best_path_length = 0

        # We will need to keep track of the target point that we want to reach because we will be re-planning path to it
        # from all sorts of different points.
        self.current_target_point = None

        self.habitat_id = level # redundant - we load random habitats and not some level

        # Selection of what we want to target - room centres or doors
        self.doors_or_centre = True

        # Dreamer stuff
        self._size = size
        self._repeat = repeat

        # Dreamer + AI2Thor
        self._actions = action_mapping
        print("AE:", self._actions)
        if "STOP" in self._actions:
            self._actions.pop("STOP")  # remove STOP action because that will be treated differently

        self._episodic = episodic
        self._random = np.random.RandomState(seed)
        config = dict(height=size[0], width=size[1], logLevel='WARN')
        if mode == 'train':
            # if level.endswith('_test'):
            #  level = level.replace('_test', '_train')
            pass
        elif mode == 'eval':
            # config.update(allowHoldOutLevels='true', mixerSeed=0x600D5EED)
            pass
        else:
            raise NotImplementedError(mode)
        config = {k: str(v) for k, v in config.items()}

        # self._env = deepmind_lab.Lab(
        #    level='contributed/dmlab30/' + level,
        #    observations=obs, config=config)
        self._current_image = None
        self._done = True

    @property
    def obs_space(self):
        spaces = {
            'image': elements.Space(np.uint8, self._size + (3,)),
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
        }
#        if self._text:
#            spaces['instr'] = elements.Space(
#                np.float32, self._instr_length * self._embed_size)
        return spaces

    @property
    def act_space(self):
        return {
            'action': elements.Space(np.int32, (), 0, len(self._actions)),
            'reset': elements.Space(bool),
        }

    def process_required_habitats(self):
        self.process_habitat(10)
        self.controller.stop()

    def load_random_habitat(self):
        # choose a random habitat from a space of train_10 till train_200
        loaded = False
        while not loaded:
            try:
                sp = elements.Space(np.int32, (), 100, 400)
                self.habitat_id = sp.sample()
                self.load_habitat(self.habitat_id)
                # enfore at least 2 rooms in a habitat
                if len(self.rooms_in_habitat) >= 2:
                    loaded = True
            except ValueError as e:
                continue

    ##
    # Load the given habitat- load it, and put agent in a random place
    ##
    def load_habitat(self, habitat_id):
        # load required habitat
        #print("AE: haba: ", habitat_id)
        self.habitat = self.atu.load_proctor_habitat(int(habitat_id))

        # Launch a controller for the loaded habitat. If we already have a controller,
        # then reset it instead of loading a new one.
        if (self.controller == None):
            self.controller = launch_controller({"scene": self.habitat,
                                                 "VISIBILITY_DISTANCE": 3.0,
                                                 "headless": False,
                                                 "IMAGE_WIDTH": 64,
                                                 "IMAGE_HEIGHT": 64,
                                                 "GRID_SIZE": self.grid_size
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
            els = elements.Space(np.int32, (), 0, len(placements))
            p = list(placements)[int(els.sample())]
            # p = placements.pop()

            # append a rotation to the place.
            yaw = rnd.sample(h_angles, 1)[0]
            place_with_rtn = p + (yaw,)
            #print("Placement: ", place_with_rtn)
            ## Teleport, then start new exploration. Achieve goal. Then repeat.
            self.rnc.teleport_to(place_with_rtn)

            # We've just been put in a random place in a habitat. We want to move now to where we want to go,
            # e.g., middle of the room, a door, etc.. For that we need to plan a path to there.
            try:
                if (self.doors_or_centre):
                    self.current_target_point = self.choose_door_target(place_with_rtn)
                    # print("self.current_target_point: ", self.current_target_point)
                else:
                    point_for_room_search = (p[0], "", p[1])
                    # print("point_for_room_search: ", point_for_room_search)
                    self.current_target_point = self.find_room_centre_target(point_for_room_search)

                cur_pos = self.rnc.get_agent_pos_and_rotation()
                self.initial_path_length = self.nu.get_path_cost_to_target_point(cur_pos,
                                                                                 self.current_target_point,
                                                                                 self.reachable_positions,
                                                                                 close_enough=self.plan_close_enough,
                                                                                 step=self.grid_size)
            except ValueError as e:
                # If the path could not be planned, then drop it and carry on with the next one
                #print(f"ERROR: {e}")
                if placement_attempts <= 10:
                    print(".", sep="", end="")
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

            path_planned = True

    ##
    # Finds the centre of the current room given the current position and the rooms in habitat.
    ##
    def find_room_centre_target(self, point_for_room_search):
        # We've just been put in a random place in a habitat. We want to move now to where we want to go,
        # e.g., middle of the room, a door, etc.. For that we need to plan a path to there.
        room_of_placement = room_this_point_belongs_to(self.rooms_in_habitat, point_for_room_search)
        room_centre = room_of_placement[2]
        return room_centre

    ##
    # Go through all the doors and find the most appropriate as a target, then add a little bit extra so that
    # we end up going through the door.
    # place_with_rtn: Place with rotation, e.g.: (6.62, 6.25, 180)
    ##
    def choose_door_target(self, place_with_rtn):
        current_target_point = None
        try:
            current_target_point = self.nu.find_door_target(place_with_rtn,
                                                       self.rooms_in_habitat,
                                                       self.reachable_positions,
                                                       self.habitat,
                                                       self.controller, close_enough=self.plan_close_enough, step=self.grid_size, extend_path=True)

            #t1 = time.time()
            pose = ((place_with_rtn[0], 0.0, place_with_rtn[1]), (0.0, place_with_rtn[2], 0.0)) # place_with_rtn in AI2-Thor format
            path_length = self.nu.get_path_cost_to_target_point(pose,
                                                                current_target_point,
                                                                self.reachable_positions, close_enough=self.plan_close_enough, step=self.grid_size)
            #print("AE: path plan time: ", (time.time() - t1))
        except ValueError as e:
            path_length = 0
            #print("AE: No Path Found", e)
            print('£', end='', sep='')

        if current_target_point == None:
            raise ValueError("No suitable doors were found")

        #print("AE: Path Length: ", path_length)
        (cur_path, reachable_positions, start, dest) = self.nu.get_last_path_and_params()
        #print("AE: Path: ", cur_path)
        #atu.visualise_path2(cur_path, reachable_positions, unreachable_postions, rooms_in_habitat, start, dest,
        #                    show_unreachable_pos=False,
        #                    show_reachable_pos=False)
        # atu.visualise_path2(cur_path, reachable_positions, buf_unreachable_pos, rooms_in_habitat, start, dest, show_unreachable_pos=True)
        return current_target_point

    # Get all reachable positions and store them in a variable.
    def update_navigation_artifacts(self, house):
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

        return reachable_positions, unreachable_postions, full_grid, rooms_in_habitat

    # This function will calculate path length to the desired point from the current position.
    def get_current_path_length(self):
        try:
            cur_pos = self.rnc.get_agent_pos_and_rotation()
            self.current_path_length = self.nu.get_path_cost_to_target_point(cur_pos,
                                                                             self.current_target_point,
                                                                             self.reachable_positions,
                                                                             close_enough=self.plan_close_enough,
                                                                             step=self.grid_size)

        except ValueError as e:
            #print(f"ERROR: {e}")
            #print("Using previous current_path_length: ", self.current_path_length)
            print('!', end='', sep='')
            self._bad_spot = True
            self._bad_spot_cnt += 1

        return self.current_path_length

    # Calculates the reward for the current position of the agent. It is based on the length of the initial
    # path and the current path to the target.
    def current_reward(self):
        return (self.initial_path_length - self.get_current_path_length()) / self.initial_path_length

    ##
    # AE: A sparse reward- giving reward only when we reach the target and only once. In all other cases give penalty of -1
    ##
    def sparse_reward(self):
        self.get_current_path_length()
        #return 1000 if self.have_we_arrived(0.25) else -1
        return 20 if self.have_we_arrived(self.reward_close_enough) else -1

    ##
    # AE: A reward that penalises going back or staying in place and rewards progress.
    ##
    def path_progress_reward(self):
        cur_path_length = self.get_current_path_length()

        if self.have_we_arrived(self.reward_close_enough):
            # If we're there, then give 20
            return 20
        elif self.best_path_length > cur_path_length:
            # If we improved the path, then give 1
            self.best_path_length = cur_path_length
            return 1
        elif self.best_path_length == cur_path_length:
            # if we stayed in place, then small penalty
            return -0.1
        else:
            # If we went backwards then penalise
            return -0.3

    # Compares the current reward with the maximum reward. If they're the same, then we have arrived.
    def have_we_arrived(self, epsilon = 0.0):
        return (self.current_path_length <= epsilon)

    # Advances the simulation using the selected action
    def step(self, action):
        #print("STEP T diff: ", (time.time() - self.step_time))
        #self.step_time = time.time()
        # AE: If this is a terminal state or we need to end, then reset environment
        if action['reset'] or self._done:
            # self._env.reset(seed=self._random.randint(0, 2 ** 31 - 1))
            # load a new random habitat
            self.load_random_habitat()
            self._done = False
            self._bad_spot = False
            return self._obs(0.0, is_first=True)
        elif self._bad_spot:
            # ignore bad spot. Our path planning can now recover. For plan not found it will just get usual penalty
            # and carry on.
            #self.choose_random_placement_in_habitat()

            # No, do not ignore, we can still get a series of bad spots and hang the process (e.g. habitat 46 iirc
            # supposedly has like 4 rooms, but only 2 can be accessed or something similar.
            # Instead if choosing a good random placement has failed 10 times, then catch the error and load
            # a different habitat.
            try:
                self.choose_random_placement_in_habitat()
            except ValueError as e:
                self.load_random_habitat()

            self._bad_spot = False
        # raw_action = np.array(self._actions[action['action']], np.intc)
        raw_action = index_to_action(int(action['action']))
        # print(raw_action)
        # reward = self._env.step(raw_action, num_steps=self._repeat)
        #ets = time.time()
        self.rnc.execute_action(raw_action, grid_size=self.grid_size, adhere_to_grid=True)
        #print("AE: Execute Action time: ", (time.time() - ets))
        #rts = time.time()
        #reward = self.current_reward()
        #reward = self.sparse_reward()
        reward = self.path_progress_reward()
        self._total_reward_for_this_run += reward

        #print("AE: Reward time: ", (time.time() - rts), " reward: ", reward)
        self._done = self.have_we_arrived(self.reward_close_enough)

        return self._obs(reward, is_last=self._done)

    # Returns the observations from the last performed step
    def _obs(self, reward, is_first=False, is_last=False):
        #rts = time.time()
        if not self._done:
            # self._current_image = self._env.observations()['RGB_INTERLEAVED']

            event = self.controller.last_event
            #            self._current_image = event.cv2img

            #            event = self.controller.step(action="Pass", renderImage=True)  # Forces a render update
            self._current_image = event.cv2img

        #            event = self.controller.last_event

        #            print(event.metadata["screenWidth"], event.metadata["screenHeight"])

        #            if event.frame is not None:
        #                self._current_image = event.frame[..., ::-1]  # BGR to RGB if needed
        #            else:
        #                raise RuntimeError("Frame not rendered in headless mode!")

        obs = dict(
            image=self._current_image,
            reward=np.float32(reward),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_last if self._episodic else False,
        )
        #print(reward)
        #print("AE: OBS time: ", (time.time() - rts), " reward: ", reward)
        return obs

    def _embed(self, text):
        tokens = self.TOKENIZER.findall(text.lower())
        indices = [self._hash(token) for token in tokens]
        # print('EMBED', text, '->', tokens, '->', indices)
        indices = indices + [0] * (self._instr_length - len(indices))
        embeddings = [self._embeddings[i] for i in indices]
        return np.concatenate(embeddings)

    @functools.cache
    def _hash(self, token):
        return zlib.crc32(token.encode('utf-8')) % self._vocab_buckets

    def close(self):
        if (self.controller != None):
            self.controller.stop()


if __name__ == "__main__":
    # AE: A word about actions:
    ## AE: There is no documentation or useful comments in this code or elements package, so I have to guess,
    # but it looks like this is a uniform space generator within the specified limits., so for example
    # if we have els = elements.Space(np.int32, (), 0, 3), then that should generate an array of [0, 1, 2].
    # We can then call els.sample() and it will return a random element from that uniform space (array).
    # els = elements.Space(np.int32, (), 0, 7)
    # print("els.sample(): ", els.sample())

    # AE: This creates an action space- a dictionary of two normal distributions- one labelled "action" and
    # the other: "reset".
    # act_space = {
    #    'action': els,
    #    'reset': elements.Space(bool),
    # }
    # Now we go through both normal distributions and choose a random member in either of them and put that under
    # the dictionary keyword. I.e., we are choosing a random action and a random reset flag, although since the
    # reset flag is chosen from only 1 member (which is True in this case), then it is not really random, but always
    # True.
    # act = {k: v.sample() for k, v in act_space.items()}

    # print("act", act)
    # print("act['action'], act['reset']", act['action'], act['reset'])

    # bels = elements.Space(bool)
    # print("bels: ", bels, " bels.sample(): ", bels.sample())

    els = elements.Space(np.int32, (), 0, 3)
    act_space = {
        'action': els,
        'reset': elements.Space(bool),
    }
    dml = AI2ThorEnv(43)
    dml.load_habitat(43) # load habitat and choose first random placement

    # for i in range(10):
    #     act = {k: v.sample() for k, v in act_space.items()}
    #     print(act)
    #     a = index_to_action(int(act['action']))
    #     print(a)
    #     act['reset'] = False
    #     observation = dml.step(act)

    dml.choose_random_placement_in_habitat() # choose another placement.

    dml.close()

    # dml = AI2ThorEnv("rooms_watermaze")
    # print(dml.act_space)
    # for i in range(10): # do 10 actions
    #     # select a random action from the action space
    #     act = {k: v.sample() for k, v in dml.act_space.items()}
    #     # Make sure this is not a terminal state
    #     act['reset'] = False
    #     # Feed the selected random action to the environment
    #     observation = dml.step(act)
    #     #print(observation)
    #     plt.imshow(observation['image'])
    #     plt.pause(0.001)
    #     plt.draw()
    #
    # # and then quit
    # act = {k: v.sample() for k, v in dml.act_space.items()}
    # act['reset'] = True
    # observation = dml.step(act)
    # #print(observation)
    #
    # plt.imshow(observation['image'])
    # plt.pause(0.001)
    # plt.draw()
