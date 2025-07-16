import os, time
from enum import Enum
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import functools
import re
import zlib

#import deepmind_lab
import elements
import numpy as np

#import matplotlib.pyplot as plt

## AE: From my AI2-Thor simulation packages and for their purposes
from ai2_thor_model_training.training_data_extraction import (RobotNavigationControl, action_mapping,
                                                              action_to_index, index_to_action, inverted_action_mapping,
                                                              AI2THORUtils, get_path_length, get_centre_of_the_room,
                                                              room_this_point_belongs_to, get_rooms_ground_truth)

from thortils import launch_controller
from thortils.utils.math import sep_spatial_sample
import thortils as tt
import random, cv2
from thortils.agent import thor_agent_pose, thor_pose_as_tuple
from thortils.controller import _resolve
from thortils.navigation import get_shortest_path_to_object, get_navigation_actions, _round_pose, _same_pose, transform_pose, _valid_pose, _cost

from thortils.agent import thor_reachable_positions, thor_agent_position, thor_agent_pose
from thortils.utils import roundany, PriorityQueue, normalize_angles, euclidean_dist
from thortils.constants import MOVEMENT_PARAMS

from PIL import Image


class AI2ThorEnv():
    TOKENIZER = re.compile(r'([A-Za-z_]+|[^A-Za-z_ ]+)')

    def __init__(
            self, level, repeat=4, size=(64, 64), mode='train',
            actions='popart', episodic=True, text=None, seed=None):

        # AE: AI2-Thor simulation stuff
        self.rnc = RobotNavigationControl()
        self.controller = None
        self.atu = AI2THORUtils()
        self.rooms_in_habitat = None
        self.current_path_length = 1000
        self.reachable_positions = None
        self.grid_size = 0.0

        # when we select a random position and plan path to the room centre, we will assign a value to this parameter
        # with the A* path length from that random position to the desired point. This will help calculate reward from all
        # further points.
        self.initial_path_length = 0

        # We will need to keep track of the target point that we want to reach because we will be re-planning path to it
        # from all sorts of different points.
        self.current_target_point = None

        self.habitat_id = level

        # Dreamer stuff
        self._size = size
        self._repeat = repeat

        # Dreamer + AI2Thor
        self._actions = action_mapping
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
        if self._text:
            spaces['instr'] = elements.Space(
                np.float32, self._instr_length * self._embed_size)
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

    ##
    # Load the given habitat- load it, and put agent in a random place
    ##
    def load_habitat(self, habitat_id):
        # load required habitat
        habitat = self.atu.load_proctor_habitat(habitat_id)

        # Launch a controller for the loaded habitat. If we already have a controller,
        # then reset it instead of loading a new one.
        if (self.controller == None):
            self.controller = launch_controller({"scene": habitat,
                                                 "VISIBILITY_DISTANCE": 3.0,
                                                 "headless": False,
                                                 "IMAGE_WIDTH": 64,
                                                 "IMAGE_HEIGHT": 64
                                                 #"RENDER_DEPTH": False,
                                                 #"RENDER_INSTANCE_SEGMENTATION": False,
                                                 #"RENDER_IMAGE": True
                                                 #"IMAGE_WIDTH": 64,
                                                 #"IMAGE_HEIGHT": 64
                                                 })
            self.rnc.set_controller(
                self.controller)  # This allows our control scripts to interact with AI2-THOR environment
        else:
            self.controller.reset(habitat)
            #self.reset_state()
            self.rnc.reset_state()
            # self.rnc.set_controller(self.controller)

        # In this habitat we have these rooms
        self.rooms_in_habitat = get_rooms_ground_truth(habitat)

        # Take a snapshot of all available positions- these won't change while we're in this habitat,
        # so no need to re-do them everytime we plan a path.
        self.grid_size = self.controller.initialization_parameters["gridSize"]
        self.reachable_positions = self.update_reachable_positions()

        # Now place the robot in a random position and figure out the target from there.
        self.choose_random_placement_in_habitat()

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

        #reachable_positions = tt.thor_reachable_positions(self.controller)
        #self.reachable_positions
        placements = sep_spatial_sample(self.reachable_positions, sep, num_stops,
                                        rnd=rnd)

        #print(placements)

        # Choose one placement in the set of placements and then plan path from that placement to
        # the middle of the room. If planning path is not possible, then choose another one.
        path_planned = False
        while not path_planned:
            els = elements.Space(np.int32, (), 0, len(placements))
            p = list(placements)[int(els.sample())]
            #p = placements.pop()

            # append a rotation to the place.
            yaw = rnd.sample(h_angles, 1)[0]
            place_with_rtn = p + (yaw,)
            print("Placement: ", place_with_rtn)
            ## Teleport, then start new exploration. Achieve goal. Then repeat.
            self.rnc.teleport_to(place_with_rtn)

            # We've just been put in a random place in a habitat. We want to move now to where we want to go,
            # e.g., middle of the room, a door, etc.. For that we need to plan a path to there.
            point_for_room_search = (p[0], "", p[1])
            room_of_placement = room_this_point_belongs_to(self.rooms_in_habitat, point_for_room_search)
            room_centre = room_of_placement[2]

            # Now plan path to the centre of the room
#            try:
#                path_and_plan = self.get_path_to_target_point(room_centre)
#            except ValueError as e:
#                # If the path could not be planned, then drop it and carry on with the next one
#                print(f"ERROR: {e}")
#                continue

            self.current_target_point = room_centre
            try:
                self.initial_path_length = self.get_path_cost_to_target_point(self.current_target_point)
            except ValueError as e:
            # If the path could not be planned, then drop it and carry on with the next one
                print(f"ERROR: {e}")
                continue

            # print("PATH & PLAN: ", path_and_plan)
#            path = path_and_plan[0]
#            plan = path_and_plan[1]
            #self.prev_pose = thor_agent_pose(self.controller)  # This is where we are before the plan started
            # place_with_rtn
            # thor_pose_as_tuple(self.prev_pose)
            print("AE poses: place_with_rtn: ", place_with_rtn, " p: ", p, " self.rnc.get_agent_pos_and_rotation(): ", self.rnc.get_agent_pos_and_rotation())
#            cur_pos = self.rnc.get_agent_pos_and_rotation()
#            self.initial_path_length = get_path_length(path, cur_pos)

            # at this point current path length is the initial path length. We will re-calculate current path length
            # many times and reward will be calculated using it.
            self.current_path_length = self.initial_path_length

            path_planned = True

    # Get all reachable positions and store them in a variable.
    def update_reachable_positions(self):
        reachable_positions = [
            tuple(map(lambda x: roundany(x, self.grid_size), pos))
            for pos in thor_reachable_positions(self.controller)]
        #print(reachable_positions, self.grid_size)
        return reachable_positions

    class AStarNode:
        def __init__(self, pose, g=0, h=0, parent=None):
            self.x = pose[0][0]
            self.y = pose[0][2]
            self.yaw = pose[1][1]
            self.g = g  # Cost from start to current node
            self.h = h  # Heuristic cost estimate to goal
            self.f = g + h  # Total cost
            self.parent = parent

        def update_heuristic(self, new_h):
            self.h = new_h
            self.f = self.h + self.g

        def update_real_cost(self, new_g):
            self.g = new_g
            self.f = self.h + self.g

        def __lt__(self, other):
            return self.f < other.f

        def __eq__(self, other):
            return self.x == other.x and self.y == other.y

        def __hash__(self):
            return hash((self.x, self.y))

        def get_xy(self):
            return (self.x, self.y)

        def get_ai2thor_pose(self):
            return (self.x, 0.9009993672370911, self.y)

        def get_ai2thor_pose_and_rtn(self):
            return ((self.x, 0.9009993672370911, self.y), (0, self.yaw, 0))

    class NavigationAction(Enum):
        NORTH = (0, -0.25, 0)
        SOUTH = (0, 0.25, 180)
        EAST = (0.25, 0, 90)
        WEST = (-0.25, 0, 270)
        NORTHEAST = (0.25, -0.25, 45)
        NORTHWEST = (-0.25, -0.25, 315)
        SOUTHEAST = (0.25, 0.25, 135)
        SOUTHWEST = (-0.25, 0.25, 225)

        def __init__(self, dx, dy, yaw):
            self._dx = dx
            self._dy = dy
            self._yaw = yaw

        @property
        def dx(self):
            return self._dx

        @property
        def dy(self):
            return self._dy

        @property
        def new_yaw(self):
            return self._yaw

        # Apply the action to some X and Y coordinates.
        # This will allow us to test if the new X and Y is within reachable positions
        def apply(self, x, y):
            return x + self._dx, y + self._dy

        # Apply the action to a node. This will allow us to create a new node from a previous node
        # including X and Y coordinates and also the Yaw rotation.
        def apply_to_node(self, node, destination):
            full_pose = node.get_ai2thor_pose_and_rtn()
            # TODO: Are we updating numerical values here or the fields of the other node?
            #print("AE: full_pose[0][0]: ", full_pose[0][0], node.get_ai2thor_pose())
            new_x = full_pose[0][0] + self._dx
            new_y = full_pose[0][2] + self._dy
            #print("AE: full_pose[0][0]: ", full_pose[0][0], node.get_ai2thor_pose())

            # Cost will always be 1 for the movement ahead and some value to account for the required turns
            # And we want to calculate it before we update the yaw for the full pose which will form the new
            # node.
            turning_deg_required = abs(full_pose[1][1] - self._yaw)
            # If more than 180, then turn the other way
            if turning_deg_required > 180:
                turning_deg_required -= 180
            # We turn in 45 degree increments, so this many turns we will need
            turns_required = turning_deg_required // 45

            # Cost will always be 1 for the movement ahead and some value to account for the required turns
            new_cost = 1 + turns_required
            # now that new cost has been calculated, we can assign the new yaw to the pose fields.
            new_yaw = self._yaw
            new_full_pose = ((new_x, full_pose[0][1], new_y), (0.0, new_yaw, 0.0))
            new_node = AI2ThorEnv.AStarNode(new_full_pose, node.g + new_cost, euclidean_dist(new_full_pose[0], destination[0]), node)

            return new_node

    def normalize_to_grid(self, pose, step=0.25):
        def round_to_step(val):
            return round(val / step) * step

        def normalize_yaw(yaw):
            """
            Normalize yaw angle to the nearest multiple of 45 in [0, 315].
            Input yaw can be any real number (positive or negative).
            """
            yaw = yaw % 360  # Bring into [0, 360)
            return round(yaw / 45) * 45 % 360

        location = pose[0]
        rotation = pose[1]
        return ((round_to_step(location[0]), round_to_step(location[1]), round_to_step(location[2])),
                (0.0, normalize_yaw(rotation[1]), 0.0))

    def get_path_cost_to_target_point(self, target_point):
        # where we start
        start_point = self.rnc.get_agent_pos_and_rotation() # (start_position, start_rotation)
        # where we want to get to
        destination = ((target_point.x, 0.9009993672370911, target_point.y), start_point[1]) # the defined 2D coordinates and same rotation as start position
        # Normalize angles in start and goal to be within 0 to 360 (see top comments)
        # Also, round the poses so that we don't have irrational numbers in them that would be hard to look up
        # e.g. (10.75, 8.25) instead of (10.86666666, 8.3333333333).
        # Also angles need to be discrete values in [0, 45, 90, 135, 180, 225, 270, 315]
        start_point = _round_pose((start_point[0], normalize_angles(start_point[1])))
        destination = _round_pose((destination[0], normalize_angles(destination[1])))
        start_point = self.normalize_to_grid(start_point)
        destination = self.normalize_to_grid(destination)
        # positions that we can reach
        reachable_positions = set(self.reachable_positions)
        # The priority queue. We will keep poses in it with the estimates of their distances to the goal stored as priorities.
        worklist = PriorityQueue()
        # First pose will be the start and the estimate to the goal is its priority.
        # Make sure that all poses in the worklist and elsewhere are rounded though,
        # else they may not match later when we look them up and lead to no path found.
        start_node = self.AStarNode(start_point, h=euclidean_dist(start_point[0], destination[0]))
        # start_node.f doesn't take necessary turns into account, but as a heuristic it is admissible
        worklist.push(start_node, start_node.f)

        # cost[n] is the cost of the cheapest path from start to n currently known, where n is the pose
        cost = {}
        # Obviously from start to start the cost is 0
        cost[start_node.get_ai2thor_pose_and_rtn()] = 0
        # keep track of visited poses
        visited = set()

        # AE: Start the A* exploration. Obviously at first we will have the start node there with the estimate to the goal.
        while not worklist.isEmpty():
            current_node = worklist.pop()
            # If we've already visited this pose, then we can skip it and look at the next one
            if current_node in visited:
                continue
            # AE: If we're close enough to the end, then stop exploration and work backwards to reconstruct plan or
            # estimate path cost.
            if euclidean_dist(destination[0], current_node.get_ai2thor_pose()) <= 0.5:
                return cost[current_node.get_ai2thor_pose_and_rtn()]

            # AE: Look at all defined actions and try each of them from the current pose and see what happens
            for action in self.NavigationAction:
                nx, ny = action.apply(current_node.x, current_node.y)
                # If we end up in a legal place, then generate a new node and add it to the priority queue
                if (nx, ny) in reachable_positions:
                    # generate a new node from this action. There may be different nodes for the same location
                    # on the grid because they may have different yaw rotations and even different parents.
                    # So in the worst case there can be a node for <each grid location> * <all possible yaw rotations> * <each grid location as a parent>
                    next_node = action.apply_to_node(current_node, destination)
                    # The new nodes cost (the g value) has already been computed when it was generated, we can add it to the
                    # cost dictionary for this position and rotation if it's not already there.
                    #
                    # AE: Now we check if this pose, that we get with the chosen action, already exists in the cost set.
                    # If it does, then we'll get some number from cost.get(next_pose, float("inf"), otherwise we'll get
                    # infinity. If we got some number, but our new calculation is better than the old one, then we update
                    # the cost set with the new cost for the given pose.
                    if next_node.g < cost.get(next_node.get_ai2thor_pose_and_rtn(), float("inf")):
                        # AE: update the cost for this pose that we achieve from old pose with the selected action
                        cost[next_node.get_ai2thor_pose_and_rtn()] = next_node.g
                        #AE: push the newly discovered pose to our priority queue, giving the priority of its cost + euclidean
                        # distance from it to the goal as a heuristic (underestimate of the cost of the rest of the path).
                        worklist.push(next_node, next_node.f)
                        # AE: Keep track of where we came from so that we can reconstruct plan
                        # No need here, because each node contains a link to its parent
                        #comefrom[next_pose] = (current_pose, action)

            visited.add(current_node)

        # AE: If we're here, then that means, we could not find a path.
        # AE: print warning and return something.
        raise ValueError("Plan not found from {} to {}".format(start_point, destination))
        #return float("inf")

    # This function will calculate path length to the desired point from the current position.
    def get_current_path_length(self):
        try:
            self.current_path_length = self.get_path_cost_to_target_point(self.current_target_point)
        except ValueError as e:
            print(f"ERROR: {e}")
            print("Using previous current_path_length: ", self.current_path_length)

        return self.current_path_length

    # Calculates the reward for the current position of the agent. It is based on the length of the initial
    # path and the current path to the target.
    def current_reward(self):
        return (self.initial_path_length - self.get_current_path_length()) / self.initial_path_length

    # Compares the current reward with the maximum reward. If they're the same, then we have arrived.
    def have_we_arrived(self):
        return (self.current_path_length==0)

    # Advances the simulation using the selected action
    def step(self, action):
        # AE: If this is a terminal state or we need to end, then reset environment
        if action['reset'] or self._done:
            #self._env.reset(seed=self._random.randint(0, 2 ** 31 - 1))
            self.load_habitat(self.habitat_id)
            self._done = False
            return self._obs(0.0, is_first=True)
        #raw_action = np.array(self._actions[action['action']], np.intc)
        raw_action = index_to_action(int(action['action']))
        # print(raw_action)
        #reward = self._env.step(raw_action, num_steps=self._repeat)
        ets = time.time()
        self.rnc.execute_action(raw_action)
        #print("AE: Execute Action time: ", (time.time() - ets))
        rts = time.time()
        reward = self.current_reward()
        print("AE: Reward time: ", (time.time() - rts), " reward: ", reward)
        self._done = self.have_we_arrived()
        return self._obs(reward, is_last=self._done)

    # Returns the observations from the last performed step
    def _obs(self, reward, is_first=False, is_last=False):
        if not self._done:
            #self._current_image = self._env.observations()['RGB_INTERLEAVED']

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
        print(reward)
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
    dml = AI2ThorEnv(10)

    for i in range(10):
        act = {k: v.sample() for k, v in act_space.items()}
        print(act)
        a = index_to_action(int(act['action']))
        print(a)
        act['reset'] = False
        observation = dml.step(act)

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
