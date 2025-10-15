import numpy as np

file = "20251015T154541F148200-6XFpjmGAMTBQ5aRDIGecgb-5y5Jr3pScPJmyWqxQQs6Sd-76.npz"
with np.load(file) as data:
    print(data.files)

    ndx_start = 55
    ndx_stop = 61

    terminals = data['is_terminal']
    firsts = data['is_first']
    rewards = data['reward']
    lasts = data['is_last']
    rewards = data['reward']
    distance_lefts = data['distance_left']
    steps_after_room_changes = data['steps_after_room_change']
    room_types = data['room_type']
    actions = data['action']

    print("firsts: ", firsts[ndx_start:ndx_stop])
    print("terminals: ", terminals[ndx_start:ndx_stop])
    print("lasts: ", lasts[ndx_start:ndx_stop])
    print("rewards: ", rewards[ndx_start:ndx_stop])
    print("distance_lefts: ", distance_lefts[ndx_start:ndx_stop])
    print("steps_after_room_changes: ", steps_after_room_changes[ndx_start:ndx_stop])
    print("room_types: ", room_types[ndx_start:ndx_stop])
    print("actions: ", actions[ndx_start:ndx_stop])

    print("Start:")
    ndx_start = 0
    ndx_stop = 2
    print("firsts: ", firsts[ndx_start:ndx_stop])
    print("terminals: ", terminals[ndx_start:ndx_stop])
    print("lasts: ", lasts[ndx_start:ndx_stop])
    print("rewards: ", rewards[ndx_start:ndx_stop])
    print("distance_lefts: ", distance_lefts[ndx_start:ndx_stop])
    print("steps_after_room_changes: ", steps_after_room_changes[ndx_start:ndx_stop])
    print("room_types: ", room_types[ndx_start:ndx_stop])
    print("actions: ", actions[ndx_start:ndx_stop])