import os
import glob
import csv
import datetime
import shutil
import random
import logging
import yaml

import tqdm
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import pickle
from isaacgym import gymapi

from gqcnn.grasping import Grasp2D, SuctionPoint2D

from src.grasping.grasp_planner import GraspPlanner, RuleBasedSelectGraspAction
from src.grasping.grasp_metric import evaluate_pj_grasp, evaluate_sc_grasp
from src.simulation.handler import IsaacGymHandler
from src.simulation.handler import ObjectData, ObjectActorSet
from src.simulation.handler import BinActor
from src.simulation.handler import Camera
from src.visualizer import (depth_image_to_gray_image,
                            label_grasp_on_image,
                            visulaize_scene)


if __name__ == "__main__":
    # load config
    run_cfg = yaml.load(open('cfg/run/config.yaml', 'r'), Loader=yaml.FullLoader)
    sim_cfg = run_cfg['simulation']
    cam_cfg = run_cfg['camera']['phoxi']
    exp_cfg = run_cfg['experiment']
    visualize = False
    verbose = False

    # create gym and simulation
    handler = IsaacGymHandler(sim_cfg)
    gym = handler.gym
    sim = handler.sim
    dt = handler.dt
    physics_engine = handler.physics_engine

    # get target object list
    if False:
        object_files = glob.glob('assets/urdf/egad_eval_set_urdf/*')
        random.shuffle(object_files)
        object_files = object_files[:25]
        object_list = [ObjectData(gym, sim, p, physics_engine) for p in tqdm.tqdm(object_files)]
        np.save(os.path.join('cfg', 'object_files.npy'), object_files)
    else:
        object_files = np.load(os.path.join('cfg', 'object_files.npy'))
        object_list = [ObjectData(gym, sim, p, physics_engine) for p in tqdm.tqdm(object_files)]

    # create envs
    env_idx = 0
    env_lower = gymapi.Vec3(-0.5, -0.5, 0.0)
    env_upper = gymapi.Vec3(0.5, 0.5, 2.0)
    env = gym.create_env(sim, env_lower, env_upper, 1)

    # add bin actor vbn
    bin_actor = BinActor(gym, sim, env, (0.2, 0.5), env_idx)

    # create object_actor_set and load object meshes
    meshes = []
    for file in object_files:
        object_name = file.split('/')[-1].split('.')[0]
        mesh_file = file + '/' + object_name + '.obj'
        mesh = trimesh.load(mesh_file)
        meshes.append(mesh)
    object_actor_set = ObjectActorSet(gym, sim, env, object_list, env_idx)
    object_actor_set.meshes = meshes

    # execute policy
    num_episode = exp_cfg['num_episode']
    max_trial = exp_cfg['max_trial']
    for episode in range(num_episode):
        # set bin box and dorp objects
        bin_actor.set()
        object_actor_set.drop(len(object_actor_set))
        handler.wait(1.0)
        bin_actor.unset()
        handler.wait(1.0)

        # use tqdm
        indices = np.arange(len(object_actor_set))
        np.random.shuffle(indices)
        for i in indices:
            object_actor_set.remove(i)
            handler.wait(1.0)
