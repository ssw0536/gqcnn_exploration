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
from gqcnn.grasping.policy import GraspAction

from src.grasping.grasp_planner import GraspPlanner, RuleBasedSelectGraspAction
from src.grasping.grasp_metric import evaluate_pj_grasp, evaluate_sc_grasp
from src.simulation.handler import IsaacGymHandler
from src.simulation.handler import ObjectData, ObjectActorSet
from src.simulation.handler import BinActor
from src.simulation.handler import Camera
from src.visualizer import (depth_image_to_gray_image,
                            label_grasp_on_image,
                            visulaize_scene)


class MeshGraspData(object):
    def __init__(self, mesh_name, mesh):
        self.mesh_name = mesh_name
        self.mesh = mesh
        self.grasp_poses = []
        self.actions = []
        self.successes = []
        self.priors = []
        self.posteriors = []


if __name__ == "__main__":
    # set save dir
    # run_history_dir = os.path.join(
    #     'run_history',
    #     '2023-05-04-18-09-51')
    run_history_dir = os.path.join('/media/sungwon/WorkSpace/projects/isaacgym-gqcnn-results/2023-05-15-02-46-54')

    # load config
    run_cfg = yaml.load(open(os.path.join(run_history_dir, 'run_config.yaml'), 'r'), Loader=yaml.FullLoader)
    sim_cfg = run_cfg['simulation']
    cam_cfg = run_cfg['camera']['phoxi']
    exp_cfg = run_cfg['experiment']
    visualize = False
    verbose = False
    pj_policy_config_file = os.path.join(run_history_dir, 'pj_policy_config.yaml')

    # load meshes
    object_files = np.load(os.path.join(run_history_dir, 'object_files.npy'))
    meshes = []
    obejct_names = []

    for file in object_files:
        object_name = file.split('/')[-1].split('.')[0]
        mesh_file = file + '/' + object_name + '.obj'
        mesh = trimesh.load(mesh_file)
        meshes.append(mesh)
        obejct_names.append(object_name)

    # load grasp planner
    pj_grasp_planner = GraspPlanner(pj_policy_config_file)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(logging.WARNING)

    # Load camera parameters
    camera_intrinsic = np.load(os.path.join(run_history_dir, 'camera_intrinsic.npy'))
    camera_extrinsic = np.load(os.path.join(run_history_dir, 'camera_extrinsic.npy'))

    # create mesh grasp dataset
    pj_mesh_grasp_dataset = []
    for mesh_name, mesh in zip(obejct_names, meshes):
        pj_mesh_grasp_dataset.append(MeshGraspData(mesh_name, mesh))

    # extract target object idx
    grasp_poses_per_mesh = []
    episode_dirs = glob.glob(os.path.join(run_history_dir, 'episode_*'))
    episode_dirs.sort()
    for episode, episode_dir in enumerate(episode_dirs):
        # load action and reward
        for i in range(100):
            if not os.path.exists(os.path.join(episode_dir, 'action_{:03d}.pkl'.format(i))):
                break
            action = pickle.load(open(os.path.join(episode_dir, 'action_{:03d}.pkl'.format(i)), 'rb'))
            reward = pickle.load(open(os.path.join(episode_dir, 'reward_{:03d}.pkl'.format(i)), 'rb'))
            object_poses = pickle.load(open(os.path.join(episode_dir, 'object_poses_{:03d}.pkl'.format(i)), 'rb'))
            target_object_idx = pickle.load(open(os.path.join(episode_dir, 'target_object_idx_{:03d}.pkl'.format(i)), 'rb'))

            if isinstance(action.grasp, Grasp2D):
                # get grasp pose in object frame
                grasp_pose_w = pj_grasp_planner.get_grasp_pose_world(
                    action.grasp, camera_extrinsic)
                grasp_pose_o = np.linalg.inv(object_poses[target_object_idx]) @ grasp_pose_w

                # append data
                pj_mesh_grasp_dataset[target_object_idx].actions.append(action)
                pj_mesh_grasp_dataset[target_object_idx].successes.append(reward)
                pj_mesh_grasp_dataset[target_object_idx].grasp_poses.append(grasp_pose_o)

    # get prior prediction
    for mesh_grasp_data in pj_mesh_grasp_dataset:
        assert isinstance(mesh_grasp_data, MeshGraspData)
        if len(mesh_grasp_data.actions) == 0:
            print('No actions on {}'.format(mesh_grasp_data.mesh_name))
            continue
        pj_priors = pj_grasp_planner.policy.evaluate_action(mesh_grasp_data.actions)
        mesh_grasp_data.priors = pj_priors

    # update policy
    for mesh_grasp_data in pj_mesh_grasp_dataset:
        assert isinstance(mesh_grasp_data, MeshGraspData)
        if len(mesh_grasp_data.actions) == 0:
            print('No actions on {}'.format(mesh_grasp_data.mesh_name))
            continue
        for action, reward in zip(mesh_grasp_data.actions, mesh_grasp_data.successes):
            pj_grasp_planner.policy.update(action, reward)

    # get posterior prediction
    for mesh_grasp_data in pj_mesh_grasp_dataset:
        assert isinstance(mesh_grasp_data, MeshGraspData)
        if len(mesh_grasp_data.actions) == 0:
            print('No actions on {}'.format(mesh_grasp_data.mesh_name))
            continue
        pj_posteriors = pj_grasp_planner.policy.evaluate_action(mesh_grasp_data.actions)
        mesh_grasp_data.posteriors = pj_posteriors

    # create save dir based on mesh name
    pj_save_dir = os.path.join(run_history_dir, 'grasps_on_mesh', 'pj')
    sc_save_dir = os.path.join(run_history_dir, 'grasps_on_mesh', 'sc')
    for mesh_name in obejct_names:
        os.makedirs(os.path.join(pj_save_dir, mesh_name), exist_ok=True)
        pj_mesh_grasp_data = pj_mesh_grasp_dataset[obejct_names.index(mesh_name)]
        pickle.dump(pj_mesh_grasp_data.grasp_poses, open(os.path.join(pj_save_dir, mesh_name, 'grasp_poses.pkl'), 'wb'))
        pickle.dump(pj_mesh_grasp_data.successes, open(os.path.join(pj_save_dir, mesh_name, 'successes.pkl'), 'wb'))
        pickle.dump(pj_mesh_grasp_data.priors, open(os.path.join(pj_save_dir, mesh_name, 'priors.pkl'), 'wb'))
        pickle.dump(pj_mesh_grasp_data.posteriors, open(os.path.join(pj_save_dir, mesh_name, 'posteriors.pkl'), 'wb'))
