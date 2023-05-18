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

    # load save dir
    cur_time = '2023-05-04-18-09-51'
    save_dir = os.path.join('run_history', cur_time)
    # pj_policy_config_file = 'cfg/policy/beta_process_pj.yaml'
    # sc_policy_config_file = 'cfg/policy/beta_process_sc.yaml'
    pj_policy_config_file = os.path.join(save_dir, 'pj_policy_config.yaml')
    sc_policy_config_file = os.path.join(save_dir, 'sc_policy_config.yaml')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'fig', 'pj'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'fig', 'sc'), exist_ok=True)

    # laod meshes
    object_files = np.load(os.path.join('cfg', 'object_files.npy'))
    meshes = []
    for file in object_files:
        object_name = file.split('/')[-1].split('.')[0]
        mesh_file = file + '/' + object_name + '.obj'
        mesh = trimesh.load(mesh_file)
        meshes.append(mesh)

    # load grasp planner
    pj_grasp_planner = GraspPlanner(pj_policy_config_file)
    sc_grasp_planner = GraspPlanner(sc_policy_config_file)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(logging.WARNING)

    # execute policy
    episode_dirs = glob.glob(os.path.join(save_dir, 'episode_*'))
    episode_dirs.sort()
    for episode, episode_dir in enumerate(episode_dirs):
        # get action and reward files
        action_files = glob.glob(os.path.join(episode_dir, 'action_*.pkl'))
        reward_files = glob.glob(os.path.join(episode_dir, 'reward_*.pkl'))
        action_files.sort()
        reward_files.sort()

        # load action and reward
        for aciton_file, reward_file in zip(action_files, reward_files):
            action = pickle.load(open(aciton_file, 'rb'))
            reward = pickle.load(open(reward_file, 'rb'))

            if isinstance(action.grasp, Grasp2D):
                pj_grasp_planner.update_policy(action, reward)
            elif isinstance(action.grasp, SuctionPoint2D):
                sc_grasp_planner.update_policy(action, reward)

        # get feature map
        if episode == 0:
            pj_fig_file = os.path.join(
                save_dir, 'fig', 'pj', 'episode_{:04d}.png'.format(episode))
            pj_grasp_planner.policy.visualize_feature_map(
                show_features=True, save_path=pj_fig_file)
            sc_fig_file = os.path.join(
                save_dir, 'fig', 'sc', 'episode_{:04d}.png'.format(episode))
            sc_grasp_planner.policy.visualize_feature_map(
                show_features=True, save_path=sc_fig_file)
            plt.close('all')
            print('episode: {}'.format(episode))
            exit()
        elif episode == 99:
            pj_fig_file = os.path.join(
                save_dir, 'fig', 'pj', 'episode_{:04d}.png'.format(episode))
            pj_grasp_planner.policy.visualize_feature_map(
                show_features=True, save_path=pj_fig_file)
            sc_fig_file = os.path.join(
                save_dir, 'fig', 'sc', 'episode_{:04d}.png'.format(episode))
            sc_grasp_planner.policy.visualize_feature_map(
                show_features=True, save_path=sc_fig_file)
            plt.close('all')
            print('episode: {}'.format(episode))
        # visualize feature map
        # print(episode)
        # if episode == 99:
        #     pj_fig_file = os.path.join(
        #         save_dir, 'fig', 'pj', 'episode_{:04d}.png'.format(episode))
        #     pj_grasp_planner.policy.visualize_feature_map(
        #         show_features=True, save_path=None)
        #     plt.show()
        #     sc_fig_file = os.path.join(
        #         save_dir, 'fig', 'sc', 'episode_{:04d}.png'.format(episode))
        #     sc_grasp_planner.policy.visualize_feature_map(
        #         show_features=True, save_path=None)
        #     plt.show()
