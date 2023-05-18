import os
import time
import glob
import logging
import yaml

import numpy as np
import trimesh
import pickle

from gqcnn.grasping import Grasp2D, SuctionPoint2D

from src.grasping.grasp_planner import GraspPlanner
from src.grasping.grasp_metric import evaluate_pj_grasp, evaluate_sc_grasp


if __name__ == "__main__":
    # set save dir
    run_history_dir = os.path.join(
        'run_history',
        '2023-05-04-18-09-51')

    # load config
    run_cfg = yaml.load(open(os.path.join(run_history_dir, 'run_config.yaml'), 'r'), Loader=yaml.FullLoader)
    sim_cfg = run_cfg['simulation']
    cam_cfg = run_cfg['camera']['phoxi']
    exp_cfg = run_cfg['experiment']
    visualize = False
    verbose = False
    pj_policy_config_file = os.path.join(run_history_dir, 'pj_policy_config.yaml')
    sc_policy_config_file = os.path.join(run_history_dir, 'sc_policy_config.yaml')

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
    sc_grasp_planner = GraspPlanner(sc_policy_config_file)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(logging.WARNING)

    # Load camera parameters
    camera_intrinsic = np.load(os.path.join(run_history_dir, 'camera_intrinsic.npy'))
    camera_extrinsic = np.load(os.path.join(run_history_dir, 'camera_extrinsic.npy'))

    # create save dir based on mesh name
    save_dir = os.path.join(run_history_dir, 'grasps_on_mesh')
    os.makedirs(save_dir, exist_ok=True)
    for mesh_name in obejct_names:
        os.makedirs(os.path.join(save_dir, mesh_name), exist_ok=True)

    # extract target object idx
    episode_dirs = glob.glob(os.path.join(run_history_dir, 'episode_*'))
    episode_dirs.sort()
    for episode, episode_dir in enumerate(episode_dirs):
        # get action and reward files
        action_files = glob.glob(os.path.join(episode_dir, 'action_*.pkl'))
        reward_files = glob.glob(os.path.join(episode_dir, 'reward_*.pkl'))
        object_pose_files = glob.glob(os.path.join(episode_dir, 'object_poses_*.pkl'))
        action_files.sort()
        reward_files.sort()
        object_pose_files.sort()

        # load action and reward
        actions = []
        rewards = []
        object_poses = []
        for aciton_file, reward_file, object_pose_file in zip(action_files, reward_files, object_pose_files):
            actions.append(pickle.load(open(aciton_file, 'rb')))
            rewards.append(pickle.load(open(reward_file, 'rb')))
            object_poses.append(pickle.load(open(object_pose_file, 'rb')))

        for i, (a, p) in enumerate(zip(actions, object_poses)):
            start_time = time.time()
            if isinstance(a.grasp, Grasp2D):
                grasp_pose_w = pj_grasp_planner.get_grasp_pose_world(
                    a.grasp, camera_extrinsic)
                target_object_idx, _ = evaluate_pj_grasp(
                    grasp_pose_w, meshes, p)
            elif isinstance(a.grasp, SuctionPoint2D):
                grasp_pose_w = sc_grasp_planner.get_grasp_pose_world(
                    a.grasp, camera_extrinsic)
                target_object_idx, _ = evaluate_sc_grasp(
                    grasp_pose_w, meshes, p)

            # save target object idx
            pickle.dump(
                target_object_idx,
                open(os.path.join(episode_dir, 'target_object_idx_{:03d}.pkl'.format(i)), 'wb'))
            print('episode: {}, step: {}, target object idx: {}, took: {:.3f}s'.format(
                episode, i, target_object_idx, time.time() - start_time))
