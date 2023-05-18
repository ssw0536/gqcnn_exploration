import os
import glob
import cv2
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
    pj_policy_config_file = 'cfg/policy/beta_process_retrain_pj.yaml'
    sc_policy_config_file = 'cfg/policy/beta_process_retrain_sc.yaml'

    # save dir
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = os.path.join('run_history', cur_time)
    os.makedirs(save_dir, exist_ok=True)

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
        print('load object files from `cfg/object_files.npy`')

    # create envs
    env_idx = 0
    env_lower = gymapi.Vec3(-0.5, -0.5, 0.0)
    env_upper = gymapi.Vec3(0.5, 0.5, 2.0)
    env = gym.create_env(sim, env_lower, env_upper, 1)

    # add bin actor vbn
    bin_actor = BinActor(gym, sim, env, (0.2, 0.5), env_idx)

    # create camera
    camera = Camera(gym, sim, env, cam_cfg)

    # create object_actor_set and load object meshes
    meshes = []
    for file in object_files:
        object_name = file.split('/')[-1].split('.')[0]
        mesh_file = file + '/' + object_name + '.obj'
        mesh = trimesh.load(mesh_file)
        meshes.append(mesh)
    object_actor_set = ObjectActorSet(gym, sim, env, object_list, env_idx)
    object_actor_set.meshes = meshes

    # load grasp planner
    pj_grasp_planner = GraspPlanner(pj_policy_config_file)
    sc_grasp_planner = GraspPlanner(sc_policy_config_file)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(logging.WARNING)

    # init grasp select action policy
    select_action_policy = RuleBasedSelectGraspAction(True, True)

    # save common things
    np.save(os.path.join(save_dir, 'object_files.npy'), object_files)
    shutil.copyfile(pj_policy_config_file, os.path.join(save_dir, 'pj_policy_config.yaml'))
    shutil.copyfile(sc_policy_config_file, os.path.join(save_dir, 'sc_policy_config.yaml'))
    shutil.copyfile('cfg/run/config.yaml', os.path.join(save_dir, 'run_config.yaml'))
    np.save(os.path.join(save_dir, 'camera_intrinsic.npy'), camera.intrinsic)
    np.save(os.path.join(save_dir, 'camera_extrinsic.npy'), camera.extrinsic)

    # execute policy
    num_episode = exp_cfg['num_episode']
    max_trial = exp_cfg['max_trial']
    for episode in range(num_episode):
        # set bin box and dorp objects
        bin_actor.set()
        object_actor_set.drop(len(object_actor_set))
        handler.wait(1.0)
        bin_actor.unset()
        handler.wait(2.5)

        # create save dir
        episode_save_dir = os.path.join(save_dir, 'episode_{:04d}'.format(episode))
        os.makedirs(episode_save_dir, exist_ok=True)

        # get states
        num_trial = 0
        num_success = 0

        # use tqdm
        desc = '[E{:04d}] {}/{}'
        pbar = tqdm.tqdm(
            range(max_trial),
            desc=desc.format(episode, num_success, num_trial),
            dynamic_ncols=True)
        for step in pbar:
            num_trial += 1

            # get states
            handler.render_all_camera_sensor()
            color_image, depth_image, seg_image = camera.get_image()
            object_poses = object_actor_set.get_poses()

            # execute policy
            pj_action = pj_grasp_planner.execute_policy(
                color_image, depth_image, seg_image, camera.intrinsic)
            sc_action = sc_grasp_planner.execute_policy(
                color_image, depth_image, seg_image, camera.intrinsic)

            # select action and get reward and update gqcnn policy
            action, table = select_action_policy.select(pj_action, sc_action)
            if verbose is True:
                print(table)

            # no action then finish episode
            if action is None:
                select_action_policy.reset()
                break

            # execute parallel jaw grasp
            if isinstance(action.grasp, Grasp2D):
                grasp_pose_w = pj_grasp_planner.get_grasp_pose_world(
                    action.grasp, camera.extrinsic)
                target_object_idx, reward = evaluate_pj_grasp(
                    grasp_pose_w, object_actor_set.meshes, object_poses)
                pj_grasp_planner.update_policy(action, reward)
            elif isinstance(action.grasp, SuctionPoint2D):
                grasp_pose_w = sc_grasp_planner.get_grasp_pose_world(
                    action.grasp, camera.extrinsic)
                target_object_idx, reward = evaluate_sc_grasp(
                    grasp_pose_w, object_actor_set.meshes, object_poses)
                sc_grasp_planner.update_policy(action, reward)
            if verbose is True:
                print("reward, idx: ", reward, target_object_idx)
            if visualize is True:
                visulaize_scene(grasp_pose_w, object_actor_set.meshes, object_poses)

            # label grasp on image
            labeled_image = depth_image_to_gray_image(depth_image)
            if pj_action is not None:
                labeled_image = label_grasp_on_image(labeled_image, pj_action)
            if sc_action is not None:
                labeled_image = label_grasp_on_image(labeled_image, sc_action)
            labeled_image = select_action_policy.label_table_on_image(
                labeled_image, table, action,
                True if reward > 0.0 else False,
                scale=1.0)
            if visualize is True:
                plt.imshow(labeled_image)
                plt.show()

            # remove object
            if reward > 0.0:
                object_actor_set.remove(target_object_idx)
                num_success += 1
                handler.wait(2.0)
            handler.wait(0.1)

            # update select action policy
            select_action_policy.update(action, True if reward > 0.0 else False)

            # save action and reward and object poses and thumbnail
            with open(os.path.join(episode_save_dir, 'action_{:03d}.pkl'.format(step)), 'wb') as f:
                pickle.dump(action, f)
            with open(os.path.join(episode_save_dir, 'reward_{:03d}.pkl'.format(step)), 'wb') as f:
                pickle.dump(reward, f)
            with open(os.path.join(episode_save_dir, 'object_poses_{:03d}.pkl'.format(step)), 'wb') as f:
                pickle.dump(object_poses, f)
            cv2.imwrite(os.path.join(episode_save_dir, 'thumbnail_{:03d}.png'.format(step)), labeled_image[:, :, ::-1])

            # update progress bar
            pbar.set_description(desc.format(episode, num_success, num_trial))

        # save success rate to csv
        success_rate = num_success / num_trial
        cur_time = datetime.datetime.now().strftime("%H-%M-%S")
        with open(os.path.join(save_dir, 'success_rate.csv'), 'a') as f:
            f.write('{},{},{},{}\n'.format(episode, num_success, num_trial, success_rate))
        print('[{}] episode: {:4d}, success rate: {:.3f}'.format(cur_time, episode, success_rate))
