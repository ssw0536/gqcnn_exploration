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
    bin_size = run_cfg['bin']['size']
    visualize = False
    verbose = False
    pj_policy_config_file = 'cfg/policy/ucb-ts_128dim_pj.yaml'

    # save dir
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = os.path.join('/media/sungwon/WorkSpace/projects/isaacgym-gqcnn-results', cur_time)
    os.makedirs(save_dir, exist_ok=True)

    # create gym and simulation
    handler = IsaacGymHandler(sim_cfg)
    gym = handler.gym
    sim = handler.sim
    dt = handler.dt
    physics_engine = handler.physics_engine

    # get target object list
    if False:
        object_files = glob.glob('assets/urdf/adv_obj_urdf/*')
        random.shuffle(object_files)
        object_files = object_files[:1]
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
    bin_actor = BinActor(gym, sim, env, bin_size, env_idx)

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
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(logging.WARNING)

    # save common things
    np.save(os.path.join(save_dir, 'object_files.npy'), object_files)
    shutil.copyfile(pj_policy_config_file, os.path.join(save_dir, 'pj_policy_config.yaml'))
    shutil.copyfile('cfg/run/config.yaml', os.path.join(save_dir, 'run_config.yaml'))
    np.save(os.path.join(save_dir, 'camera_intrinsic.npy'), camera.intrinsic)
    np.save(os.path.join(save_dir, 'camera_extrinsic.npy'), camera.extrinsic)

    # execute policy
    num_episode = exp_cfg['num_episode']
    for episode in range(num_episode):
        # set bin box and dorp objects
        bin_actor.set()
        object_actor_set.drop(
            len(object_actor_set),
            bin_size=[bin_size[0]/2.0, bin_size[1]/2.0],
            grid_size=0.15)
        handler.wait(2.0)
        bin_actor.unset()
        handler.wait(2.0)

        # create save dir
        episode_save_dir = os.path.join(save_dir, 'episode_{:04d}'.format(episode))
        os.makedirs(episode_save_dir, exist_ok=True)

        # get states
        num_trial = 0
        num_success = 0

        # use tqdm
        desc = '[E{:04d}] {}/{}'
        pbar = tqdm.tqdm(
            total=num_trial+1,
            desc=desc.format(episode, num_success, num_trial),
            dynamic_ncols=True)
        while True:
            # get states
            handler.render_all_camera_sensor()
            color_image, depth_image, seg_image = camera.get_image()
            object_poses = object_actor_set.get_poses()

            # execute policy
            pj_action = pj_grasp_planner.execute_policy(
                color_image, depth_image, seg_image, camera.intrinsic)
            action = pj_action

            # no action then finish episode
            if action is None:
                break

            # execute parallel jaw grasp
            if isinstance(action.grasp, Grasp2D):
                grasp_pose_w = pj_grasp_planner.get_grasp_pose_world(
                    action.grasp, camera.extrinsic)
                target_object_idx, reward = evaluate_pj_grasp(
                    grasp_pose_w, object_actor_set.meshes, object_poses)
                pj_grasp_planner.update_policy(action, reward)
            if verbose is True:
                print("reward, idx: ", reward, target_object_idx)
            if visualize is True:
                visulaize_scene(grasp_pose_w, object_actor_set.meshes, object_poses)

            # label grasp on image
            labeled_image = depth_image_to_gray_image(depth_image)
            if pj_action is not None:
                labeled_image = label_grasp_on_image(labeled_image, pj_action)
            if visualize is True:
                plt.imshow(labeled_image)
                plt.show()
            # labeled_image = cv2.resize(labeled_image, 0.5)

            # remove object
            if reward > 0.0:
                object_actor_set.remove(target_object_idx)
                num_success += 1
                handler.wait(2.0)
            handler.wait(0.1)

            # save action and reward and object poses and thumbnail
            with open(os.path.join(episode_save_dir, 'action_{:03d}.pkl'.format(num_trial)), 'wb') as f:
                pickle.dump(action, f)
            with open(os.path.join(episode_save_dir, 'reward_{:03d}.pkl'.format(num_trial)), 'wb') as f:
                pickle.dump(reward, f)
            with open(os.path.join(episode_save_dir, 'object_poses_{:03d}.pkl'.format(num_trial)), 'wb') as f:
                pickle.dump(object_poses, f)
            with open(os.path.join(episode_save_dir, 'target_object_idx_{:03d}.pkl'.format(num_trial)), 'wb') as f:
                pickle.dump(target_object_idx, f)
            cv2.imwrite(os.path.join(episode_save_dir, 'thumbnail_{:03d}.png'.format(num_trial)), labeled_image[:, :, ::-1])

            # update progress bar
            num_trial += 1
            pbar.set_description(desc.format(episode, num_success, num_trial))
            pbar.update(1)

        # update progress bar
        pbar.close()

        # save success rate to csv
        success_rate = num_success / num_trial
        cur_time = datetime.datetime.now().strftime("%H-%M-%S")
        with open(os.path.join(save_dir, 'success_rate.csv'), 'a') as f:
            f.write('{},{},{},{}\n'.format(episode, num_success, num_trial, success_rate))
        print('[{}] episode: {:4d}, success rate: {:.3f}'.format(cur_time, episode, success_rate))
