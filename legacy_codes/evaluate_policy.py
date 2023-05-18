import os
import shutil
import time
import logging
import tqdm
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from multiprocessing import Pool
import pickle

from src.grasping.grasp_planner import GraspPlanner
from src.grasping.grasp_metric import ParallelJawGraspMetric


def evaluate_grasp(grasp_pose, meshes, mesh_poses):
    # gather inputs for parallel processing
    input_tuple = []
    for mesh, mesh_pose in zip(meshes, mesh_poses):
        grasp_pose_object = np.linalg.inv(mesh_pose) @ grasp_pose
        input_tuple.append((mesh, grasp_pose_object, 1.0, False, False))

    # parallel processing for grasp quality evaluation
    with Pool(8) as p:
        output = p.starmap(ParallelJawGraspMetric.compute, input_tuple)

    # gather outputs
    quality = []
    for q, _ in output:
        if q is None:
            quality.append(0.0)
        else:
            quality.append(q)

    max_quality = np.max(quality)

    if np.max(max_quality) > 0.002:
        return 1.0
    else:
        return 0.0


def visulaize_scene(grasp_pose, meshes, mesh_poses):
    mesh_world = []
    for mesh, mesh_pose in zip(meshes, mesh_poses):
        _mesh = mesh.copy()
        _mesh.apply_transform(mesh_pose)
        mesh_world.append(_mesh)

    scene = trimesh.Scene(mesh_world)
    scene.add_geometry(trimesh.primitives.Box(
        extents=[0.08, 0.005, 0.005], transform=grasp_pose))
    scene.add_geometry(trimesh.creation.axis(origin_size=0.01, transform=grasp_pose))
    scene.add_geometry(trimesh.creation.axis(origin_size=0.01))
    scene.show()


if __name__ == "__main__":
    target_dir = 'data/egad_eval'
    run_policy = 'beta_process'
    # run_policy = 'cem'

    # load meshes
    object_file_list = np.load(os.path.join(target_dir, 'common/object_files.npy'), allow_pickle=True)
    meshes = []
    for file in object_file_list:
        object_name = file.split('/')[-1].split('.')[0]
        mesh_file = file + '/' + object_name + '.obj'
        mesh = trimesh.load(mesh_file)
        meshes.append(mesh)

    # load camera parameters
    camera_intrinsic = np.load(os.path.join(target_dir, 'common/camera_intrinsics.npy'))
    camera_extrinsic = np.load(os.path.join(target_dir, 'common/camera_extrinsics.npy'))

    # set grasp metric
    ParallelJawGraspMetric.num_edges = 8
    ParallelJawGraspMetric.finger_radius = 0.005
    ParallelJawGraspMetric.torque_scale = 1000.0
    ParallelJawGraspMetric.gripper_width = 0.085
    ParallelJawGraspMetric.quality_method = 'ferrari_canny_l1'

    # load grasp planner
    history_dir = os.path.join(target_dir, run_policy, 'run_history')
    pj_policy_config_file = os.path.join(history_dir, '..', 'policy_config.yaml')
    pj_grasp_planner = GraspPlanner(pj_policy_config_file)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(logging.WARNING)

    # load run action and reward
    if run_policy == 'cem':
        for i in range(800):
            with open(os.path.join(history_dir, 'action_' + str(i) + '.pkl'), 'rb') as f:
                action = pickle.load(f)
            with open(os.path.join(history_dir, 'reward_' + str(i) + '.pkl'), 'rb') as f:
                reward = pickle.load(f)
            pj_grasp_planner.update_policy(action, reward)
    elif run_policy == 'beta_process':
        for i in range(800):
            with open(os.path.join(history_dir, 'action_' + str(i) + '.pkl'), 'rb') as f:
                action = pickle.load(f)
            with open(os.path.join(history_dir, 'reward_' + str(i) + '.pkl'), 'rb') as f:
                reward = pickle.load(f)
            pj_grasp_planner.update_policy(action, reward)

    # visualize feature map
    pj_grasp_planner.policy.visualize_feature_map(show_features=True)

    # execute policy
    num_success = 0
    eval_dir = os.path.join(target_dir, run_policy, 'eval_history')
    os.makedirs(eval_dir, exist_ok=True)
    for i in tqdm.tqdm(range(800, 1000)):
        # load images
        color_image = np.load(os.path.join(target_dir, 'state/color_image_' + str(i) + '.npy'))
        depth_image = np.load(os.path.join(target_dir, 'state/depth_image_' + str(i) + '.npy'))
        seg_image = np.load(os.path.join(target_dir, 'state/seg_image_' + str(i) + '.npy'))
        mesh_poses = np.load(os.path.join(target_dir, 'state/object_pose_' + str(i) + '.npy'))

        # execute policy
        action = pj_grasp_planner.execute_policy(
            color_image, depth_image, seg_image, camera_intrinsic)
        pj_grasp_pose_world = pj_grasp_planner.get_grasp_pose_world(action.grasp, camera_extrinsic)

        # visulaize scene
        # visulaize_scene(pj_grasp_pose_world, meshes, mesh_poses)

        # evaluate grasp
        reward = evaluate_grasp(pj_grasp_pose_world, meshes, mesh_poses)

        # save action and reward
        with open(os.path.join(eval_dir, 'action_' + str(i) + '.pkl'), 'wb') as f:
            pickle.dump(action, f)
        with open(os.path.join(eval_dir, 'reward_' + str(i) + '.pkl'), 'wb') as f:
            pickle.dump(reward, f)

        # update num_success
        num_success += reward
    print('success rate: ', num_success / 200.0)
