import os
import shutil
import logging
import tqdm
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import pickle

from src.grasping.grasp_planner import GraspPlanner
from src.grasping.grasp_metric import evaluate_pj_grasp, evaluate_sc_grasp


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

    # show images
    if False:
        color_image = np.load(os.path.join(target_dir, 'state/color_image_0.npy'))
        depth_image = np.load(os.path.join(target_dir, 'state/depth_image_0.npy'))
        seg_image = np.load(os.path.join(target_dir, 'state/seg_image_0.npy'))
        plt.subplot(1, 3, 1)
        plt.imshow(color_image)
        plt.subplot(1, 3, 2)
        plt.imshow(depth_image)
        plt.subplot(1, 3, 3)
        plt.imshow(seg_image)
        plt.show()

    # load grasp planner
    pj_policy_config_file = 'cfg/policy/beta_process_pj.yaml'
    pj_grasp_planner = GraspPlanner(pj_policy_config_file)
    sc_policy_config_file = 'cfg/policy/beta_process_sc.yaml'
    sc_grasp_planner = GraspPlanner(sc_policy_config_file)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(logging.WARNING)

    # test function
    i = 0
    color_image = np.load(os.path.join(target_dir, 'state/color_image_' + str(i) + '.npy'))
    depth_image = np.load(os.path.join(target_dir, 'state/depth_image_' + str(i) + '.npy'))
    seg_image = np.load(os.path.join(target_dir, 'state/seg_image_' + str(i) + '.npy'))
    mesh_poses = np.load(os.path.join(target_dir, 'state/object_pose_' + str(i) + '.npy'))

    # execute policy
    pj_action = pj_grasp_planner.execute_policy(
        color_image, depth_image, seg_image, camera_intrinsic)
    sc_action = sc_grasp_planner.execute_policy(
        color_image, depth_image, seg_image, camera_intrinsic)

    # get grasp pose in world
    pj_graps_pose_world = pj_grasp_planner.get_grasp_pose_world(pj_action.grasp, camera_extrinsic)
    sc_grasp_pose_world = sc_grasp_planner.get_grasp_pose_world(sc_action.grasp, camera_extrinsic)

    # visulaize_scene(sc_grasp_pose_world, meshes, mesh_poses)
    quality = evaluate_sc_grasp(sc_grasp_pose_world, meshes, mesh_poses, visualize=False)
    print(quality)

    exit()


    # create save directory
    run_dir = os.path.join(target_dir, 'beta_process', 'run')
    os.makedirs(run_dir, exist_ok=True)
    shutil.copyfile(pj_policy_config_file, os.path.join(run_dir, '..', 'policy_config.yaml'))

    # execute policy
    num_success = 0
    for i in tqdm.tqdm(range(800)):
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
        with open(os.path.join(run_dir, 'action_' + str(i) + '.pkl'), 'wb') as f:
            pickle.dump(action, f)
        with open(os.path.join(run_dir, 'reward_' + str(i) + '.pkl'), 'wb') as f:
            pickle.dump(reward, f)

        # update policy
        pj_grasp_planner.update_policy(action, reward)

    # visualize feature map
    # pj_grasp_planner.policy.visualize_feature_map(show_features=True)
