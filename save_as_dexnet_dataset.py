import os
import glob
import pickle as pkl
import numpy as np

from gqcnn.grasping.policy import GraspAction
from gqcnn.grasping.grasp import Grasp2D, SuctionPoint2D


def save_as_dexnet_dataset(actions, rewards, mode, save_dir):
    # mode should be 'parallel_jaw' or 'suction'
    if mode not in ['parallel_jaw', 'suction']:
        raise ValueError('mode should be either parallel_jaw or suction. got {}'.format(mode))

    # check if actions are empty
    if len(actions) == 0:
        print('No actions to save for {} mode'.format(mode))
        return
    print('Generating {} data for {} mode'.format(len(actions), mode))

    # get indices
    indices = np.arange(len(actions))

    # get data length
    bundle_size = 100
    data_length = len(indices)
    num_bundles = int(np.ceil(data_length / bundle_size))

    # create directory
    os.makedirs(os.path.join(save_dir, mode, 'norm_params'))
    os.makedirs(os.path.join(save_dir, mode, 'splits', 'image_wise'))
    for bundle_idx in range(num_bundles):
        os.makedirs(os.path.join(save_dir, mode, 'tensors_uncompressed', 'tf_depth_ims_{:05d}'.format(bundle_idx)))
        os.makedirs(os.path.join(save_dir, mode, 'tensors_uncompressed', 'grasps_{:05d}'.format(bundle_idx)))
        os.makedirs(os.path.join(save_dir, mode, 'tensors_uncompressed', 'grasp_metrics_{:05d}'.format(bundle_idx)))

    # save norm params
    norm_dir = os.path.join('data', 'norm_params')
    im_mean = np.load(os.path.join(norm_dir, '{}/im_mean.npy'.format(mode)))
    im_std = np.load(os.path.join(norm_dir, '{}/im_std.npy'.format(mode)))
    pose_mean = np.load(os.path.join(norm_dir, '{}/pose_mean.npy'.format(mode)))
    pose_std = np.load(os.path.join(norm_dir, '{}/pose_std.npy'.format(mode)))
    np.save(os.path.join(save_dir, mode, 'norm_params', 'im_mean.npy'), im_mean)
    np.save(os.path.join(save_dir, mode, 'norm_params', 'im_std.npy'), im_std)
    np.save(os.path.join(save_dir, mode, 'norm_params', 'pose_mean.npy'), pose_mean)
    np.save(os.path.join(save_dir, mode, 'norm_params', 'pose_std.npy'), pose_std)
    print('saved norm params for {} mode'.format(mode))

    # save split indices
    train_indices = np.random.choice(data_length, int(data_length * 0.8), replace=False)
    val_indices = np.setdiff1d(np.arange(data_length), train_indices)
    if len(train_indices) + len(val_indices) != data_length:
        raise ValueError('train_indices and val_indices should be unique. got {} and {}'.format(train_indices, val_indices))
    train_indices = np.sort(train_indices)
    val_indices = np.sort(val_indices)
    np.savez(os.path.join(save_dir, mode, 'splits', 'image_wise', 'train_indices.npz'), train_indices)
    np.savez(os.path.join(save_dir, mode, 'splits', 'image_wise', 'val_indices.npz'), val_indices)
    print('saved split indices for {} mode'.format(mode))

    # save tensors
    data_idx = 0
    for idx in indices:
        action = actions[idx]
        reward = rewards[idx]

        # check gripper mode
        if mode == 'parallel_jaw':
            if not isinstance(action.grasp, Grasp2D):
                continue
        elif mode == 'suction':
            if not isinstance(action.grasp, SuctionPoint2D):
                continue

        # convert to gqcnn dataset format
        tf_depth_im = np.expand_dims(action.image.data, axis=-1)
        grasp = np.array([
            action.grasp.center[0], action.grasp.center[1], action.grasp.depth,
            action.grasp.angle, action.grasp.approach_angle, 0.0])
        grasp_metric = reward

        # save data
        bundle_idx = int(np.floor(data_idx / bundle_size))
        file_idx = data_idx % bundle_size
        np.save(os.path.join(save_dir, mode, 'tensors_uncompressed', 'tf_depth_ims_{:05d}'.format(bundle_idx), '{}.npy'.format(file_idx)), tf_depth_im)
        np.save(os.path.join(save_dir, mode, 'tensors_uncompressed', 'grasps_{:05d}'.format(bundle_idx), '{}.npy'.format(file_idx)), grasp)
        np.save(os.path.join(save_dir, mode, 'tensors_uncompressed', 'grasp_metrics_{:05d}'.format(bundle_idx), '{}.npy'.format(file_idx)), grasp_metric)

        # update data_idx
        data_idx += 1
    print('{} data for {} mode'.format(data_idx, mode))


if __name__ == '__main__':
    ###################
    # User Parameters #
    ###################
    # run information
    exp_name = '2023-05-04-18-09-51'
    run_dir = os.path.join('run_history', exp_name)

    # save information
    save_dir = '/media/sungwon/WorkSpace/dataset/AUTOLAB/gqcnn_training_dataset/dexnet_4.0_synt'
    save_dir = os.path.join(save_dir, exp_name)

    # --------------------------------- #
    # load all actions and rewards
    pj_actions = []
    pj_rewards = []
    sc_actions = []
    sc_rewards = []
    episode_dirs = glob.glob(os.path.join(run_dir, 'episode_*'))
    episode_dirs.sort()
    for episode_dir in episode_dirs:
        action_files = glob.glob(os.path.join(episode_dir, 'action_*.pkl'))
        rewrad_files = glob.glob(os.path.join(episode_dir, 'reward_*.pkl'))
        action_files.sort()
        rewrad_files.sort()
        for action_file, reward_file in zip(action_files, rewrad_files):
            action = pkl.load(open(action_file, 'rb'))
            reward = pkl.load(open(reward_file, 'rb'))
            if isinstance(action.grasp, Grasp2D):
                pj_actions.append(action)
                pj_rewards.append(reward)
            elif isinstance(action.grasp, SuctionPoint2D):
                sc_actions.append(action)
                sc_rewards.append(reward)
            else:
                raise ValueError('Unknown grasp type: {}'.format(type(action.grasp)))

    # save as dexnet dataset
    save_as_dexnet_dataset(pj_actions, pj_rewards, 'parallel_jaw', save_dir)
    save_as_dexnet_dataset(sc_actions, sc_rewards, 'suction', save_dir)
