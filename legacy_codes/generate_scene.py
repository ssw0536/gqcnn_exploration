import os
import glob
import random

import tqdm
import numpy as np
import matplotlib.pyplot as plt
from isaacgym import gymapi

from src.simulation.handler import IsaacGymHandler
from src.simulation.handler import ObjectData, ObjectActorSet
from src.simulation.handler import BinActor
from src.simulation.handler import Camera


if __name__ == "__main__":
    sim_cfg = {
        'dt': 0.01,
        'sync_frame_time': True,
    }

    # create gym and simulation
    handler = IsaacGymHandler(sim_cfg)
    gym = handler.gym
    sim = handler.sim
    dt = handler.dt
    physics_engine = handler.physics_engine

    # load object data
    object_files = glob.glob('assets/urdf/egad_eval_set_urdf/*')

    # get target object list
    random.shuffle(object_files)
    object_files = object_files[:20]
    object_list = [ObjectData(gym, sim, p, physics_engine) for p in tqdm.tqdm(object_files)]

    # create envs
    env_idx = 0
    env_lower = gymapi.Vec3(-0.5, -0.5, 0.0)
    env_upper = gymapi.Vec3(0.5, 0.5, 2.0)
    env = gym.create_env(sim, env_lower, env_upper, 1)

    # add actors
    bin_actor = BinActor(gym, sim, env, (0.2, 0.5), env_idx)

    # create camera
    camera = Camera(gym, sim, env)

    # create object_actor_set
    object_actor_set = ObjectActorSet(gym, sim, env, object_list, env_idx)

    # save object file list
    data_dir = 'data/egad_eval'
    common_dir = os.path.join(data_dir, 'common')
    state_dir = os.path.join(data_dir, 'state')
    os.makedirs(common_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)

    # save common data
    np.save(os.path.join(common_dir, 'object_files.npy'), object_files)
    np.save(os.path.join(common_dir, 'camera_intrinsics.npy'), camera.intrinsic)
    np.save(os.path.join(common_dir, 'camera_extrinsics.npy'), camera.extrinsic)

    # run simulation
    for i in tqdm.tqdm(range(5000)):
        # set bin box and dorp objects
        bin_actor.set()
        num_object = random.randint(1, len(object_list))
        object_actor_set.drop(num_object)
        handler.wait(1.0)

        # remove bin
        bin_actor.unset()
        handler.wait(1.0)

        # get images
        handler.render_all_camera_sensor()
        images = camera.get_image()

        # get object poses
        object_poses = object_actor_set.get_pose()

        # save state
        # np.save('data/color_image_{}.npy'.format(i), images[0])
        # np.save('data/depth_image_{}.npy'.format(i), images[1])
        # np.save('data/seg_image_{}.npy'.format(i), images[2])
        # np.save('data/object_pose_{}.npy'.format(i), object_poses)
        np.save(os.path.join(state_dir, 'color_image_{}.npy'.format(i)), images[0])
        np.save(os.path.join(state_dir, 'depth_image_{}.npy'.format(i)), images[1])
        np.save(os.path.join(state_dir, 'seg_image_{}.npy'.format(i)), images[2])
        np.save(os.path.join(state_dir, 'object_pose_{}.npy'.format(i)), object_poses)
