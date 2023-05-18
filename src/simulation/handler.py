# import python modules
import os
import trimesh

# import isaacgym modules
from isaacgym import gymapi, gymutil
from isaacgym import gymtorch

# import 3rd party modules
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from .utils import transform_to_numpy, transform_to_gym


class IsaacGymHandler(object):
    def __init__(self, sim_cfg):
        # parse arguments / isaacgym configuration
        args = gymutil.parse_arguments(
            description='GQ-CNN simulation',
            headless=True,
            no_graphics=True,
        )

        # initial parameters
        self._dt = sim_cfg["dt"]
        self._headless = args.headless
        self._sync_frame_time = sim_cfg["sync_frame_time"]
        self._enable_viewer_sync = True
        self._wait_timer = 0.0
        self._physics_engine = args.physics_engine

        # create gym interface
        self._gym = gymapi.acquire_gym()
        self._sim = self._create_sim(self._gym, self._dt, args)
        self._viewer = self._create_veiwer(self._gym, self._sim, self._headless)
        self._device = "cuda:0" if args.use_gpu_pipeline else "cpu"

    def __del__(self):
        self._gym.destroy_sim(self._sim)
        if not self._headless:
            self._gym.destroy_viewer(self._viewer)

    @property
    def dt(self):
        return self._dt

    @property
    def gym(self):
        return self._gym

    @property
    def sim(self):
        return self._sim

    @property
    def physics_engine(self):
        return self._physics_engine

    @staticmethod
    def _create_sim(gym, dt, args):
        """Create simulation.

        Args:
            gym (Gym): isaacgym gym
            dt (float): simulation time step in seconds
            args (argparse.Namespace): isaacgym arguments

        Returns:
            gymapi.Sim: isaacgym sim handler
        """
        # configure sim
        # check `issacgy.gymapi.SimParams` for more options
        sim_params = gymapi.SimParams()
        sim_params.dt = dt

        # set physics engine
        if args.physics_engine == gymapi.SIM_FLEX:
        # if args.physics_engine == gymapi.SIM_FLEX and not args.use_gpu_pipeline:
            sim_params.substeps = 3
            sim_params.flex.solver_type = 5
            sim_params.flex.num_outer_iterations = 4
            sim_params.flex.num_inner_iterations = 20
            sim_params.flex.relaxation = 0.75
            sim_params.flex.warm_start = 0.8
            sim_params.flex.shape_collision_margin = 0.001
            sim_params.flex.friction_mode = 2
        elif args.physics_engine == gymapi.SIM_PHYSX:
            sim_params.substeps = 5
            sim_params.physx.solver_type = 4
            sim_params.physx.num_position_iterations = 4
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.num_threads = args.num_threads
            sim_params.physx.use_gpu = args.use_gpu
            sim_params.physx.rest_offset = 0.0
        else:
            raise Exception("GPU pipeline is only available with PhysX")

        # set gpu pipeline
        sim_params.use_gpu_pipeline = args.use_gpu_pipeline

        # set up axis as Z-up
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0, 0, -9.81)

        # create sim
        sim = gym.create_sim(
            args.compute_device_id,
            args.graphics_device_id,
            args.physics_engine,
            sim_params)
        if sim is None:
            raise Exception("*** Failed to create sim")

        # create ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
        plane_params.distance = 0
        plane_params.static_friction = 0.3
        plane_params.dynamic_friction = 0.15
        plane_params.restitution = 0
        gym.add_ground(sim, plane_params)
        return sim

    @staticmethod
    def _create_veiwer(gym, sim, headless):
        """Create viewer.

        Args:
            gym (gymapi.Gym): isaacgym gym.
            sim (gymapi.Sim): isaacgym sim handler.
            headless (bool): whether to run in headless mode.

        Returns:
            gymapi.Viewer: isaacgym viewer handler. None if headless.
        """
        if headless:
            return None
        else:
            # create viewer
            viewer = gym.create_viewer(sim, gymapi.CameraProperties())

            # set viewer camera pose
            cam_pos = gymapi.Vec3(4.5, 3.0, 2.0)
            cam_target = gymapi.Vec3(2.0, 0.5, 0.0)
            gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

            # key callback
            gym.subscribe_viewer_keyboard_event(
                viewer, gymapi.KEY_ESCAPE, "QUIT")
            gym.subscribe_viewer_keyboard_event(
                viewer, gymapi.KEY_V, "toggle_viewer_sync")

            if viewer is None:
                raise Exception("*** Failed to create viewer")
            return viewer

    def render(self):
        """Draw the frame to the viewer, and check for keyboard events."""
        if self._viewer:
            # check for window closed
            if self._gym.query_viewer_has_closed(self._viewer):
                exit()

            # check for keyboard events
            for evt in self._gym.query_viewer_action_events(self._viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self._enable_viewer_sync = not self._enable_viewer_sync

            # step graphics
            if self._enable_viewer_sync:
                self._gym.step_graphics(self._sim)
                self._gym.draw_viewer(self._viewer, self._sim, True)
                if self._sync_frame_time:
                    self._gym.sync_frame_time(self._sim)
        if self._headless:
            self._gym.step_graphics(self._sim)

    def wait(self, sleep_time=0.5):
        """Wait for a given time in seconds.

        Blocks the execution of the program for a given time in seconds.
        Simulation steps are executed in the background.

        Args:
            sleep_time (float, optional): sleep time. Defaults to 0.5.

        Returns:
            success (bool): True if successful, False otherwise.
        """
        # wait for time seconds
        while not self._wait_timer > sleep_time:
            # step the physics
            self._gym.simulate(self._sim)
            # refresh results
            self._gym.fetch_results(self._sim, True)
            self._gym.refresh_dof_state_tensor(self._sim)
            # step rendering
            self.render()
            # update timer
            self._wait_timer += self._dt

        self._wait_timer = 0.0
        return True

    def step(self):
        """Step the simulation."""
        # step the physics
        self._gym.simulate(self._sim)
        # refresh results
        self._gym.fetch_results(self._sim, True)
        self._gym.refresh_dof_state_tensor(self._sim)
        # step rendering
        self.render()

    def render_all_camera_sensor(self):
        """Render all camera sensors."""
        self.gym.render_all_camera_sensors(self.sim)


class HandHandler(object):
    def __init__(self, gym_handler):
        assert isinstance(gym_handler, IsaacGymHandler)
        self._gym_handler = gym_handler

        self._gripper_width = 0.08

        self._asset, self._num_dofs = self._load_asset()

        self._hand_handles = []
        self._default_dof_pos = []

    def _load_asset(self):
        """Create environments."""
        # get sim and gym handles
        gym = self._gym_handler._gym
        sim = self._gym_handler._sim

        # load hand asset
        asset_root = "./assets"
        hand_asset_file = "urdf/franka_description/robots/franka_hand.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.thickness = 0.001
        asset_options.disable_gravity = True
        asset_options.density = 1000
        asset_options.override_inertia = True
        asset_options.flip_visual_attachments = True
        hand_asset = gym.load_asset(
            sim, asset_root, hand_asset_file, asset_options)

        # count num dofs
        num_dofs = gym.get_asset_dof_count(hand_asset)
        return hand_asset, num_dofs

    def create_actor(self, env, group, filter):
        # get sim and gym handles
        gym = self._gym_handler._gym

        # default hand pose
        hand_offset = 1
        hand_pose = gymapi.Transform()
        hand_pose.p = gymapi.Vec3(0, 0, hand_offset + 0.1122)
        hand_pose.r = gymapi.Quat(0, 1, 0, 0)

        # default hand dof state
        num_dofs = gym.get_asset_dof_count(self._asset)
        default_dof_pos = np.array(
            [0.0, 0.0, hand_offset, 0.0, self._gripper_width / 2.0, self._gripper_width / 2.0],
            dtype=np.float32)
        default_dof_state = np.zeros(num_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] = default_dof_pos

        # configure dof properties
        hand_dof_props = gym.get_asset_dof_properties(self._asset)
        hand_dof_props["driveMode"][:4].fill(gymapi.DOF_MODE_POS)
        hand_dof_props["stiffness"][:4].fill(1000.0)
        hand_dof_props["damping"][:4].fill(100.0)
        hand_dof_props["driveMode"][4:].fill(gymapi.DOF_MODE_POS)
        hand_dof_props["stiffness"][4:].fill(800.0)
        hand_dof_props["damping"][4:].fill(40.0)

        # create hand actor
        hand_handle = gym.create_actor(
            env, self._asset, hand_pose, "hand", group, filter)

        # set hand dof states
        gym.set_actor_dof_properties(env, hand_handle, hand_dof_props)
        gym.set_actor_dof_states(env, hand_handle, default_dof_state, gymapi.STATE_ALL)
        gym.set_actor_dof_position_targets(env, hand_handle, default_dof_pos)

        # append hand handle
        self._default_dof_pos.append(default_dof_pos)

    def prepare_tensor(self, num_envs):
        gym = self._gym_handler._gym
        sim = self._gym_handler._sim
        device = self._gym_handler._device

        # get dof states
        _dof_states = gym.acquire_dof_state_tensor(sim)
        self._dof_states = gymtorch.wrap_tensor(_dof_states)
        self._dof_pos = self._dof_states[:, 0].view(num_envs, self._num_dofs)
        self._dof_vel = self._dof_states[:, 1].view(num_envs, self._num_dofs)

        # make default dof pos tensor
        self._default_dof_pos = np.array(self._default_dof_pos, dtype=np.float32)
        self._default_dof_pos = torch.from_numpy(self._default_dof_pos).to(device)

    def set_dof_position_targets(self, dof_position_targets):
        """Set the target position of the hand dofs.

        Args:
            dof_position_targets (np.array): target positions of the hand dofs.
        """
        # get sim and gym handles
        gym = self._gym_handler._gym
        sim = self._gym_handler._sim

        # set hand dof position targets
        gym.set_dof_position_target_tensor(
            sim, gymtorch.unwrap_tensor(dof_position_targets))

    def get_dof_positions(self):
        """Get the current position of the hand dofs.

        Returns:
            np.array: current positions of the hand dofs.
        """
        return torch.clone(self._dof_pos)


class ObjectData(object):
    def __init__(self, gym, sim, target_file, physics_engine=gymapi.SIM_FLEX):
        # load object asset
        name = os.path.basename(target_file)
        name = name.split('.')[0]

        asset = self._load_asset(gym, sim, target_file, physics_engine)
        stable_poses, stable_probs = self._load_stable_poses(target_file)
        mesh = self._load_mesh(target_file)

        # store data
        self._asset = asset
        self._mesh = mesh
        self._stable_poses = stable_poses
        self._stable_probs = stable_probs
        self._name = name

    def _load_asset(self, gym, sim, target_file, physics_engine):
        name = os.path.basename(target_file)
        name = name.split('.')[0]
        target_file = os.path.join(target_file, name + ".urdf")

        # load object asset
        asset_root = "./"
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.005
        asset_options.fix_base_link = False
        asset_options.thickness = 0.005
        asset_options.override_inertia = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        if physics_engine == gymapi.SIM_PHYSX:
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params.resolution = 300000
            asset_options.vhacd_params.max_convex_hulls = 50
            asset_options.vhacd_params.max_num_vertices_per_ch = 1000
        object_asset = gym.load_asset(sim, asset_root, target_file, asset_options)
        return object_asset

    def _load_stable_poses(self, target_file):
        stable_poses = np.load('{}/stable_poses.npy'.format(target_file))
        stable_probs = np.load('{}/stable_prob.npy'.format(target_file))
        return stable_poses, stable_probs

    def _load_mesh(self, target_file):
        name = os.path.basename(target_file)
        name = name.split('.')[0]
        mesh = trimesh.load('{}/{}.obj'.format(target_file, name))
        return mesh

    @property
    def name(self):
        return self._name

    @property
    def asset(self):
        return self._asset

    @property
    def stable_poses(self):
        return self._stable_poses

    @property
    def stable_probs(self):
        return self._stable_probs

    @property
    def mesh(self):
        return self._mesh

    def get_random_stable_pose(self):
        """Get a random stable pose.

        Returns:
            np.array: random stable pose.
        """
        # get stable pose
        rand_idx = np.random.randint(0, len(self._stable_poses))
        stable_pose = self._stable_poses[rand_idx]
        return stable_pose

    def get_stochastic_stable_pose(self):
        """Get a stochastic stable pose.

        Returns:
            np.array: stochastic stable pose.
        """
        # get stable pose
        stable_probs = self._stable_probs
        stable_probs = stable_probs / np.sum(stable_probs)
        rand_idx = np.random.choice(len(self._stable_poses), p=stable_probs)
        stable_pose = self._stable_poses[rand_idx]
        return stable_pose


class ObjectActorSet(object):
    def __init__(self, gym, sim, env, object_data_list, group):
        self._gym = gym
        self._sim = sim
        self._env = env
        self._object_data_list = object_data_list

        # create object actor set
        self._actor_handles = self._create_actor_set(object_data_list, group)
        self._rigid_body_handles = self._get_rigid_body_handles(env, self._actor_handles)
        self._set_segmentation_id(env, self._actor_handles)

    def _create_actor_set(self, object_data_list, collision_group):
        gym = self._gym
        env = self._env

        # create object actor set
        handle_list = []
        for od in object_data_list:
            assert isinstance(od, ObjectData)

            # get random pose
            rand_idx = np.random.randint(0, len(od.stable_poses))
            pose_numpy = od.stable_poses[rand_idx]
            pose_gym = transform_to_gym(pose_numpy)
            handle = gym.create_actor(
                env, od.asset, pose_gym, od.name, collision_group, 0)
            handle_list.append(handle)
        return handle_list

    def _get_rigid_body_handles(self, env, actor_handles):
        gym = self._gym

        rigid_body_handles = []
        for ah in actor_handles:
            rb_handle = gym.get_actor_rigid_body_handle(env, ah, 0)
            rigid_body_handles.append(rb_handle)
        return rigid_body_handles

    def _set_segmentation_id(self, env, handles):
        gym = self._gym
        for handle in handles:
            gym.set_rigid_body_segmentation_id(env, handle, 0, 1)

    def __len__(self):
        """Get number of objects in the set.

        Returns:
            num_objects: int, number of objects in the set.
        """
        return len(self._actor_handles)

    def get_pose(self, i):
        """Get pose of the i-th object in the set.

        Args:
            i (`int`): index of the object.

        Returns:
            pose: (4, 4) numpy array.
        """
        pose = self._gym.get_rigid_transform(
            self._env, self._rigid_body_handles[i])
        pose = transform_to_numpy(pose)
        return pose

    def get_poses(self):
        """Get pose of all objects in the set.

        Returns:
            poses: (N, 4, 4) numpy array, where N is the number of objects.
        """
        poses = []
        for i in range(len(self)):
            pose = self.get_pose(i)
            poses.append(pose)
        poses = np.array(poses)
        return poses

    def set_pose(self, i, pose):
        """Set pose of the i-th object in the set.

        Args:
            i (`int`): index of the object.
            pose (`numpy.ndarray`): (4, 4) numpy array.
        """
        pose = transform_to_gym(pose)
        self._gym.set_rigid_transform(
            self._env, self._rigid_body_handles[i], pose)

    def set_poses(self, poses):
        """Set pose of all objects in the set.

        Args:
            poses: (N, 4, 4) numpy array, where N is the number of objects.
        """
        for i, pose in enumerate(poses):
            self.set_pose(i, pose)

    def get_velocity(self, i):
        """Get velocity of all objects in the set.

        Returns:
            velocities: (6, ) numpy array, where N is the number of objects.
        """
        linear_vel = self._gym.get_rigid_linear_velocity(
            self._env, self._rigid_body_handles[i])
        angular_vel = self._gym.get_rigid_angular_velocity(
            self._env, self._rigid_body_handles[i])

        # convert to numpy array
        velocity = np.array(
            [linear_vel.x, linear_vel.y, linear_vel.z,
                angular_vel.x, angular_vel.y, angular_vel.z])
        return velocity

    def set_velocity(self, i, velocity):
        """Set velocity of all objects in the set.

        Args:
            i (`int`): index of the object.
            velocity (`numpy.ndarray`): (6, ) numpy array. 3 linear velocity, 3 angular velocity.
        """
        linear_vel = gymapi.Vec3(velocity[0], velocity[1], velocity[2])
        angular_vel = gymapi.Vec3(velocity[3], velocity[4], velocity[5])
        self._gym.set_rigid_linear_velocity(
            self._env, self._rigid_body_handles[i], linear_vel)
        self._gym.set_rigid_angular_velocity(
            self._env, self._rigid_body_handles[i], angular_vel)

    def disable_gravity(self, i):
        """Disable gravity for all objects in the set.

        Args:
            i (`int`): index of the object.
        """
        obj_props = self._gym.get_actor_rigid_body_properties(
            self._env, self._actor_handles[i])
        obj_props[0].flags = gymapi.RIGID_BODY_DISABLE_GRAVITY
        self._gym.set_actor_rigid_body_properties(
            self._env, self._actor_handles[i], obj_props, False)

    def enable_gravity(self, i):
        """Enable gravity for all objects in the set.

        Args:
            i (`int`): index of the object.
        """
        obj_props = self._gym.get_actor_rigid_body_properties(
            self._env, self._actor_handles[i])
        obj_props[0].flags = gymapi.RIGID_BODY_NONE
        self._gym.set_actor_rigid_body_properties(
            self._env, self._actor_handles[i], obj_props, False)

    def drop(self, target_object_num, bin_size=[0.2, 0.2], grid_size=0.1):
        """Drop objects from the sky.

        Args:
            target_object_num (int): number of objects to drop.
            bin_size (list): size of the bin to drop objects. [width(x), depth(y)].
            grid_size (float): size of the grid to drop objects.
        """
        # set gravity and zero velocity
        for i in range(len(self)):
            self.enable_gravity(i)
            self.set_velocity(i, np.zeros(6))

        # calculate grid number
        num_objects = len(self)
        x_num = np.floor(bin_size[0] / grid_size).astype(np.int32)
        y_num = np.floor(bin_size[1] / grid_size).astype(np.int32)
        z_num = np.ceil(num_objects / x_num / y_num).astype(np.int32)
        grid_num = [x_num, y_num, z_num]

        # get grid xyz
        grid_x = np.linspace(
            -bin_size[0] / 2 + grid_size / 2,
            bin_size[0] / 2 - grid_size / 2,
            grid_num[0])
        grid_y = np.linspace(
            -bin_size[1] / 2 + grid_size / 2,
            bin_size[1] / 2 - grid_size / 2,
            grid_num[1])
        grid_z = np.linspace(0.1, grid_num[2] * grid_size, grid_num[2])
        grid_xyz = np.meshgrid(grid_x, grid_y, grid_z)
        grid_xyz = np.stack(grid_xyz, axis=-1).reshape(-1, 3)

        # offset pose to control the number of objects in the bin
        offset_xyz = np.zeros((num_objects, 3)) + [2.0, 2.0, 0.0]
        offset_xyz[:target_object_num] = np.array([0.0, 0.0, 0.0])

        # shuffle offset xyz
        np.random.shuffle(offset_xyz)

        # get object xyz pose
        object_poses = []
        rand_grid_idx = np.random.choice(
            np.arange(len(grid_xyz)), size=num_objects, replace=False)
        for i in range(num_objects):
            rand_obj_pose = self._object_data_list[i].get_random_stable_pose()

            # apply random rotation
            rand_obj_pose[:3, :3] = np.matmul(
                rand_obj_pose[:3, :3], R.random().as_matrix())

            # set grid xyz
            rand_obj_pose[:3, 3] = grid_xyz[rand_grid_idx[i]] + offset_xyz[i]

            # append
            object_poses.append(rand_obj_pose)
        object_poses = np.array(object_poses)

        # set pose
        self.set_poses(object_poses)

    def remove(self, i):
        pose = self.get_pose(i)
        pose[0, 3] = np.random.uniform(5.0, 50.0)
        pose[1, 3] = np.random.uniform(5.0, 50.0)
        pose[2, 3] = np.random.uniform(5.0, 50.0)
        self.set_pose(i, pose)
        self.disable_gravity(i)
        self.set_velocity(i, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


class BinActor(object):
    _asset_up = None
    _asset_down = None
    _asset_left = None
    _asset_right = None

    def __init__(self, gym, sim, env, size, group):
        # size (edge width, depth, height)
        self._gym = gym
        self._sim = sim
        self._env = env
        self._size = size

        # create bin box asset
        if self._asset_up is None:
            BinActor._asset_up = self._create_wall_asset(size[0], size[2])
            BinActor._asset_down = self._create_wall_asset(size[0], size[2])
            BinActor._asset_left = self._create_wall_asset(size[1], size[2])
            BinActor._asset_right = self._create_wall_asset(size[1], size[2])

        # create bin box actor
        self._actor_handles = self._create_bin_box(env, group)
        self._rigid_body_handles = self._get_rigid_body_handles(env, self._actor_handles)

    @property
    def default_pose(self):
        # create bin box actor
        width = self._size[0]
        depth = self._size[1]

        pose_up = gymapi.Transform()
        pose_up.p = gymapi.Vec3(0, 0.5 * depth, 0)
        pose_up.r = gymapi.Quat(0, 0, 0, 1)
        pose_down = gymapi.Transform()
        pose_down.p = gymapi.Vec3(0, -0.5 * depth, 0)
        pose_down.r = gymapi.Quat(0, 0, 0, 1)
        pose_left = gymapi.Transform()
        pose_left.p = gymapi.Vec3(-0.5 * width, 0, 0)
        pose_left.r = gymapi.Quat(0, 0, 0.707, 0.707)
        pose_right = gymapi.Transform()
        pose_right.p = gymapi.Vec3(0.5 * width, 0, 0)
        pose_right.r = gymapi.Quat(0, 0, 0.707, 0.707)
        return [pose_up, pose_down, pose_left, pose_right]

    @property
    def hide_pose(self):
        poses = self.default_pose
        for pose in poses:
            pose.p.z = -10
        return poses

    def _create_wall_asset(self, length, height):
        # size (edge width, depth, height)
        gym = self._gym
        sim = self._sim

        # get size
        # width = size[0]  # x
        # thickness = 0.001  # y
        # height = size[1]  # z

        # create bin box asset
        asset_options = gymapi.AssetOptions()
        asset_options.density = 10.0
        asset_options.fix_base_link = True
        asset = gym.create_box(sim, length, 0.001, height, asset_options)
        return asset

    def _create_bin_box(self, env, group):
        gym = self._gym

        # get default poses
        poses = self.default_pose
        pose_up, pose_down, pose_left, pose_right = poses

        # create bin box actor
        h_up = gym.create_actor(
            env, self._asset_up, pose_up, 'bin_box_up', group, 0)
        h_down = gym.create_actor(
            env, self._asset_down, pose_down, 'bin_box_down', group, 0)
        h_left = gym.create_actor(
            env, self._asset_left, pose_left, 'bin_box_left', group, 0)
        h_right = gym.create_actor(
            env, self._asset_right, pose_right, 'bin_box_right', group, 0)
        handles = [h_up, h_down, h_left, h_right]
        return handles

    def _get_rigid_body_handles(self, env, actor_handles):
        gym = self._gym

        rigid_body_handles = []
        for ah in actor_handles:
            rb_handle = gym.get_actor_rigid_body_handle(env, ah, 0)
            rigid_body_handles.append(rb_handle)
        return rigid_body_handles

    def set(self):
        gym = self._gym
        env = self._env

        # set default pose
        for handle, pose in zip(self._rigid_body_handles, self.default_pose):
            gym.set_rigid_transform(env, handle, pose)

    def unset(self):
        gym = self._gym
        env = self._env

        # set default pose
        for handle, pose in zip(self._rigid_body_handles, self.hide_pose):
            gym.set_rigid_transform(env, handle, pose)


class Camera(object):
    def __init__(self, gym, sim, env, cfg):
        self._gym = gym
        self._sim = sim
        self._env = env

        cx = cfg['cx']
        cy = cfg['cy']
        fx = cfg['fx']
        fy = cfg['fy']

        self._intrinsic = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]])
        self._camera_handle = self._create_camera_sensor()

    def _create_camera_sensor(self):
        """Create camera sensor.

        Returns:
            camera_handle (int): Camera handle.
        """
        gym = self._gym
        env = self._env

        # add camera sensor
        camera_props = gymapi.CameraProperties()
        camera_props.width = int(self.intrinsic[0, 2] * 2.0)
        camera_props.height = int(self.intrinsic[1, 2] * 2.0)
        camera_props.horizontal_fov = 2 * np.arctan2(
            self.intrinsic[0, 2], self.intrinsic[0, 0]) * 180 / np.pi
        camera_props.far_plane = 1.0
        camera_handle = gym.create_camera_sensor(env, camera_props)
        gym.set_camera_transform(camera_handle, env, self._camera_pose)
        return camera_handle

    @property
    def _camera_pose(self):
        """Camera pose in gym coordinate.

        Returns:
            cam_pose (gymapi.Transform): Camera pose in gym coordinate.
        """
        # gym camera pos def: x=optical axis, y=left, z=down | convention: OpenGL
        cam_pose = gymapi.Transform()
        # cam_pose.p = gymapi.Vec3(
        #     -0.7 * np.sin(0.075 * np.pi), 0.0, 0.7 * np.cos(0.075 * np.pi))
        # cam_pose.r = gymapi.Quat.from_axis_angle(
        #     gymapi.Vec3(0, 1, 0), np.deg2rad(90 - 13.5))

        cam_pose.p = gymapi.Vec3(
            -0.75 * np.sin(0.075 * np.pi), 0.0, 0.75 * np.cos(0.075 * np.pi))
        cam_pose.r = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0, 1, 0), np.deg2rad(90 - 13.5))
        return cam_pose

    @property
    def intrinsic(self):
        """Camera intrinsic matrix.

        Returns:
            camera_intr (np.ndarray): 3x3 camera intrinsic matrix.
        """
        return self._intrinsic

    @property
    def extrinsic(self):
        """Camera extrinsic matrix.

        Returns:
            camera_extr (np.ndarray): 4x4 camera extrinsic matrix.
        """
        # get camera extrinsic once
        # convert z = x, x = -y, y = -z
        rot = R.from_quat([
            self._camera_pose.r.x,
            self._camera_pose.r.y,
            self._camera_pose.r.z,
            self._camera_pose.r.w]).as_matrix()
        rot_convert = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        rot = np.dot(rot, rot_convert)

        # get camera extrinsic
        camera_extr = np.eye(4)
        camera_extr[:3, :3] = rot
        camera_extr[:3, 3] = np.array([
            self._camera_pose.p.x,
            self._camera_pose.p.y,
            self._camera_pose.p.z])
        return camera_extr

    def get_image(self):
        """Get camera image.

        Returns:
            color_image (np.ndarray): 3 channel RGB image.
            depth_image (np.ndarray): 1 channel depth image.
            segmask_image (np.ndarray): 1 channel segmentation mask.
        """
        gym = self._gym
        sim = self._sim
        env = self._env

        # get camera image
        color_image = gym.get_camera_image(
            sim, env, self._camera_handle, gymapi.IMAGE_COLOR)  # 4x 8 bit RGBA
        depth_image = gym.get_camera_image(
            sim, env, self._camera_handle, gymapi.IMAGE_DEPTH)
        segmask_image = gym.get_camera_image(
            sim, env, self._camera_handle, gymapi.IMAGE_SEGMENTATION)

        # convert to numpy
        color_image = np.array(color_image)
        color_image = color_image.reshape(
            (int(self.intrinsic[1, 2] * 2), int(self.intrinsic[0, 2] * 2), 4))
        color_image = color_image[:, :, 0:3]
        depth_image = np.array(depth_image, dtype=np.float32) * -1
        segmask_image = np.array(segmask_image)
        return (color_image, depth_image, segmask_image)
