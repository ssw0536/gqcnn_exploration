# import python modules
import os
import time
import yaml
import trimesh

# import isaacgym modules
from isaacgym import gymapi, gymutil
from isaacgym import gymtorch

# import 3rd party modules
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from src.gqcnn import GQCNN
from src.grasp_sampler import AntipodalGrasp, AntipodalGraspSampler
from src.policy import (CrossEntropyGraspingPolicy,
                        PosteriorGraspingPolicy,
                        CrossEntorpyPosteriorGraspingPolicy,
                        QLearningGraspingPolicy,)
from src.grasp_metric import ParallelJawGraspMetric


class SigulatePickingEnv(object):
    def __init__(self):
        # initialize the gym
        self.gym = gymapi.acquire_gym()

        # parse arguments / isaacgym configuration
        self.args = gymutil.parse_arguments(
            description='GQ-CNN simulation',
            headless=True,
            no_graphics=True,
            custom_parameters=[
                {
                    "name": "--config",
                    "type": str,
                    "default": "cfg/config.yaml",
                    "help": "configuration file"
                },
                {
                    "name": "--save_results",
                    "action": 'store_true',
                    "help": "Save the results"
                },
            ],
        )
        # self.num_envs = self.args.num_envs
        self.headless = self.args.headless
        self.save_results = self.args.save_results
        config_file = self.args.config

        # load configuration
        with open(config_file, "r") as f:
            cfg = yaml.safe_load(f)
        sim_cfg = cfg["simulation"]
        policy_cfg = cfg["policy"]
        sampling_cfg = cfg["sampling"]
        self.grasp_metric_cfg = cfg["grasp_metric"]

        # load gqcnn, sampler, and policy
        gqcnn = GQCNN()
        sampler = AntipodalGraspSampler(sampling_cfg)
        if policy_cfg["type"] == "posterior":
            self.policy = PosteriorGraspingPolicy(
                gqcnn, sampler, policy_cfg)
            self.num_update_features_per_step = policy_cfg["posterior"]["num_update_features_per_step"]
        elif policy_cfg["type"] == "cross_entropy":
            self.policy = CrossEntropyGraspingPolicy(
                gqcnn, sampler, policy_cfg)
        elif policy_cfg["type"] == "cross_entropy_posterior":
            self.policy = CrossEntorpyPosteriorGraspingPolicy(
                gqcnn, sampler, policy_cfg)
            self.num_update_features_per_step = policy_cfg["cross_entropy_posterior"]["num_update_features_per_step"]
        elif policy_cfg["type"] == "q_learning":
            prior_gqcnn = GQCNN()
            self.policy = QLearningGraspingPolicy(
                gqcnn, prior_gqcnn, sampler, policy_cfg)

        # load parallel jaw grasp metric
        # TODO: make in config file
        ParallelJawGraspMetric.num_edges = 8
        ParallelJawGraspMetric.finger_radius = 0.01
        ParallelJawGraspMetric.torque_scale = 1000.0
        ParallelJawGraspMetric.gripper_width = 0.8
        ParallelJawGraspMetric.quality_method = 'ferrari_canny_l1'

        # set up simulation
        self.dt = sim_cfg["dt"]
        self.render_freq = 1 / sim_cfg["render_freq"]
        self.num_main_iters = sim_cfg["num_main_iters"]
        self.num_sub_iters = sim_cfg["num_sub_iters"]
        self.target_object_name = sim_cfg["target_object"]
        self.target_dataset_name = sim_cfg["target_dataset"]
        self.object_rand_pose_range = sim_cfg["object_random_pose_range"]
        self.grasp_pose_z_offset = sim_cfg["grasp_pose_z_offset"]
        self.gripper_width = sim_cfg["gripper_width"]

        # choose simulation method
        self.sim_method = sim_cfg['method']
        if self.sim_method == 'multi_stable_pose':
            self.min_stable_pose_prob = sim_cfg[self.sim_method]['min_stable_pose_prob']
            self.max_num_stable_pose = sim_cfg[self.sim_method]['max_num_stable_pose']
        elif self.sim_method == 'single_stable_pose':
            self.stable_pose_idx = sim_cfg[self.sim_method]['stable_pose_idx']
            self.max_num_stable_pose = sim_cfg[self.sim_method]['max_num_stable_pose']

        cx = sim_cfg["camera"]["phoxi"]["cx"]
        cy = sim_cfg["camera"]["phoxi"]["cy"]
        fx = sim_cfg["camera"]["phoxi"]["fx"]
        fy = sim_cfg["camera"]["phoxi"]["fy"]
        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]])

        # configure save directory
        root_dir = '/media/sungwon/WorkSpace/projects/gqcnn_thomphson_sampling'
        self.save_dir = os.path.join(root_dir, self.target_object_name, policy_cfg["type"])
        if self.save_results:
            # create save directory
            os.makedirs(os.path.join(self.save_dir, 'data'), exist_ok=True)

        # save config files
        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, 'config.yaml'), "w") as f:
            yaml.safe_dump(cfg, f)

        # set up tensorboard
        self.writer = SummaryWriter(os.path.join(self.save_dir, 'logs'))

        # custom env variables
        self.envs = []
        self.hand_handles = []
        self.camera_handles = []
        self.object_handles = []
        self.default_dof_pos = []
        self.object_mesh = None

        self.enable_viewer_sync = True
        self.task_status = None
        self.wait_timer = 0.0
        self.render_timer = self.render_freq + self.dt

        # intialize sim
        self._create_sim()
        self._create_ground()
        self._create_veiwer()
        self._create_envs()
        self.gym.prepare_sim(
            self.sim)  # Prepares simulation with buffer allocations

        # metrics for tensorboard
        self.average_reward = np.zeros((self.num_sub_iters, self.num_envs), dtype=np.float32)
        self.average_q_value = np.zeros((self.num_sub_iters, self.num_envs), dtype=np.float32)
        self.average_posterior = np.zeros((self.num_sub_iters, self.num_envs), dtype=np.float32)

    def __del__(self):
        self.gym.destroy_sim(self.sim)
        if not self.headless:
            self.gym.destroy_viewer(self.viewer)

    def _create_sim(self):
        # configure sim
        # check `issacgy.gymapi.SimParams` for more options
        sim_params = gymapi.SimParams()
        sim_params.dt = self.dt

        if self.args.physics_engine == gymapi.SIM_FLEX:
        # if self.args.physics_engine == gymapi.SIM_FLEX and not self.args.use_gpu_pipeline:
            sim_params.substeps = 3
            sim_params.flex.solver_type = 5
            sim_params.flex.num_outer_iterations = 4
            sim_params.flex.num_inner_iterations = 20
            sim_params.flex.relaxation = 0.75
            sim_params.flex.warm_start = 0.8
            sim_params.flex.shape_collision_margin = 0.001
            sim_params.flex.friction_mode = 2

            # see deform
            sim_params.stress_visualization = True
            sim_params.stress_visualization_min = 0.0
            sim_params.stress_visualization_max = 1.e+5
        elif self.args.physics_engine == gymapi.SIM_PHYSX:
            sim_params.substeps = 5
            sim_params.physx.solver_type = 4
            sim_params.physx.num_position_iterations = 4
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.num_threads = self.args.num_threads
            sim_params.physx.use_gpu = self.args.use_gpu
            sim_params.physx.rest_offset = 0.0
        else:
            raise Exception("GPU pipeline is only available with PhysX")

        # set gpu pipeline
        sim_params.use_gpu_pipeline = self.args.use_gpu_pipeline
        if sim_params.use_gpu_pipeline:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        # set up axis as Z-up
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0, 0, -9.81)

        # create sim
        self.sim = self.gym.create_sim(
            self.args.compute_device_id,
            self.args.graphics_device_id,
            self.args.physics_engine,
            sim_params,
        )
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

    def _create_ground(self):
        # create ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
        plane_params.distance = 0
        plane_params.static_friction = 0.3
        plane_params.dynamic_friction = 0.15
        plane_params.restitution = 0

        # create the ground plane
        self.gym.add_ground(self.sim, plane_params)

    def _create_veiwer(self):
        if self.headless:
            self.viewer = None
        else:
            # create viewer
            self.viewer = self.gym.create_viewer(self.sim,
                                                 gymapi.CameraProperties())

            # set viewer camera pose
            cam_pos = gymapi.Vec3(4.5, 3.0, 2.0)
            cam_target = gymapi.Vec3(2.0, 0.5, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

            # key callback
            self.gym.subscribe_viewer_keyboard_event(self.viewer,
                                                     gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V,
                                                     "toggle_viewer_sync")

            if self.viewer is None:
                print("*** Failed to create viewer")
                quit()

    def _create_envs(self):
        asset_root = "./assets"

        ###################
        # LOAD HAND ASSET #
        ###################
        # load hand asset
        hand_asset_file = "urdf/franka_description/robots/franka_hand.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.thickness = 0.001
        asset_options.disable_gravity = True
        asset_options.density = 1000
        asset_options.override_inertia = True
        asset_options.flip_visual_attachments = True
        hand_asset = self.gym.load_asset(self.sim, asset_root, hand_asset_file, asset_options)

        # default hand pose
        hand_offset = 1
        hand_pose = gymapi.Transform()
        hand_pose.p = gymapi.Vec3(0, 0, hand_offset + 0.1122)
        hand_pose.r = gymapi.Quat(0, 1, 0, 0)

        # default hand dof state
        self.num_dofs = self.gym.get_asset_dof_count(hand_asset)
        default_dof_pos = np.array(
            [0.0, 0.0, hand_offset, 0.0, self.gripper_width / 2.0, self.gripper_width / 2.0],
            dtype=np.float32)
        default_dof_state = np.zeros(self.num_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] = default_dof_pos

        # configure dof properties
        hand_dof_props = self.gym.get_asset_dof_properties(hand_asset)
        hand_dof_props["driveMode"][:4].fill(gymapi.DOF_MODE_POS)
        hand_dof_props["stiffness"][:4].fill(1000.0)
        hand_dof_props["damping"][:4].fill(100.0)
        hand_dof_props["driveMode"][4:].fill(gymapi.DOF_MODE_POS)
        hand_dof_props["stiffness"][4:].fill(800.0)
        hand_dof_props["damping"][4:].fill(40.0)

        ##################
        # LOAD OBJ ASSET #
        ##################
        # load object asset
        object_asset_file = "urdf/{}/{}/{}.urdf".format(
            self.target_dataset_name,
            self.target_object_name,
            self.target_object_name)
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.fix_base_link = False
        asset_options.thickness = 0.001
        asset_options.override_inertia = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        if self.args.physics_engine == gymapi.SIM_PHYSX:
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params.resolution = 300000
            asset_options.vhacd_params.max_convex_hulls = 50
            asset_options.vhacd_params.max_num_vertices_per_ch = 1000
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, asset_options)

        # load object stable poses
        object_stable_prob = np.load(
            "assets/urdf/{}/{}/stable_prob.npy".format(
                self.target_dataset_name,
                self.target_object_name))
        object_stable_poses = np.load(
            "assets/urdf/{}/{}/stable_poses.npy".format(
                self.target_dataset_name,
                self.target_object_name))
        if self.sim_method == "multi_stable_pose":
            object_stable_prob = object_stable_prob[object_stable_prob > self.min_stable_pose_prob][:self.max_num_stable_pose]
            self.object_stable_prob = object_stable_prob / np.sum(object_stable_prob)
            object_stable_poses = object_stable_poses[:len(object_stable_prob)]
        elif self.sim_method == "single_stable_pose":
            object_stable_poses = [object_stable_poses[self.stable_pose_idx]] * self.max_num_stable_pose
            self.object_stable_prob = np.ones(len(object_stable_poses)) / self.max_num_stable_pose

        self.object_stable_poses = []
        for pose in object_stable_poses:
            # 4x4 transform matrix to gymapi.Transform
            t = gymapi.Transform()
            t.p = gymapi.Vec3(pose[0, 3], pose[1, 3], pose[2, 3])

            # 3x3 rotation matrix to quaternion
            r = R.from_matrix(pose[:3, :3])
            quat = r.as_quat()
            t.r = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])
            self.object_stable_poses.append(t)

        # load object mesh
        self.object_mesh = trimesh.load(
            "assets/urdf/{}/{}/{}.obj".format(
                self.target_dataset_name,
                self.target_object_name,
                self.target_object_name))

        ##############
        # CREATE ENV #
        ##############
        # configure env grid
        self.num_envs = len(self.object_stable_poses)
        num_per_row = int(np.ceil(np.sqrt(self.num_envs)))
        env_lower = gymapi.Vec3(-0.5, -0.5, 0.0)
        env_upper = gymapi.Vec3(0.5, 0.5, 2.0)
        print("Creating %d environments" % self.num_envs)
        for env_idx in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper,
                                      num_per_row)

            # add hand
            hand_handle = self.gym.create_actor(env, hand_asset, hand_pose, "hand_pose", env_idx, 1)

            # set hand properties
            self.gym.set_actor_dof_properties(env, hand_handle, hand_dof_props)
            self.gym.set_actor_dof_states(env, hand_handle, default_dof_state, gymapi.STATE_ALL)
            self.gym.set_actor_dof_position_targets(env, hand_handle, default_dof_pos)
            self.default_dof_pos.append(default_dof_pos)

            # add object
            object_pose = self.object_stable_poses[np.random.randint(len(self.object_stable_poses))]
            object_handle = self.gym.create_actor(env, object_asset, object_pose, "object", env_idx, 0)

            # set segment id
            self.gym.set_rigid_body_segmentation_id(env, object_handle, 0, 1)

            # add camera sensor
            camera_props = gymapi.CameraProperties()
            camera_props.width = int(self.camera_matrix[0, 2] * 2.0)
            camera_props.height = int(self.camera_matrix[1, 2] * 2.0)
            camera_props.horizontal_fov = 2 * np.arctan2(self.camera_matrix[0, 2], self.camera_matrix[0, 0]) * 180 / np.pi
            camera_props.far_plane = 1.0
            camera_handle = self.gym.create_camera_sensor(env, camera_props)

            # gym camera pos def: x=optical axis, y=left, z=down | convention: OpenGL
            cam_pose = gymapi.Transform()
            cam_pose.p = gymapi.Vec3(-0.7 * np.sin(0.075 * np.pi), 0.0, 0.7 * np.cos(0.075 * np.pi))
            cam_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.deg2rad(90 - 13.5))
            self.gym.set_camera_transform(
                camera_handle,
                env,
                cam_pose)

            # get camera extrinsic once
            if env_idx == 0:
                # convert z = x, x = -y, y = -z
                rot = R.from_quat([cam_pose.r.x, cam_pose.r.y, cam_pose.r.z, cam_pose.r.w]).as_matrix()
                rot_convert = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
                rot = np.dot(rot, rot_convert)
                self.camera_extr = np.eye(4)
                self.camera_extr[:3, :3] = rot
                self.camera_extr[:3, 3] = np.array([cam_pose.p.x, cam_pose.p.y, cam_pose.p.z])

            # append handles
            self.envs.append(env)
            self.hand_handles.append(hand_handle)
            self.object_handles.append(object_handle)
            self.camera_handles.append(camera_handle)

        # prepare tensor
        self._dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(self._dof_states)
        self.dof_pos = self.dof_states[:, 0].view(self.num_envs, self.num_dofs)
        self.dof_vel = self.dof_states[:, 1].view(self.num_envs, self.num_dofs)

        self.default_dof_pos = np.array(self.default_dof_pos, dtype=np.float32)
        self.default_dof_pos = torch.from_numpy(self.default_dof_pos).to(self.device)
        self.default_dof_vel = torch.zeros_like(self.default_dof_pos)
        self.target_dof_pos = torch.clone(self.default_dof_pos)

    def render(self):
        """Draw the frame to the viewer, and check for keyboard events."""
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # step graphics
            if self.enable_viewer_sync:
                if self.render_timer > self.render_freq:
                    self.gym.step_graphics(self.sim)
                    self.gym.draw_viewer(self.viewer, self.sim, True)
                    # self.gym.sync_frame_time(self.sim)
                    self.render_timer = 0
                else:
                    self.gym.step_graphics(self.sim)
                    self.gym.poll_viewer_events(self.viewer)
                    self.render_timer += self.dt
            else:
                self.gym.step_graphics(self.sim)
                self.gym.poll_viewer_events(self.viewer)

        # step graphics for offscreen rendering
        if self.headless:
            self.gym.step_graphics(self.sim)

    def reset_env(self):
        # reset hand
        self.target_dof_pos[:] = self.default_dof_pos[:]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.target_dof_pos))

        self.dof_pos[:] = self.default_dof_pos[:]
        self.dof_vel[:] = self.default_dof_vel[:]
        self.gym.set_dof_state_tensor(self.sim, self._dof_states)

        # reset objects to random stable poses
        reset_object_poses = []
        for env_idx in range(self.num_envs):
            # get rigid body handle
            rigid_body_handle = self.gym.get_actor_rigid_body_handle(
                self.envs[env_idx], self.object_handles[env_idx], 0)

            # get stable pose
            t_stable = self.object_stable_poses[env_idx]
            p_stable = t_stable.p
            q_stable = t_stable.r

            # get random pose
            p_random = gymapi.Vec3(
                np.random.uniform(-self.object_rand_pose_range, self.object_rand_pose_range),
                np.random.uniform(-self.object_rand_pose_range, self.object_rand_pose_range),
                0.0021,)

            q_random = R.from_euler('z', np.random.uniform(0, 2*np.pi), degrees=False).as_quat()
            q_random = gymapi.Quat(q_random[0], q_random[1], q_random[2], q_random[3])

            # get random stable pose
            p = q_random.rotate(p_stable) + p_random
            q = (q_random * q_stable).normalize()
            t = gymapi.Transform(p, q)

            # set random stable pose and zero vel
            self.gym.set_rigid_transform(self.envs[env_idx], rigid_body_handle, t)
            self.gym.set_rigid_linear_velocity(self.envs[env_idx], rigid_body_handle, gymapi.Vec3(0, 0, 0))
            self.gym.set_rigid_angular_velocity(self.envs[env_idx], rigid_body_handle, gymapi.Vec3(0, 0, 0))

            # disable gravity
            obj_props = self.gym.get_actor_rigid_body_properties(self.envs[env_idx], self.object_handles[env_idx])
            obj_props[0].flags = gymapi.RIGID_BODY_DISABLE_GRAVITY
            self.gym.set_actor_rigid_body_properties(self.envs[env_idx], self.object_handles[env_idx], obj_props, False)
            reset_object_poses.append(t)

        # delete all existing lines
        if self.viewer:
            self.gym.clear_lines(self.viewer)

        return reset_object_poses

    def step(self, main_step=0, sub_step=0):
        # reset evn
        object_poses = self.reset_env()
        self.wait(0.01)

        # get depth image
        depth_images, segmasks = self.get_camera_image()

        # run policy
        action = self.policy.action(depth_images, segmasks)
        grasps_2d = action[0]
        q_values = np.array(action[1], dtype=np.float32)
        im_tensors = np.array(action[2], dtype=np.float32)
        depth_tensors = np.array(action[3], dtype=np.float32)
        features = np.array(action[4], dtype=np.float32)
        posteriors = np.array(action[5], dtype=np.float32)

        # visualize planned grasps
        self.visualize_grasp_on_simulator(grasps_2d)

        # execute grasp
        if (self.args.physics_engine == gymapi.SIM_PHYSX) &\
                (self.grasp_metric_cfg['type'] == 'physx'):
            rewards = self.execute_grasp_physx(grasps_2d)
        elif (self.args.physics_engine == gymapi.SIM_FLEX) &\
                (self.grasp_metric_cfg['type'] == 'flex'):
            rewards = self.execute_grasp_flex(grasps_2d)
        elif self.grasp_metric_cfg['type'] == 'analytic':
            rewards = self.execute_grasp_analytic(grasps_2d, object_poses)
        else:
            raise NotImplementedError

        # TODO: update policy
        # update policy per main step
        if sub_step == self.num_sub_iters - 1:
            # update likelihood
            if isinstance(self.policy, PosteriorGraspingPolicy) or \
               isinstance(self.policy, CrossEntorpyPosteriorGraspingPolicy):
                # get valid indices
                valid_indices = []
                for env_idx in range(self.num_envs):
                    if grasps_2d[env_idx] is not None:
                        valid_indices.append(env_idx)
                valid_indices = np.array(valid_indices, dtype=np.int32)

                # get update based on stable pose probability
                num_samples = min(self.num_update_features_per_step, len(valid_indices))
                prob = self.object_stable_prob[valid_indices]
                prob /= np.sum(prob)
                update_idx = np.random.choice(
                    valid_indices,
                    size=num_samples,
                    p=prob,
                    replace=False)
                self.policy.update(features[update_idx], rewards[update_idx])

                # save update history
                if self.save_results:
                    np.save(os.path.join(self.save_dir, 'data', 'updated_feature_{:03d}_{:02d}.npy'.format(main_step, sub_step)), features[update_idx])
                    np.save(os.path.join(self.save_dir, 'data', 'updated_reward_{:03d}_{:02d}.npy'.format(main_step, sub_step)), rewards[update_idx])

            # update for q-learning policy
            if isinstance(self.policy, QLearningGraspingPolicy):
                self.policy.update(im_tensors, depth_tensors, rewards)

        # print result
        self.average_reward[sub_step, :] = rewards
        self.average_q_value[sub_step, :] = q_values
        self.average_posterior[sub_step, :] = posteriors
        if sub_step == self.num_sub_iters - 1:
            average_reward = np.mean(self.average_reward, axis=0)
            average_reward = np.dot(average_reward, self.object_stable_prob)
            average_q_value = np.mean(self.average_q_value, axis=0)
            average_q_value = np.dot(average_q_value, self.object_stable_prob)
            average_posterior = np.mean(self.average_posterior, axis=0)
            average_posterior = np.dot(average_posterior, self.object_stable_prob)

            metrics = {
                'average_reward': average_reward,
                'average_q_value': average_q_value,
                'average_posterior': average_posterior}

            self.writer.add_scalars('reward', metrics, main_step)
            self.writer.flush()
            print('main_step: {:03d}, E[r]: {:.3f}, E[q]: {:.3f}, E[p]: {:.3f}'.format(
                main_step, average_reward, average_q_value, average_posterior))

        # save data
        if self.save_results:
            np.save(os.path.join(self.save_dir, 'data', 'q_values_{:03d}_{:02d}.npy'.format(main_step, sub_step)), q_values)
            np.save(os.path.join(self.save_dir, 'data', 'im_tensors_{:03d}_{:02d}.npy'.format(main_step, sub_step)), im_tensors)
            np.save(os.path.join(self.save_dir, 'data', 'depth_tensors_{:03d}_{:02d}.npy'.format(main_step, sub_step)), depth_tensors)
            np.save(os.path.join(self.save_dir, 'data', 'features_{:03d}_{:02d}.npy'.format(main_step, sub_step)), features)
            np.save(os.path.join(self.save_dir, 'data', 'rewards_{:03d}_{:02d}.npy'.format(main_step, sub_step)), rewards)
            np.save(os.path.join(self.save_dir, 'data', 'grasp_contact_points_{:03d}_{:02d}.npy'.format(main_step, sub_step)), grasp_contact_points)
            np.save(os.path.join(self.save_dir, 'data', 'posteriors_{:03d}_{:02d}.npy'.format(main_step, sub_step)), posteriors)
            if isinstance(self.policy, QLearningGraspingPolicy):
                if main_step == 99:
                    self.policy.save(os.path.join(self.save_dir, 'model_{:03d}_{:02d}.pt'.format(main_step, sub_step)))
        return None

    ############################################
    # robot, sensor and environment interfaces #
    ############################################
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
        while not self.wait_timer > sleep_time:
            # step the physics
            self.gym.simulate(self.sim)
            # refresh results
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            # step rendering
            self.render()
            # update timer
            self.wait_timer += self.dt

        self.wait_timer = 0.0
        return True

    def grasps_to_joint_positions(self, grasps):
        """Convert antipodal grasp to (x, y, z, theta) joint positions.

        Args:
            grasps (list of AntipodalGrasp): Antipodal grasps.

        Returns:
            joint_poistions (numpy.ndarray): (x, y, z, theta) of target joint positions. (N, 4).
        """
        joint_positions = []
        for grasp in grasps:
            # if grasp is not None
            if isinstance(grasp, AntipodalGrasp):
                grasp_pose_camera = grasp.get_3d_grasp_pose(
                    np.array([0, 0, -1], dtype=np.float32),
                    self.camera_matrix,
                    self.camera_extr)
                grasp_pose_world = self.camera_extr @ grasp_pose_camera
                grasp_theta = -np.arctan2(grasp_pose_world[1, 0], grasp_pose_world[0, 0])
                if grasp_theta > (np.pi / 2):
                    grasp_theta -= np.pi
                elif grasp_theta < (-np.pi / 2):
                    grasp_theta += np.pi
                grasp_pose = np.array([grasp_pose_world[0, 3], grasp_pose_world[1, 3], grasp_pose_world[2, 3], grasp_theta], dtype=np.float32)
                grasp_pose[2] += self.grasp_pose_z_offset
            elif grasp is None:
                grasp_pose = np.array([0, 0, 0.5, 0], dtype=np.float32)
            joint_positions.append(grasp_pose)
        joint_positions = np.array(joint_positions, dtype=np.float32)
        joint_positions = torch.from_numpy(joint_positions).to(self.device)
        return joint_positions

    def visualize_grasp_on_simulator(self, grasps, width=0.1, color=(1.0, 0.0, 0.0)):
        """Visualize grasps in the environment

        Args:
            grasps (list of AntipodalGrasp): Antipodal grasps.
            width (float, optional): width of the grasp. Defaults to 0.1.
            color (tuple, optional): color of the grasp. Defaults to (1.0, 0.0, 0.0).
        """
        # get grasp pose in (x, y, z, theta)
        grasp_poses = self.grasps_to_joint_positions(grasps).cpu().numpy()
        color = gymapi.Vec3(color[0], color[1], color[2])

        # get grasp pose in object frame
        for env_idx in range(self.num_envs):
            center = grasp_poses[env_idx][:3]
            theta = grasp_poses[env_idx][3]

            # get end points of the grasp
            end_point1 = np.array(
                [width * np.sin(theta), width * np.cos(theta), 0.0], dtype=np.float32) + center
            end_point2 = np.array(
                [-width * np.sin(theta), -width * np.cos(theta), 0.0], dtype=np.float32) + center

            # convert to Vec3
            end_point1 = gymapi.Vec3(end_point1[0], end_point1[1], end_point1[2])
            end_point2 = gymapi.Vec3(end_point2[0], end_point2[1], end_point2[2])

            # draw grasps in gym
            if self.viewer:
                gymutil.draw_line(
                    end_point1, end_point2, color,
                    self.gym, self.viewer, self.envs[env_idx])

        # wait dt to update scene
        self.wait(self.dt)
        # for i in range(20*5):
        #     self.wait(self.dt)
        #     time.sleep(0.05)

    def execute_grasp_physx(self, grasps):
        """Execute grasp in simulation

        Args:
            grasps (list of AntipodalGrasp): Antipodal grasps.

        Returns:
            success (np.ndarray): 1.0 if successful, 0.0 otherwise. (N, )
        """
        # get grasp pose in (x, y, z, theta)
        target_joint_positions = self.grasps_to_joint_positions(grasps)

        # 1) execute x, y, theta movement
        xyt_idx = [0, 1, 3]
        self.target_dof_pos[:, xyt_idx] = target_joint_positions[:, xyt_idx]
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.target_dof_pos))
        self.wait(5.0)

        # 2) execute z movement
        self.target_dof_pos[:, 2] = target_joint_positions[:, 2]
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.target_dof_pos))
        self.wait(5.0)

        # 3) execute grasp
        self.target_dof_pos[:, 4:] = 0.0
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.target_dof_pos))
        self.wait(5.0)

        # 4) enable gravity on objects and execute lift
        self.target_dof_pos[:, 2] = 0.5
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.target_dof_pos))
        # enable gravity
        for env_idx in range(self.num_envs):
            obj_props = self.gym.get_actor_rigid_body_properties(self.envs[env_idx], self.object_handles[env_idx])
            obj_props[0].flags = gymapi.RIGID_BODY_NONE
            self.gym.set_actor_rigid_body_properties(self.envs[env_idx], self.object_handles[env_idx], obj_props, False)
        self.wait(5.0)

        # 5) check if grasp is successful
        success = np.zeros(self.num_envs, dtype=np.float32)
        for env_idx in range(self.num_envs):
            obj_height = self.gym.get_actor_rigid_body_states(
                self.envs[env_idx],
                self.object_handles[env_idx],
                gymapi.STATE_POS)[0]['pose']['p']['z']
            if obj_height > 0.25:
                success[env_idx] = 1.0
        return success

    def execute_grasp_flex(self, grasps):
        """Execute grasp in simulation with Flex engine.

        Args:
            grasps (list of AntipodalGrasp): Antipodal grasps.

        Returns:
            success (np.ndarray): 1.0 if successful, 0.0 otherwise. (N, )
        """
        # get grasp pose in (x, y, z, theta)
        target_joint_positions = self.grasps_to_joint_positions(grasps)

        # 1) execute x, y, theta movement
        xyt_idx = [0, 1, 3]
        self.target_dof_pos[:, xyt_idx] = target_joint_positions[:, xyt_idx]
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.target_dof_pos))
        self.wait(0.5)

        # 2) execute z movement
        self.target_dof_pos[:, 2] = target_joint_positions[:, 2]
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.target_dof_pos))
        self.wait(0.5)

        # 3) execute grasp
        self.target_dof_pos[:, 4:] = 0.0
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.target_dof_pos))
        self.wait(0.5)

        # 4) enable gravity on objects and execute lift
        self.target_dof_pos[:, 2] = 0.5
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.target_dof_pos))
        # enable gravity
        for env_idx in range(self.num_envs):
            obj_props = self.gym.get_actor_rigid_body_properties(
                self.envs[env_idx], self.object_handles[env_idx])
            obj_props[0].flags = gymapi.RIGID_BODY_NONE
            self.gym.set_actor_rigid_body_properties(
                self.envs[env_idx], self.object_handles[env_idx], obj_props, False)
        self.wait(2.0)

        # 5) check if grasp is successful
        success = np.zeros(self.num_envs, dtype=np.float32)
        for env_idx in range(self.num_envs):
            obj_height = self.gym.get_actor_rigid_body_states(
                self.envs[env_idx],
                self.object_handles[env_idx],
                gymapi.STATE_POS)[0]['pose']['p']['z']
            if obj_height > 0.25:
                success[env_idx] = 1.0
        return success

    def execute_grasp_analytic(self, grasps, object_poses):
        """Execute grasp in simulation with analytic grasp metric.

        Args:
            grasps (list of AntipodalGrasp): Antipodal grasps.
            object_poses (list of gymapi.Transform): Object poses.

        Returns:
            success (np.ndarray): 1.0 if successful, 0.0 otherwise. (N, )
        """
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        for i in range(self.num_envs):
            # get grasp pose in world frame
            grasp_pose_camera = grasps[i].get_3d_grasp_pose(
                    np.array([0, 0, -1], dtype=np.float32),
                    self.camera_matrix,
                    self.camera_extr)
            # rotate on local z axis 90 degrees
            rot = np.array([[0, -1, 0],
                            [1, 0, 0],
                            [0, 0, 1]], dtype=np.float32)
            grasp_pose_camera[:3, :3] = rot @ grasp_pose_camera[:3, :3]
            grasp_pose_world = self.camera_extr @ grasp_pose_camera

            # get object pose
            object_world = self.transform_to_numpy(object_poses[i])

            # get grasp pose in object frame
            grasp_pose_object = np.linalg.inv(object_world) @ grasp_pose_world

            # get grasp metric
            grasp_metric = ParallelJawGraspMetric.from_mesh(
                self.object_mesh, grasp_pose_object, friction_coef=0.5)
            print('grasp_quality: {}'.format(grasp_metric.quality()))
            if grasp_metric.quality() > 0.002:
                rewards[i] = 1.0
            else:
                rewards[i] = 0.0
        return rewards

    def visualize_camera_axis(self):
        """Visualize camera axis"""
        # draw camera pose with line
        camera_pose = self.camera_extr

        # 4x4 matrix to gymapi.Transform
        rot = camera_pose[:3, :3]
        quat = R.from_matrix(rot).as_quat()
        pose = gymapi.Transform()
        pose.r = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])
        pose.p = gymapi.Vec3(camera_pose[0, 3], camera_pose[1, 3], camera_pose[2, 3])

        # draw line
        axes_geom = gymutil.AxesGeometry(1.0)
        for env_idx in range(self.num_envs):
            gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[env_idx], pose)
        self.wait(self.dt)

    def get_camera_image(self):
        """Get images from camera

        Returns:
            depth_images (numpy.ndarray): image of shape (num_envs, H, W, 3)
            segmasks (numpy.ndarray): segmentation mask of shape (num_envs, H, W)
        """
        depth_images = []
        segmasks = []
        self.gym.render_all_camera_sensors(self.sim)
        for i in range(self.num_envs):
            depth_image = self.gym.get_camera_image(self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_DEPTH)
            segmask = self.gym.get_camera_image(self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_SEGMENTATION)
            depth_images.append(depth_image)
            segmasks.append(segmask)
        depth_images = np.array(depth_images) * -1
        segmasks = np.array(segmasks)

        return depth_images, segmasks

    #########
    # utils #
    #########
    @staticmethod
    def transform_to_numpy(transform):
        """Convert gymapi.Transform to numpy array

        Args:
            transform (gymapi.Transform): transform to convert

        Returns:
            numpy.ndarray: transform in numpy array
        """
        # convert to numpy array
        p = np.array([transform.p.x, transform.p.y, transform.p.z])
        q = np.array([transform.r.x, transform.r.y, transform.r.z, transform.r.w])

        # convert to homogeneous transform
        rot = R.from_quat(q).as_matrix()
        t = np.eye(4)
        t[:3, :3] = rot
        t[:3, 3] = p
        return t


if __name__ == '__main__':
    env = SigulatePickingEnv()

    for i in range(env.num_main_iters):
        for j in range(env.num_sub_iters):
            start_time = time.time()
            env.step(i, j)
            # if i == 0:
            #     env.reset_env()
            # env.wait(2.0)
            print('[{}][{}] step time: {:.3f}'.format(i, j, time.time() - start_time))
