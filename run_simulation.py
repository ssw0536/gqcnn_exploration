# import python modules
import os
import time
import yaml
from datetime import datetime

# import isaacgym modules
from isaacgym import gymapi, gymutil
from isaacgym import gymtorch

# import 3rd party modules
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


from src.gqcnn import GQCNN
from src.grasp_sampler import AntipodalGrasp, AntipodalGraspSampler
from src.policy import (CrossEntropyGraspingPolicy,
                        PosteriorGraspingPolicy,
                        CrossEntorpyPosteriorGraspingPolicy)


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
                    "type": bool,
                    "default": False,
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

        # load gqcnn, sampler, and policy
        gqcnn = GQCNN()
        sampler = AntipodalGraspSampler(sampling_cfg)
        if policy_cfg["type"] == "posterior":
            self.policy = PosteriorGraspingPolicy(
                gqcnn, sampler, policy_cfg)
        elif policy_cfg["type"] == "cross_entropy":
            self.policy = CrossEntropyGraspingPolicy(
                gqcnn, sampler, policy_cfg)
        else:
            policy_cfg["type"] = "cross_entropy_posterior"
            self.policy = CrossEntorpyPosteriorGraspingPolicy(
                gqcnn, sampler, policy_cfg)

        # set up simulation
        self.dt = sim_cfg["dt"]
        self.num_main_iters = sim_cfg["num_main_iters"]
        self.num_sub_iters = sim_cfg["num_sub_iters"]
        self.target_object_name = sim_cfg["target_object"]
        self.object_rand_pose_range = sim_cfg["object_random_pose_range"]
        self.max_num_stable_pose = sim_cfg["max_num_stable_pose"]
        cx = sim_cfg["camera"]["phoxi"]["cx"]
        cy = sim_cfg["camera"]["phoxi"]["cy"]
        fx = sim_cfg["camera"]["phoxi"]["fx"]
        fy = sim_cfg["camera"]["phoxi"]["fy"]
        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]])

        # configure save directory
        if self.save_results:
            root_dir = '/media/sungwon/WorkSpace/projects/gqcnn_thomphson_sampling'
            self.save_dir = os.path.join(root_dir, self.target_object_name, policy_cfg["type"])
            os.makedirs(self.save_dir, exist_ok=True)

            # save config files
            save_cfg_file = os.path.join(root_dir, self.target_object_name) + "/{}_config.yaml".format(policy_cfg["type"])
            with open(os.path.join(save_cfg_file), "w") as f:
                yaml.safe_dump(cfg, f)

        # custom env variables
        self.envs = []
        self.hand_handles = []
        self.camera_handles = []
        self.object_handles = []
        self.default_dof_pos = []

        self.enable_viewer_sync = True
        self.task_status = None
        self._frame_count = 0
        self.wait_timer = 0.0

        # intialize sim
        self._create_sim()
        self._create_ground()
        self._create_veiwer()
        self._create_envs()
        self.gym.prepare_sim(
            self.sim)  # Prepares simulation with buffer allocations

        # prepare some tensors
        self.average_reward = np.zeros((self.num_sub_iters, self.num_envs), dtype=np.float32)

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
            sim_params.substeps = 2
            sim_params.flex.solver_type = 5
            sim_params.flex.num_outer_iterations = 5
            sim_params.flex.num_inner_iterations = 10
            sim_params.flex.relaxation = 0.75
            sim_params.flex.warm_start = 0.8
            sim_params.flex.shape_collision_margin = 0.001
            sim_params.flex.friction_mode = 2

            # see deform
            sim_params.stress_visualization = True
            sim_params.stress_visualization_min = 0.0
            sim_params.stress_visualization_max = 1.e+5
        elif self.args.physics_engine == gymapi.SIM_PHYSX:
            sim_params.substeps = 1
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
        hand_pose.p = gymapi.Vec3(0, 0, hand_offset+0.1122)
        hand_pose.r = gymapi.Quat(0, 1, 0, 0)

        # default hand dof state
        self.num_dofs = self.gym.get_asset_dof_count(hand_asset)
        default_dof_pos = np.array([0.0, 0.0, hand_offset, 0.0, 0.04, 0.04], dtype=np.float32)
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
        object_asset_file = "urdf/egad_eval_set_urdf/{}/{}.urdf".format(
            self.target_object_name,
            self.target_object_name)
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.fix_base_link = False
        asset_options.thickness = 0.001
        asset_options.override_inertia = True
        # asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        if self.args.physics_engine == gymapi.SIM_PHYSX:
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params.resolution = 300000
            asset_options.vhacd_params.max_convex_hulls = 50
            asset_options.vhacd_params.max_num_vertices_per_ch = 1000
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, asset_options)

        # load object stable poses
        object_stable_prob = np.load(
            "assets/urdf/egad_eval_set_urdf/{}/stable_prob.npy".format(
                self.target_object_name))[:self.max_num_stable_pose]
        self.object_stable_prob = object_stable_prob / np.sum(object_stable_prob)
        object_stable_poses = np.load(
            "assets/urdf/egad_eval_set_urdf/{}/stable_poses.npy".format(
                self.target_object_name))[:self.max_num_stable_pose]
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
            camera_props.horizontal_fov = 2*np.arctan2(self.camera_matrix[0, 2], self.camera_matrix[0, 0])*180/np.pi
            camera_props.far_plane = 1.0
            camera_handle = self.gym.create_camera_sensor(env, camera_props)

            # gym camera pos def: x=optical axis, y=left, z=down | convention: OpenGL
            cam_pose = gymapi.Transform()
            cam_pose.p = gymapi.Vec3(-0.7*np.sin(0.075*np.pi), 0.0, 0.7*np.cos(0.075*np.pi))
            cam_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.deg2rad(90-13.5))
            self.gym.set_camera_transform(
                camera_handle,
                env,
                cam_pose,
                )

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
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                self.gym.sync_frame_time(self.sim)
            else:
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
            q = (q_random*q_stable).normalize()
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

        # sample grasp
        action = self.policy.action(depth_images, segmasks)
        grasps = action[0]
        q_values = np.array(action[1], dtype=np.float32)
        im_tensors = np.array(action[2], dtype=np.float32)
        depth_tensors = np.array(action[3], dtype=np.float32)
        features = np.array(action[4], dtype=np.float32)

        # convert antipodal grasp to 3d grasp
        grasp_poses = self.convert_to_grasp_pose(grasps)

        # visualize grasps
        grasp_contact_points = self.get_grasp_pose_on_object(grasp_poses, object_poses)

        # execute grasp
        self.execute_grasp(grasp_poses)

        # evaluate grasp
        rewards = self.evaluate_grasp()

        # update likelihood per main step
        if sub_step == 0:
            # update likelihood
            if isinstance(self.policy, PosteriorGraspingPolicy):
                # get valid indices
                valid_indices = []
                for env_idx in range(self.num_envs):
                    if grasps[env_idx] is not None:
                        valid_indices.append(env_idx)
                valid_indices = np.array(valid_indices, dtype=np.int32)

                # get update based on stable pose probability
                prob = self.object_stable_prob[valid_indices]
                prob /= np.sum(prob)
                update_idx = np.random.choice(
                    valid_indices,
                    size=1,
                    p=prob)
                self.policy.update(features[update_idx], rewards[update_idx])

                # save update history
                if self.save_results:
                    np.save(os.path.join(self.save_dir, 'updated_feature_{:03d}_{:02d}.npy'.format(main_step, sub_step)), features[update_idx])
                    np.save(os.path.join(self.save_dir, 'updated_reward_{:03d}_{:02d}.npy'.format(main_step, sub_step)), rewards[update_idx])
            elif isinstance(self.policy, CrossEntorpyPosteriorGraspingPolicy):
                pass

        # print result
        self.average_reward[sub_step, :] = rewards
        if sub_step == self.num_sub_iters - 1:
            average_reward = np.mean(self.average_reward, axis=0)
            average_reward = np.dot(average_reward, self.object_stable_prob)
            print('main_step: {:03d}, average_reward: {:.3f}'.format(main_step, average_reward))

        # save data
        if self.save_results:
            np.save(os.path.join(self.save_dir, 'q_values_{:03d}_{:02d}.npy'.format(main_step, sub_step)), q_values)
            np.save(os.path.join(self.save_dir, 'im_tensors_{:03d}_{:02d}.npy'.format(main_step, sub_step)), im_tensors)
            np.save(os.path.join(self.save_dir, 'depth_tensors_{:03d}_{:02d}.npy'.format(main_step, sub_step)), depth_tensors)
            np.save(os.path.join(self.save_dir, 'features_{:03d}_{:02d}.npy'.format(main_step, sub_step)), features)
            np.save(os.path.join(self.save_dir, 'rewards_{:03d}_{:02d}.npy'.format(main_step, sub_step)), rewards)
            np.save(os.path.join(self.save_dir, 'grasp_contact_points_{:03d}_{:02d}.npy'.format(main_step, sub_step)), grasp_contact_points)
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

    def convert_to_grasp_pose(self, grasps):
        """Convert antipodal grasp to (x, y, z, theta) grasp pose in world frame.

        Args:
            grasps (list of AntipodalGrasp): Antipodal grasps.

        Returns:
            grasp_poses (numpy.ndarray): (x, y, z, theta) of grasp. (N, 4)
        """
        grasp_poses = []
        for grasp in grasps:
            if isinstance(grasp, AntipodalGrasp):
                grasp_pose_camera = grasp.get_3d_grasp_pose(
                    np.array([0, 0, -1], dtype=np.float32),
                    self.camera_matrix,
                    self.camera_extr)
                grasp_pose_world = self.camera_extr @ grasp_pose_camera
                grasp_theta = -np.arctan2(grasp_pose_world[1, 0], grasp_pose_world[0, 0])
                if grasp_theta > np.pi/2:
                    grasp_theta -= np.pi
                elif grasp_theta < -np.pi/2:
                    grasp_theta += np.pi
                grasp_pose = np.array([grasp_pose_world[0, 3], grasp_pose_world[1, 3], grasp_pose_world[2, 3], grasp_theta], dtype=np.float32)
            else:
                grasp_pose = np.array([0, 0, 0.5, 0], dtype=np.float32)
            grasp_poses.append(grasp_pose)
        grasp_poses = np.array(grasp_poses, dtype=np.float32)
        grasp_poses = torch.from_numpy(grasp_poses).to(self.device)
        return grasp_poses

    def get_grasp_pose_on_object(self, grasp_poses, object_poses, visualize=True):
        """Visualize grasps in the environment

        Args:
            grasp_poses (numpy.ndarray): (x, y, z, theta) grasp poses in the world frame. (num_envs, 4)
            object_poses (numpy.ndarray): object poses in the world frame. (num_envs, 4, 4)

        Returns:
            grasp_contact_points (numpy.ndarray): contact points of the grasp in the world frame. (num_envs, 2, 3)
        """
        # get grasp pose in object frame
        grasp_cp1 = []
        grasp_cp2 = []
        for env_idx in range(self.num_envs):
            center = grasp_poses[env_idx][:3]
            theta = grasp_poses[env_idx][3]

            l = 0.1
            cp1_offset = np.array([l*np.sin(theta), l*np.cos(theta), 0.0], dtype=np.float32)
            cp2_offset = -cp1_offset

            cp1 = center + cp1_offset
            cp1 = gymapi.Vec3(cp1[0], cp1[1], cp1[2])
            cp2 = center + cp2_offset
            cp2 = gymapi.Vec3(cp2[0], cp2[1], cp2[2])
            color = gymapi.Vec3(1.0, 0.0, 0.0)

            # draw grasps in gym
            if self.viewer:
                if visualize:
                    gymutil.draw_line(
                        cp1, cp2, color,
                        self.gym, self.viewer, self.envs[env_idx])

            # convert to object frame
            t_wo = object_poses[env_idx]
            cp1 = t_wo.inverse().transform_point(cp1)
            cp2 = t_wo.inverse().transform_point(cp2)
            grasp_cp1.append(np.array([cp1.x, cp1.y, cp1.z], dtype=np.float32))
            grasp_cp2.append(np.array([cp2.x, cp2.y, cp2.z], dtype=np.float32))
        grasp_cp1 = np.array(grasp_cp1, dtype=np.float32)
        grasp_cp2 = np.array(grasp_cp2, dtype=np.float32)
        grasp_contact_points = np.stack([grasp_cp1, grasp_cp2], axis=1)
        self.wait(self.dt)
        return grasp_contact_points

    def execute_grasp(self, grasp_poses):
        """Execute grasp in simulation

        Args:
            grasp_poses (numpy.ndarray): grasp poses in world frame. (num_envs, (x, y, z, theta))
        """
        # 1) execute x, y, theta movement
        xyt_idx = [0, 1, 3]
        self.target_dof_pos[:, xyt_idx] = grasp_poses[:, xyt_idx]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.target_dof_pos))
        self.wait(0.5)

        # 2) execute z movement
        self.target_dof_pos[:, 2] = grasp_poses[:, 2]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.target_dof_pos))
        self.wait(0.5)

        # 3) execute grasp
        self.target_dof_pos[:, 4:] = 0.0
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.target_dof_pos))
        self.wait(0.5)

        # 4) enable gravity on objects and execute lift
        self.target_dof_pos[:, 2] = 0.5
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.target_dof_pos))
        # enable gravity
        for env_idx in range(self.num_envs):
            obj_props = self.gym.get_actor_rigid_body_properties(self.envs[env_idx], self.object_handles[env_idx])
            obj_props[0].flags = gymapi.RIGID_BODY_NONE
            self.gym.set_actor_rigid_body_properties(self.envs[env_idx], self.object_handles[env_idx], obj_props, False)
        self.wait(1.0)

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

    def evaluate_grasp(self):
        reward = np.zeros(self.num_envs, dtype=np.float32)
        for env_idx in range(self.num_envs):
            obj_height = self.gym.get_actor_rigid_body_states(
                self.envs[env_idx],
                self.object_handles[env_idx],
                gymapi.STATE_POS)[0]['pose']['p']['z']
            if obj_height > 0.25:
                reward[env_idx] = 1.0
        return reward


if __name__ == '__main__':
    env = SigulatePickingEnv()

    for i in range(env.num_main_iters):
        for j in range(env.num_sub_iters):
            start_time = time.time()
            env.step(i, j)
            print('[{}][{}] step time: {:.3f}'.format(i, j, time.time() - start_time))
