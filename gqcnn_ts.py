# import python modules
import os
import time
from datetime import datetime

# import isaacgym modules
from isaacgym import gymapi, gymutil
from isaacgym import gymtorch

# import 3rd party modules
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# import user modules
from grasp_planner import ThompsonSamplingGraspPlanner
from grasp_planner import AntipodalGrasp


class SigulatePickingEnv(object):
    def __init__(self):
        # initialize the gym
        self.gym = gymapi.acquire_gym()

        # parse arguments / isaacgym configuration
        self.args = gymutil.parse_arguments(
            description='GQ-CNN with TS',
            headless=True,
            no_graphics=True,
            custom_parameters=[
                {
                    "name": "--num_envs",
                    "type": int,
                    "default": 16,
                    "help": "Number of environments to create"
                },
                {
                    "name": "--save_results",
                    "type": bool,
                    "default": False,
                    "help": "Save the results"
                },
            ],
        )
        self.num_envs = self.args.num_envs
        self.headless = self.args.headless
        self.save_results = self.args.save_results
        self.num_per_row = 4
        self.dt = 1.0 / 100.0

        # grasp simulation configuration
        self.target_object_name = "A2"
        self.object_rand_pose_range = 0.1
        self.camera_matrix = np.array([525, 0, 320, 0, 525, 240, 0, 0, 1]).reshape(3, 3)  # primsense 1.08
        # self.camera_matrix = np.array([552.5, 0, 258, 0, 552.5, 193, 0, 0, 1]).reshape(3, 3)  # phoxi s

        # save directory
        if self.save_results:
            root_dir = '/media/sungwon/WorkSpace/projects/gqcnn_thomphson_sampling'
            cur_date = datetime.today().strftime("%Y-%m-%d-%H%M")
            self.save_dir = os.path.join(root_dir, cur_date)
            os.makedirs(self.save_dir, exist_ok=True)

            # make a directory each data type
            os.makedirs(os.path.join(self.save_dir, 'success_features'), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, 'failure_features'), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, 'success_q_values'), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, 'failure_q_values'), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, 'grasp_cp1'), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, 'grasp_cp2'), exist_ok=True)

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

        # initialize grasp planner
        self.grasp_planner = ThompsonSamplingGraspPlanner()

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
        object_stable_poses = np.load("assets/urdf/egad_eval_set_urdf/{}/stable_poses.npy".format(self.target_object_name))[2:3]
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
            break

        ##############
        # CREATE ENV #
        ##############
        # configure env grid
        print("Creating %d environments" % self.num_envs)
        self.num_per_row = 4
        env_lower = gymapi.Vec3(-0.5, -0.5, 0.0)
        env_upper = gymapi.Vec3(0.5, 0.5, 2.0)
        for env_idx in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper,
                                      self.num_per_row)

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
            # 2.0 : primsense 1.08
            # 4.0 : phoxi s
            camera_props = gymapi.CameraProperties()
            camera_props.width = int(self.camera_matrix[0, 2] * 2.0)
            camera_props.height = int(self.camera_matrix[1, 2] * 2.0)
            camera_props.horizontal_fov = 2*np.arctan2(self.camera_matrix[0, 2], self.camera_matrix[0, 0])*180/np.pi
            camera_props.far_plane = 1.0
            camera_handle = self.gym.create_camera_sensor(env, camera_props)

            #! gym camera pos def: x=optical axis, y=left, z=down | convention: OpenGL
            cam_pose = gymapi.Transform()
            cam_pose.p = gymapi.Vec3(-0.7*np.sin(0.075*np.pi), 0.0, 0.7*np.cos(0.075*np.pi))
            cam_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.deg2rad(90-13.5))
            self.gym.set_camera_transform(
                camera_handle,
                env,
                cam_pose,
                )

            # convert z = x, x = -y, y = -z
            rot = R.from_quat([cam_pose.r.x, cam_pose.r.y, cam_pose.r.z, cam_pose.r.w]).as_matrix()
            rot_convert = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
            rot = np.dot(rot, rot_convert)
            self.camera_extr = np.eye(4)
            self.camera_extr[:3, :3] = rot
            self.camera_extr[:3, 3] = np.array([cam_pose.p.x, cam_pose.p.y, cam_pose.p.z])

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
            t_stable = self.object_stable_poses[np.random.randint(len(self.object_stable_poses))]
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

    def step(self, time_step=0):
        ############
        # STEPPING #
        ############
        # step the physics
        self.gym.simulate(self.sim)

        # refresh results
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)

        # step rendering
        self.render()

        ##################
        # AFTER STEPPING #
        ##################
        if False:
            self.reset_env()
            self.wait(0.01)

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
            gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[15], pose)
            self.wait(10.0)

            self.gym.render_all_camera_sensors(self.sim)
            depth_image = self.gym.get_camera_image(self.sim, self.envs[15], self.camera_handles[15], gymapi.IMAGE_DEPTH) * -1
            segmask = self.gym.get_camera_image(self.sim, self.envs[15], self.camera_handles[15], gymapi.IMAGE_SEGMENTATION)
            plt.imshow(depth_image)
            plt.show()
            return

        if False:
            self.gym.render_all_camera_sensors(self.sim)
            for i in range(self.num_envs):
                depth_image = self.gym.get_camera_image(self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_DEPTH)
                segmask = self.gym.get_camera_image(self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_SEGMENTATION)
                plt.imshow(depth_image)
                plt.show()

        # reset evn
        object_poses = self.reset_env()
        self.wait(0.01)

        if True:
            # sample grasp
            features, q_values, indices, grasps = self.sample_grasp()

            # convert antipodal grasp to 3d grasp
            grasp_poses = []
            for grasp in grasps:
                if grasp is None:
                    grasp_pose = np.array([0, 0, 0.5, 0], dtype=np.float32)
                else:
                    assert isinstance(grasp, AntipodalGrasp)
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
                grasp_poses.append(grasp_pose)
            grasp_poses = np.array(grasp_poses, dtype=np.float32)
            grasp_poses = torch.from_numpy(grasp_poses).to(self.device)
            self.wait(0.01)

        # debug sample mode
        if False:
            grasp_pose = np.array([0.0, 0.0, 0.02, np.pi/2], dtype=np.float32)
            grasp_pose = np.tile(grasp_pose, [self.num_envs, 1])
            grasp_poses = torch.from_numpy(grasp_pose).to(self.device)
            print('sample grasps')
            self.wait(0.01)

        # visualize grasps
        # get grasp pose in object frame
        grasp_cp1 = []
        grasp_cp2 = []
        for env_idx in range(self.num_envs):
            # if there is no grasp, skip
            if grasps[env_idx] is None:
                continue

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
                gymutil.draw_line(cp1, cp2, color, self.gym, self.viewer, self.envs[env_idx])

            # convert to object frame
            t_wo = object_poses[env_idx]
            cp1 = t_wo.inverse().transform_point(cp1)
            cp2 = t_wo.inverse().transform_point(cp2)
            grasp_cp1.append(np.array([cp1.x, cp1.y, cp1.z], dtype=np.float32))
            grasp_cp2.append(np.array([cp2.x, cp2.y, cp2.z], dtype=np.float32))
        grasp_cp1 = np.array(grasp_cp1, dtype=np.float32)
        grasp_cp2 = np.array(grasp_cp2, dtype=np.float32)
        self.wait(1.0)

        # 2) execute x, y, theta movement
        xyt_idx = [0, 1, 3]
        self.target_dof_pos[:, xyt_idx] = grasp_poses[:, xyt_idx]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.target_dof_pos))
        self.wait(0.5)

        # 3) execute z movement
        self.target_dof_pos[:, 2] = grasp_poses[:, 2]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.target_dof_pos))
        self.wait(0.5)

        # 4) execute grasp
        self.target_dof_pos[:, 4:] = 0.0
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.target_dof_pos))
        self.wait(0.5)

        # 5) enable gravity on objects and execute lift
        self.target_dof_pos[:, 2] = 0.5
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.target_dof_pos))
        # enable gravity
        for env_idx in range(self.num_envs):
            obj_props = self.gym.get_actor_rigid_body_properties(self.envs[env_idx], self.object_handles[env_idx])
            obj_props[0].flags = gymapi.RIGID_BODY_NONE
            self.gym.set_actor_rigid_body_properties(self.envs[env_idx], self.object_handles[env_idx], obj_props, False)
        self.wait(1.0)

        # 6) evaluate grasp
        success_features = []
        failure_features = []
        success_q_values = []
        failure_q_values = []
        for env_idx in range(self.num_envs):
            # if there is no grasp, skip
            if indices[env_idx] is None:
                continue

            # evaluate object is successfully lifted up
            obj_height = self.gym.get_actor_rigid_body_states(
                self.envs[env_idx],
                self.object_handles[env_idx],
                gymapi.STATE_POS)[0]['pose']['p']['z']
            if obj_height > 0.25:
                success_features.append(
                    features[env_idx][indices[env_idx]])
                success_q_values.append(
                    q_values[env_idx][indices[env_idx]])
            else:
                failure_features.append(
                    features[env_idx][indices[env_idx]])
                failure_q_values.append(
                    q_values[env_idx][indices[env_idx]])
        success_features = np.array(success_features)
        failure_features = np.array(failure_features)
        success_q_values = np.array(success_q_values)
        failure_q_values = np.array(failure_q_values)

        # update likelihood
        self.grasp_planner.update(success_features, failure_features)
        print('success: {}, failure: {}'.format(success_features.shape[0], failure_features.shape[0]))

        # save data
        if self.save_results:
            np.save(os.path.join(self.save_dir, 'success_features', '{}.npy'.format(int(time_step))), success_features)
            np.save(os.path.join(self.save_dir, 'failure_features', '{}.npy'.format(int(time_step))), failure_features)
            np.save(os.path.join(self.save_dir, 'success_q_values', '{}.npy'.format(int(time_step))), success_q_values)
            np.save(os.path.join(self.save_dir, 'failure_q_values', '{}.npy'.format(int(time_step))), failure_q_values)

            # save object poses
            np.save(os.path.join(self.save_dir, 'grasp_cp1', '{}.npy'.format(int(time_step))), grasp_cp1)
            np.save(os.path.join(self.save_dir, 'grasp_cp2', '{}.npy'.format(int(time_step))), grasp_cp2)

        return None

    ############################################
    # robot, sensor and environment interfaces #
    ############################################
    # def wait(self, time=0.5):
    #     # wait for time seconds
    #     self.wait_timer += self.dt
    #     if self.wait_timer > time:
    #         self.wait_timer = 0.0
    #         return True
    #     else:
    #         return False

    def wait(self, sleep_time=0.5):
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

    def sample_grasp(self):
        # render depth images ans segmasks
        depth_images = []
        segmasks = []
        self.gym.render_all_camera_sensors(self.sim)
        for i in range(self.num_envs):
            depth_image = self.gym.get_camera_image(self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_DEPTH)
            segmask = self.gym.get_camera_image(self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_SEGMENTATION)
            depth_images.append(depth_image)
            segmasks.append(segmask)
        depth_images = np.array(depth_images) * -1.0
        segmasks = np.array(segmasks)

        # get grasps
        features, q_values, indices, grasps = self.grasp_planner.plan(depth_images, segmasks)
        return features, q_values, indices, grasps


if __name__ == '__main__':
    picking_evn = SigulatePickingEnv()

    for i in range(500):
        picking_evn.step(i)
        # a = input("{}".format(i))
