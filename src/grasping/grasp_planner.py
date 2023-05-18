#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import prettytable
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from autolab_core import (YamlConfig, CameraIntrinsics, ColorImage,
                          DepthImage, BinaryImage, RgbdImage)
from visualization import Visualizer2D as vis
from gqcnn.grasping import (Grasp2D, SuctionPoint2D, RgbdImageState,
                            CrossEntropyRobustGraspingPolicy,
                            UCBThompsonSamplingGraspingPolicy,
                            BetaProcessCrossEntropyRobustGraspingPolicy)
from gqcnn.utils import GripperMode, NoValidGraspsException
from gqcnn.grasping.grasp import Grasp2D, SuctionPoint2D
from gqcnn.grasping.policy import GraspAction


class GraspPlanner(object):
    def __init__(self, config_filename):
        # init policy
        planner_config = YamlConfig(config_filename)
        policy_config = planner_config['policy']

        if policy_config['type'] == 'cem':
            self.policy = CrossEntropyRobustGraspingPolicy(policy_config)
        elif policy_config['type'] == 'ucb-ts':
            self.policy = UCBThompsonSamplingGraspingPolicy(policy_config)
        elif policy_config['type'] == 'bp':
            self.policy = BetaProcessCrossEntropyRobustGraspingPolicy(policy_config)
        elif policy_config['type'] == 'ql':
            raise NotImplementedError

    def get_rgbd_state(self, rgb_image, depth_image, segmask, camera_intr):
        """
        Args:
            rgb_image (np.ndarray): rgb image (HxWx3)
            depth_image (np.ndarray): depth image (HxW)
            segmask (np.ndarray): segmentation mask (HxW)
            camera_intr (CameraIntrinsics): camera intrinsics (3, 3)

        Returns:
            RgbdImageState: rgbd state
        """

        # wrap images in perception objects
        frame = "camera"
        color_im = ColorImage(rgb_image, frame)
        depth_im = DepthImage(depth_image, frame)
        segmask[segmask > 0] = 255
        segmask = segmask.astype('uint8')
        segmask = BinaryImage(segmask, frame)
        camera_intr = CameraIntrinsics(
            frame=frame,
            fx=camera_intr[0, 0],
            fy=camera_intr[1, 1],
            cx=camera_intr[0, 2],
            cy=camera_intr[1, 2],
            height=depth_im.height,
            width=depth_im.width)

        # inpaint
        color_im = color_im.inpaint(rescale_factor=0.5)
        depth_im = depth_im.inpaint(rescale_factor=0.5)

        # create rgbd state
        rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
        rgbd_state = RgbdImageState(rgbd_im, camera_intr, segmask)
        return rgbd_state

    def execute_policy(self, rgb_image, depth_image, segmask, camera_intr):
        """Execute the policy.

        Args:
            rgbd_state (RgbdImageState): rgbd state

        Returns:
            GraspAction: grasp action
        """
        rgbd_state = self.get_rgbd_state(
            rgb_image, depth_image, segmask, camera_intr)

        try:
            action = self.policy(rgbd_state)
        except NoValidGraspsException:
            action = None

        return action

    def update_policy(self, action, reward):
        """Update the policy.

        Args:
            action (GraspAction): grasp action
            reward (float): reward
        """
        self.policy.update(action, reward)

    def get_grasp_pose_world(self, grasp, camera_extrinsic):
        """Get grasp pose in world frame.

        Args:
            grasp (Grasp2D | SuctionPoint2D): Grasp.
            camera_extrinsic (np.ndarray): Camera extrinsic matrix.

        Returns:
            np.ndarray: (4, 4) homogeneous transformation matrix of grasp pose in world frame.
        """
        if isinstance(grasp, Grasp2D):
            return self._get_pj_grasp_pose_world(grasp, camera_extrinsic)
        elif isinstance(grasp, SuctionPoint2D):
            return self._get_sc_grasp_pose_world(grasp, camera_extrinsic)

    def _get_pj_grasp_pose_world(self, grasp, camera_extrinsic):
        """Transform pose of grasp to world frame.

        The approach pose of the grasp is parallel to the camera optical axis.
        First, the pose is transformed to the camera frame. Then, modify approach pose to be
        normal to the table surface.

        Args:
            grasp (`Grasp2D`): Parallel-jaw grasp.

        Returns:
            `ParallelJawGrasp`: Transformed parallel-jaw grasp.
        """
        assert isinstance(grasp, Grasp2D)

        grasp_tf_camera = grasp.pose().matrix
        camera_tf_world = camera_extrinsic
        grasp_tf_world = np.dot(camera_tf_world, grasp_tf_camera)

        # 1) rotate 90(deg) along the y-axis to match with the Moveit config
        # 2) ratate -90(deg) along the z-axis to make parallel-jaw grasp competitable
        grasp_rot_mat = R.from_matrix(grasp_tf_world[0:3, 0:3]).as_matrix()
        y90_mat = R.from_euler('y', np.pi / 2).as_matrix()
        z90_mat = R.from_euler('z', -np.pi / 2).as_matrix()
        grasp_rot_mat = np.linalg.multi_dot([grasp_rot_mat, y90_mat, z90_mat])

        # 3) make grasp pose be normal to the table
        approach_dir = np.array([0, 0, -1])
        grasp_z = grasp_rot_mat[0:3, 2]
        rot_angle = np.arccos(np.dot(grasp_z, approach_dir))
        rot_dir = np.cross(grasp_z, approach_dir)
        rot_dir = rot_dir / np.linalg.norm(rot_dir)
        grasp_rot_mat = np.dot(
            R.from_rotvec(rot_angle * rot_dir).as_matrix(),
            grasp_rot_mat)

        # 4) update grasp pose
        grasp_tf_world[0:3, 0:3] = grasp_rot_mat
        return grasp_tf_world

    def _get_sc_grasp_pose_world(self, grasp, camera_extrinsic):
        """Transform pose of grasp to world frame.

        The approach pose of the grasp is parallel to the camera optical axis.

        Args:
            grasp (`SuctionPoint2D`): Parallel-jaw grasp.

        Returns:
            `ParallelJawGrasp`: Transformed parallel-jaw grasp.
        """
        assert isinstance(grasp, SuctionPoint2D)

        grasp_tf_camera = grasp.pose().matrix
        camera_tf_world = camera_extrinsic
        grasp_tf_world = np.dot(camera_tf_world, grasp_tf_camera)
        return grasp_tf_world


class RuleBasedSelectGraspAction():
    def __init__(self, init_pj_success, init_sc_success):
        self.prev_success_mask = {'pj': init_pj_success, 'sc': init_sc_success}

    def select(self, pj_action, sc_action, verbose=False):
        """Policy for choosing grasp.

        Priority: 1) existance 2) prev_success 3) score

        Args:
            pj_action (`PallelJawGrasp`): Best parallel-jaw grasp
            sc_action (`SuctionCupGrasp`): Best suction-cup grasp
            verbose (`bool`): If True, print debug message. Defaults to False.

        Returns:
            grasp_action (`GraspAction`): Selected grasp action.
            policy_log (`PrettyTable`): Log of policy.
        """
        if pj_action is not None:
            assert isinstance(pj_action, GraspAction)
            assert isinstance(pj_action.grasp, Grasp2D)
        if sc_action is not None:
            assert isinstance(sc_action, GraspAction)
            assert isinstance(sc_action.grasp, SuctionPoint2D)

        # if all success mask it false, reset to true
        if not any(self.prev_success_mask.values()):
            # print('Reset success mask to true. (all false)')
            self.reset()

        # print grasp information
        table = prettytable.PrettyTable(['Grasp', 'Existance', 'Score', 'Prev Success'])
        table.add_row([
            'Parallel-jaw',
            pj_action is not None,
            pj_action.q_value if pj_action is not None else 0.0,
            self.prev_success_mask['pj']])
        table.add_row([
            'Suction-cup',
            sc_action is not None,
            sc_action.q_value if sc_action is not None else 0.0,
            self.prev_success_mask['sc']])
        if verbose:
            print('Select grasp policy.\n{}'.format(table.get_string()))

        # priority 1: existance
        if (pj_action is None) and (sc_action is None):
            print('No valid grasp. Shutdown!')
            return None, table
        elif (pj_action is not None) and (sc_action is None):
            if verbose:
                print('Choose parallel-jaw grasp: Only parallel-jaw grasp is valid.')
            return pj_action, table
        elif (pj_action is None) and (sc_action is not None):
            if verbose:
                print('Choose suction-cup grasp: Only suction-cup grasp is valid.')
            return sc_action, table

        # priority 3: prev_success
        if (not self.prev_success_mask['sc']) and self.prev_success_mask['pj']:
            if verbose:
                print('Choose parallel-jaw grasp: Previous suction-cup grasp failed.')
            return pj_action, table
        elif self.prev_success_mask['sc'] and (not self.prev_success_mask['pj']):
            if verbose:
                print('Choose suction-cup grasp: Previous parallel-jaw grasp failed.')
            return sc_action, table

        # priority 4: score
        if pj_action.q_value > sc_action.q_value:
            if verbose:
                print('Choose parallel-jaw grasp: Parallel-jaw grasp has higher q_value.')
            return pj_action, table
        elif pj_action.q_value < sc_action.q_value:
            if verbose:
                print('Choose suction-cup grasp: Suction-cup grasp has higher q_value.')
            return sc_action, table

    def update(self, action, success):
        """Update success mask.

        Args:
            action (`GraspAction`): Grasp action.
            success (`bool`): If True, grasp is successful.
        """
        if success:
            self.reset()
            return

        if isinstance(action.grasp, Grasp2D):
            self.prev_success_mask['pj'] = success
        elif isinstance(action.grasp, SuctionPoint2D):
            self.prev_success_mask['sc'] = success
        else:
            raise ValueError('Unknown action type: {}'.format(type(action)))

    def reset(self):
        """Reset success mask.
        """
        self.prev_success_mask['pj'] = True
        self.prev_success_mask['sc'] = True

    @staticmethod
    def label_table_on_image(color_img, table, action, success, scale=1.0):
        """Label table on image.

        Args:
            color_img (`np.ndarray`): Color image. (H, W, 3)
            table (`PrettyTable`): Table to label
            action (`GraspAction`): Grasp action
            success (`bool`): If True, grasp is successful
            scale (`float`): Scale of the image. Defaults to 1.0.

        Returns:
            labeled_image (`np.ndarray`): Labeled image. (H + 80, W, 3)
        """
        rows = [table._field_names] + table._rows

        # create caption area on the image
        H, W, C = color_img.shape
        labeled_image = np.zeros((int(H + 110), W, C), dtype=np.uint8)
        labeled_image[:H, :, :] = color_img

        if isinstance(action.grasp, Grasp2D):
            select_grasp_type = 0
        elif isinstance(action.grasp, SuctionPoint2D):
            select_grasp_type = 1

        # write success or failure on the image
        msg = '{}: {}'.format(
            'Parallel-jaw' if select_grasp_type == 0 else 'Suction-cup',
            'Success' if success else 'Fail')
        cv2.putText(
            img=labeled_image,
            text=msg,
            org=(20, H + 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=(0, 255, 0) if success else (255, 0, 0),
            thickness=2)

        # draw table on the image
        y0, x0 = H + 30, 20
        dy, dx = 20, 105
        for i, row in enumerate(rows):
            y = y0 + (i + 1) * dy
            for j, cell in enumerate(row):
                x = x0 + (j * dx)

                if isinstance(cell, float):
                    text = '{:.4f}'.format(cell)
                    color = (255, 255, 255)
                elif isinstance(cell, bool):
                    text = str(cell)
                    color = (0, 255, 0) if cell else (255, 0, 0)
                else:
                    text = str(cell)
                    color = (255, 255, 255)
                cv2.putText(
                    img=labeled_image,
                    text=text,
                    org=(x, y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=color,
                    thickness=1,
                    lineType=cv2.LINE_AA)

        # resize image to save disk space
        size = (int(scale * labeled_image.shape[1]),
                int(scale * labeled_image.shape[0]))
        labeled_image = cv2.resize(labeled_image, size)
        return labeled_image
