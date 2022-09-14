import os
import sys
import time
from multiprocessing import Pool, Process
from functools import partial
from itertools import repeat
from importlib.util import spec_from_file_location, module_from_spec

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch


class AntipodalGrasp(object):
    """Antipodal grasp in depth image.
    Args:
        contact_point1 (numpy.ndarray): [y, x], contact point[pixel] in image
        contact_point2 (numpy.ndarray): [y, x], contact point[pixel] in image
        axis_vector (numpy.ndarray): [y, x], grasp axis = norm(contact_point2 - contact_point1)
        depth (float): depth value[m] of the grasp center in image
    """
    def __init__(self, contact_point1, contact_point2, axis_vector, depth):
        self.contact_point1 = contact_point1
        self.contact_point2 = contact_point2
        self.axis_vector = axis_vector
        self.depth = depth

    @property
    def center_point(self):
        """Grasp center position in image, [y, x].
        Returns:
            numpy.ndarray: [y, x], grasp center position in image.
        """
        return ((self.contact_point1 + self.contact_point2)/2.0).astype('uint16')

    @property
    def angle(self):
        """Grasp axis angle in deg.
        Returns:
            float: garsp axis angle.
        """
        return np.arctan2(self.axis_vector[0], self.axis_vector[1]) * 180.0 / np.pi

    @property
    def grasp_width_in_px(self):
        return np.linalg.norm(self.contact_point2 - self.contact_point1)

    def get_grasp_width_in_3d(self, camera_matrix):
        """Get grasp width in [m].
        Args:
            camera_matrix (`numpy.ndarray`): [description]
        Returns:
            float: grasp width in [m].
        """
        contact_point1_camera = self.depth * np.linalg.inv(camera_matrix).dot([self.contact_point1[1], self.contact_point1[0],  1.0])
        contact_point2_camera = self.depth * np.linalg.inv(camera_matrix).dot([self.contact_point2[1], self.contact_point2[0],  1.0])
        return np.linalg.norm(contact_point1_camera - contact_point2_camera)

    def get_3d_grasp_pose(self, table_normal, camera_matrix, camera_tf):
        """Get 3d grasp pose from the camera frame.
        Args:
            table_normal (`numpy.ndarray`): unit normal of the table in wolrd frame.
            camera_matrix (`numpy.ndarray`): 3x3 camera intrinsic matrix.
            camera_tf (`numpy.ndarray`): 4x4 camera extrinsic matirx(transformation matrix).
        Returns:
            `numpy.ndarray`: 4x4 grasp pose transformation matrix in camera frame.
        """
        camera_matrix = np.array(camera_matrix)
        camera_tf = np.array(camera_tf)

        # compute 3d grasp axis in camera frame
        grasp_axis_image = np.flip(self.axis_vector) / np.linalg.norm(self.axis_vector)
        grasp_axis_camera = np.array([grasp_axis_image[0], grasp_axis_image[1], 0])
        grasp_axis_camera = grasp_axis_camera / np.linalg.norm(grasp_axis_camera)

        # convert 3d grasp pose in camera frame
        camera_rot = camera_tf[0:3, 0:3]

        grasp_z_camera = np.linalg.inv(camera_rot).dot(table_normal)
        grasp_x_camera = np.cross(grasp_axis_camera, grasp_z_camera)
        grasp_y_camera = np.cross(grasp_z_camera, grasp_x_camera)

        # make unit vector
        grasp_x_camera = grasp_x_camera / np.linalg.norm(grasp_x_camera)
        grasp_y_camera = grasp_y_camera / np.linalg.norm(grasp_y_camera)
        grasp_z_camera = grasp_z_camera / np.linalg.norm(grasp_z_camera)
        grasp_rot_camera = np.array([grasp_x_camera, grasp_y_camera, grasp_z_camera]).T

        # get 3D position of grasp center in camera frame
        grasp_center_camera = self.depth * np.linalg.inv(camera_matrix).dot([self.center_point[1], self.center_point[0],  1.0])
        grasp_tf = np.zeros((4, 4), dtype='float')
        grasp_tf[0:3, 0:3] = grasp_rot_camera
        grasp_tf[0:3, 3] = grasp_center_camera
        grasp_tf[3, 3] = 1.0
        return grasp_tf


def grasp_to_tensor(depth_image, grasps):
    """Convert candidate grasps to tensors.
    Args:
        depth_image (numpy.ndarry): HxW depth image.
        grasps (list): list with candidate grasp as AntipodalGrasp.
    Returns:
        tuple: (image_tensors, depth_tensors) with numpy.ndarray
    """
    # get height, width
    height, width = depth_image.shape
    num_sample = len(grasps)

    # pre-allocate tensors
    image_tensors = np.zeros(
        (num_sample, 32, 32, 1),
        dtype=np.float32,
        )
    depth_tensors = np.zeros(
        (num_sample, 1),
        dtype=np.float32,
    )

    # convert grasp to tensors
    for i, grasp in enumerate(grasps):
        # deep copy the image
        image_tensor = depth_image.copy()

        # get center in (x,y)
        center = (grasp.center_point[1], grasp.center_point[0])

        # rotate image
        matrix = cv2.getRotationMatrix2D(
            center,
            grasp.angle,
            1,
            )
        image_tensor = cv2.warpAffine(image_tensor, matrix, (width, height))

        # crop and resize image
        image_tensor = image_tensor[
            center[1]-48:center[1]+48,
            center[0]-48:center[0]+48]  # -55, +56??
        try:
            image_tensor = cv2.resize(
                image_tensor, dsize=(32, 32),
                interpolation=cv2.INTER_LINEAR,
                )
        except Exception as e:
            print('center: ', center)
            print(image_tensor)
            plt.subplot(1, 2, 1)
            plt.imshow(depth_image)
            plt.subplot(1, 2, 2)
            plt.imshow(image_tensor)
            plt.show()
            raise e
            
        image_tensor = image_tensor.reshape((32, 32, 1))

        # update to the tensor
        image_tensors[i, :] = image_tensor
        depth_tensors[i] = grasp.depth
    return (image_tensors, depth_tensors)


def antipodal_sampler(
        depth_image,
        segmask_image,
        config,
        ):
    """Image based antipodal sampler.
    The performance of this fuction depends on the edge detection algorithm.
    Args:
        depth_image (numpy.ndarray): HxW depth images in [m].
        segmask_image (numpy.ndarray): HxW object segment mask. Value of the mask pixel is 1, others 0.
        save_fig_dir (string, optional): Save figure directory. Defaults to None.
        save_fig (bool, optional): If save_fig=`True`, then save the rgb, depth, edge, surface normal images. Defaults to False.
    Returns:
        lsit: Grasp candidates in list. Each element is `AntipodalGrasp` instance.
    """
    max_gripper_width = config['max_gripper_width']
    depth_laplacian_threshold = config['depth_laplacian_threshold']
    antipodal_threshold_angle = config['antipodal_threshold_angle']
    max_num_sample = config['max_num_sample']
    num_depth_sample = config['num_depth_sample']
    depth_offset_min = config['depth_offset_min']
    depth_offset_max = config['depth_offset_max']

    # deep copy the images
    azure_depth_img = depth_image.copy()
    segmask_img = segmask_image.copy()

    # get edge using Laplacian with depth
    depth_laplacian = cv2.Laplacian(azure_depth_img, cv2.CV_64F, ksize=5)
    depth_laplacian = np.abs(depth_laplacian)
    depth_edge_image_laplacian = np.zeros(azure_depth_img.shape).astype('uint8')
    depth_edge_image_laplacian[depth_laplacian > depth_laplacian_threshold] = 255

    # visualize edge image
    if False:
        plt.subplot(1, 3, 1)
        plt.imshow(segmask_img, cmap='gray')
        plt.subplot(1, 3, 2)
        plt.imshow(depth_edge_image_laplacian, cmap='jet', alpha=0.5)
        edge_image = depth_edge_image_laplacian * (segmask_img.astype('uint8'))
        plt.subplot(1,3,3)
        plt.imshow(edge_image, cmap='gray')
        plt.show()

    # get edge pixels
    edge_image = depth_edge_image_laplacian * segmask_img
    edge_image[edge_image > 0] = 255
    edge_pixel_y, edge_pixel_x = np.where(edge_image == 255)

    # get surface nomal using gradient
    grad_image = np.copy(azure_depth_img)
    min_depth = np.min(grad_image)
    max_depth = np.max(grad_image)
    grad_image = (grad_image-min_depth)/(max_depth-min_depth) * 255
    grad_image = grad_image.astype(np.uint8)
    grad_dx_image, grad_dy_image = cv2.spatialGradient(grad_image)

    normal_y = grad_dy_image[edge_pixel_y, edge_pixel_x].astype("float")
    normal_x = grad_dx_image[edge_pixel_y, edge_pixel_x].astype("float")
    length = np.linalg.norm(np.c_[normal_y, normal_x], axis=1)

    # remove 0 normals
    zero_length_indicies = np.where(length == 0)[0]
    edge_pixel_y = np.delete(edge_pixel_y, zero_length_indicies)
    edge_pixel_x = np.delete(edge_pixel_x, zero_length_indicies)
    normal_y = np.delete(normal_y, zero_length_indicies)
    normal_x = np.delete(normal_x, zero_length_indicies)
    length = np.delete(length, zero_length_indicies)
    unit_normals = np.c_[normal_y/length, normal_x/length]
    edge_pixels = np.c_[edge_pixel_y, edge_pixel_x]

    # visualzie surface normals
    if False:
        azure_depth_3ch_img = np.copy(azure_depth_img)
        azure_depth_3ch_img = np.reshape(
            azure_depth_3ch_img,
            (azure_depth_3ch_img.shape[0], azure_depth_3ch_img.shape[1], 1),
            )
        azure_depth_3ch_img = np.concatenate(
            (azure_depth_3ch_img, azure_depth_3ch_img, azure_depth_3ch_img),
            axis=2,
            )
        for idx, unit_normal in enumerate(unit_normals):
            pt1 = (edge_pixel_x[idx], edge_pixel_y[idx])
            unit_normal = 10*unit_normal
            unit_normal = unit_normal.astype(np.int8)
            pt2 = (pt1[0] + unit_normal[1], pt1[1] + unit_normal[0])
            cv2.arrowedLine(azure_depth_3ch_img, pt1, pt2, (0, 255, 0), 1)
        cv2.imshow("normal", azure_depth_3ch_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # get antipodal normals
    threshold_deg = antipodal_threshold_angle
    normal_inner_products = np.dot(unit_normals, np.transpose(unit_normals))
    distance_matrix = np.linalg.norm(
        edge_pixels[:, None, :] - edge_pixels[None, :, :],
        axis=-1,
        )

    candidate_index = np.where(
        (normal_inner_products < -np.cos(threshold_deg*np.pi/180.0)) &
        (distance_matrix < max_gripper_width)
        )
    pair1 = candidate_index[0][candidate_index[0] < candidate_index[1]]
    pair2 = candidate_index[1][candidate_index[0] < candidate_index[1]]

    axis_vector = (edge_pixels[pair2, :] - edge_pixels[pair1, :]).astype("float")
    axis_length = np.linalg.norm(axis_vector, axis=1)
    axis_vector = np.c_[axis_vector[:, 0]/axis_length, axis_vector[:, 1]/axis_length]
    candidate_index = np.where(
        ((axis_vector * unit_normals[pair1]).sum(axis=1) < -np.cos(threshold_deg*np.pi/180.0)) &
        ((axis_vector * unit_normals[pair2]).sum(axis=1) < np.cos(threshold_deg*np.pi/180.0))
        )[0]

    # random sampling from the antipodal candidate grasps
    grasp_candidates = []
    if candidate_index.shape[0] < max_num_sample:
        num_sample = candidate_index.shape[0]
    else:
        num_sample = max_num_sample
    rand_indicies = np.random.choice(candidate_index, num_sample, replace=False)
    for rand_index in rand_indicies:
        pair1_index = pair1[rand_index]
        pair2_index = pair2[rand_index]
        contact_pixel1 = edge_pixels[pair1_index, :]
        contact_pixel2 = edge_pixels[pair2_index, :]
        center_pixel = ((contact_pixel1 + contact_pixel2)/2.0).astype('uint16')
        raw_depth = azure_depth_img[center_pixel[0], center_pixel[1]]

        depth_offset_samples = np.random.rand(num_depth_sample) * (depth_offset_max - depth_offset_min) + depth_offset_min
        depth_offset_samples += raw_depth

        # add grasp to grasp candidates list
        for depth_sample in depth_offset_samples:
            grasp = AntipodalGrasp(
                contact_pixel1,
                contact_pixel2,
                axis_vector[rand_index],
                depth_sample,
                )
            grasp_candidates.append(grasp)

        # visualzie individual antipodal candidate grasp
        if False:
            # convert to drawable image
            azure_depth_3ch_img = np.copy(azure_depth_img)
            azure_depth_3ch_img = np.reshape(
                azure_depth_3ch_img,
                (azure_depth_3ch_img.shape[0], azure_depth_3ch_img.shape[1], 1),
                )
            azure_depth_3ch_img = np.concatenate(
                (azure_depth_3ch_img, azure_depth_3ch_img, azure_depth_3ch_img),
                axis=2,
                )

            # draw surface normal and contact point1
            pt1 = (contact_pixel1[1], contact_pixel1[0])
            unit_normal = 15*unit_normals[pair1_index]
            unit_normal = unit_normal.astype(np.int8)
            pt2 = (pt1[0] + unit_normal[1], pt1[1] + unit_normal[0])
            cv2.circle(azure_depth_3ch_img, pt1, 3, (0, 0, 255), 1)
            cv2.arrowedLine(azure_depth_3ch_img, pt1, pt2, (0, 0, 255), 1)

            # draw surface normal and contact point1
            pt1 = (contact_pixel2[1], contact_pixel2[0])
            unit_normal = 15*unit_normals[pair2_index]
            unit_normal = unit_normal.astype(np.int8)
            pt2 = (pt1[0] + unit_normal[1], pt1[1] + unit_normal[0])
            cv2.circle(azure_depth_3ch_img, pt1, 3, (0, 0, 255), 1)
            cv2.arrowedLine(azure_depth_3ch_img, pt1, pt2, (0, 0, 255), 1)

            # draw center pixel
            cv2.circle(
                azure_depth_3ch_img,
                center=(center_pixel[1], center_pixel[0]),
                radius=3,
                color=(0, 0, 255),
                thickness=1,
                )
            cv2.imshow("candidate grasp on depth image", azure_depth_3ch_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return grasp_candidates


def get_activation(activations, name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook


# TODO: do with numpy first
class ThompsonSamplingGraspPlanner(object):
    def __init__(self):
        # grasp sample config, TODO: move to config file
        max_gripper_width = 0.08
        camera_matrix = np.array([[525.0, 0.0, 320.0], [0.0, 525.0, 240.0], [0.0, 0.0, 1.0]])
        max_grasp_point1_3d = [max_gripper_width, 0, 0.75]
        max_grasp_point2_3d = [0, 0, 0.75]
        max_grasp_point1_image = np.dot(camera_matrix, max_grasp_point1_3d) / 0.75
        max_grasp_point2_image = np.dot(camera_matrix, max_grasp_point2_3d) / 0.75
        max_gripper_width = np.linalg.norm(max_grasp_point2_image - max_grasp_point1_image)
        self.sample_config = {
            'max_gripper_width': max_gripper_width, # max_gripper_width
            'depth_laplacian_threshold': 0.04,
            'antipodal_threshold_angle': 10,
            'max_num_sample': 200,
            'num_depth_sample': 5,
            'depth_offset_min': 0.005,
            'depth_offset_max': 0.03}
        self.prior_strength = 10.0
        self.similarity_threshold = 0.95

        # load gqcnn
        root_dir = '/media/sungwon/WorkSpace/projects/feature_space_analysis'
        model_name = '2022-08-16-2023'

        spec = spec_from_file_location('model.GQCNN', os.path.join(root_dir, 'models', model_name, 'model.py'))
        module = module_from_spec(spec)
        if 'model.GQCNN' in sys.modules:
            del sys.modules['model.GQCNN']
        sys.modules['model.GQCNN'] = module
        spec.loader.exec_module(module)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = module.GQCNN()
        self.model.load_state_dict(torch.load(os.path.join(root_dir, 'models', model_name, 'model.pt')))
        self.model.to(device)
        self.model.train(False)

        self.im_mean = np.load(os.path.join(root_dir, 'models', model_name, 'im_mean.npy'))
        self.im_std = np.load(os.path.join(root_dir, 'models', model_name, 'im_std.npy'))
        self.pose_mean = np.load(os.path.join(root_dir, 'models', model_name, 'pose_mean.npy'))
        self.pose_std = np.load(os.path.join(root_dir, 'models', model_name, 'pose_std.npy'))

        self.activations = {}
        try:
            self.model.merge_stream[-2].register_forward_hook(get_activation(self.activations, '(n-1)layer'))
        except:
            self.model.feature_fc_stack[-2].register_forward_hook(get_activation(self.activations, '(n-1)layer'))

        # posterior variables
        self.success_features = None
        self.failure_features = None

    def update(self, success_feature, failure_feature):
        if (len(success_feature) > 0) and (self.success_features is None):
            self.success_features = success_feature
        elif len(success_feature) > 0:
            self.success_features = np.r_[self.success_features, success_feature]

        if (len(failure_feature) > 0) and (self.failure_features is None):
            self.failure_features = failure_feature
        elif len(failure_feature) > 0:
            self.failure_features = np.r_[self.failure_features, failure_feature]

    def posterior_sample(self, features, q_values):
        # if there are no samples, return None
        if len(q_values) == 0:
            print('Cannot posterior sample from empty samples')
            return None

        # get likelihood
        if self.success_features is None:
            n = np.zeros(features.shape[0])
        else:
            pos_dist = np.dot(features, self.success_features.T)
            n = np.count_nonzero(pos_dist > self.similarity_threshold, axis=1)
        if self.failure_features is None:
            m = np.zeros(features.shape[0])
        else:
            neg_dist = np.dot(features, self.failure_features.T)
            m = np.count_nonzero(neg_dist > self.similarity_threshold, axis=1)

        # get prior
        a = self.prior_strength*q_values + 1e-10
        b = self.prior_strength*(1.0 - q_values) + 1e-10

        # get posterior
        prob = np.random.beta(a + n, b + m)
        idx = np.argmax(prob)

        print('prios: {:.2f}, {:.2f} | likelihood: {:04.0f}, {:04.0f} | posterior: {:07.2f}, {:07.2f} | prob: {:.2f}'.format(
            a[idx], b[idx], n[idx], m[idx], (a+n)[idx], (b+m)[idx], prob[idx]))
        return idx

    def plan(self, depth_images, segmasks, verbose=False):
        # pool is not efficient for this task
        # get antipodal grasps from  depth images
        start_time = time.time()
        grasp_candidates = []
        for depth_image, segmask in zip(depth_images, segmasks):
            grasp_candidates.append(antipodal_sampler(depth_image, segmask, self.sample_config))
        if verbose:
            print('Get antipodal grasp time: {:.2f}s'.format(time.time() - start_time))

        # get grasp candidates
        start_time = time.time()
        im_tensors, depth_tensors = [], []
        grasp_to_tensor(depth_images[0], grasp_candidates[0])
        with Pool(16) as p:
            temp = p.starmap(grasp_to_tensor, zip(depth_images, grasp_candidates))
        for i in range(len(temp)):
            im_tensors.append(temp[i][0])
            depth_tensors.append(temp[i][1])
        if verbose:
            print('Convert grasp to tensor time: {:.2f}s'.format(time.time() - start_time))

        # get grasp quality
        start_time = time.time()
        features, q_values = [], []
        with torch.no_grad():
            for im_tensor, depth_tensor in zip(im_tensors, depth_tensors):
                # normalize
                im_tensor = (im_tensor - self.im_mean) / self.im_std
                depth_tensor = (depth_tensor - self.pose_mean) / self.pose_std

                # numpy to tensor
                im_tensor = torch.from_numpy(im_tensor).to('cuda').view(-1, 1, 32, 32)
                depth_tensor = torch.from_numpy(depth_tensor).to('cuda')

                # get result
                output = self.model(im_tensor, depth_tensor)
                q_value = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
                feature = self.activations['(n-1)layer'].cpu().numpy()
                feature = feature/np.linalg.norm(feature, axis=1, keepdims=True)

                # # debug
                # r = np.random.randint(20, 50)
                # feature = feature[:r]
                # q_value = q_value[:r]

                features.append(feature)
                q_values.append(q_value)
        if verbose:
            print('GQ-CNN inference time: {:.2f}s'.format(time.time() - start_time))

        # do posterior sampling
        indices = []
        grasps = []
        for feature, q_value, grasp_candidate in zip(features, q_values, grasp_candidates):
            idx = self.posterior_sample(feature, q_value)
            indices.append(idx)
            if idx is not None:
                grasps.append(grasp_candidate[idx])
            else:
                grasps.append(None)

        features = np.array(features)
        q_values = np.array(q_values)
        indices = np.array(indices)

        # DEBUG: visualzie grasp in full scene
        if False:
            fig, ax = plt.subplots(1)
            ax.imshow(depth_images[0])
            g = grasps[0]
            assert isinstance(g, AntipodalGrasp)
            circ = Circle((g.contact_point1[1], g.contact_point1[0]), 10, color='red')
            ax.add_patch(circ)
            circ = Circle((g.contact_point2[1], g.contact_point2[0]), 10, color='red')
            ax.add_patch(circ)
            plt.show()

        return features, q_values, indices, grasps


# TODO: do with numpy first
class CEMThompsonSamplingGraspPlanner(object):
    def __init__(self):
        # grasp sample config, TODO: move to config file
        max_gripper_width = 0.07
        camera_matrix = np.array([[525.0, 0.0, 320.0], [0.0, 525.0, 240.0], [0.0, 0.0, 1.0]])
        max_grasp_point1_3d = [max_gripper_width, 0, 0.75]
        max_grasp_point2_3d = [0, 0, 0.75]
        max_grasp_point1_image = np.dot(camera_matrix, max_grasp_point1_3d) / 0.75
        max_grasp_point2_image = np.dot(camera_matrix, max_grasp_point2_3d) / 0.75
        max_gripper_width = np.linalg.norm(max_grasp_point2_image - max_grasp_point1_image)
        self.sample_config = {
            'max_gripper_width': max_gripper_width, # max_gripper_width
            'depth_laplacian_threshold': 0.04,
            'antipodal_threshold_angle': 10,
            'max_num_sample': 100,
            'num_depth_sample': 10,
            'depth_offset_min': 0.005,
            'depth_offset_max': 0.03}
        self.prior_strength = 50.0
        self.similarity_threshold = 0.95

        # load gqcnn
        root_dir = '/media/sungwon/WorkSpace/projects/feature_space_analysis'
        model_name = '2022-08-16-2023'

        spec = spec_from_file_location('model.GQCNN', os.path.join(root_dir, 'models', model_name, 'model.py'))
        module = module_from_spec(spec)
        if 'model.GQCNN' in sys.modules:
            del sys.modules['model.GQCNN']
        sys.modules['model.GQCNN'] = module
        spec.loader.exec_module(module)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = module.GQCNN()
        self.model.load_state_dict(torch.load(os.path.join(root_dir, 'models', model_name, 'model.pt')))
        self.model.to(device)
        self.model.train(False)

        self.activations = {}
        try:
            self.model.merge_stream[-2].register_forward_hook(get_activation(self.activations, '(n-1)layer'))
        except:
            self.model.feature_fc_stack[-2].register_forward_hook(get_activation(self.activations, '(n-1)layer'))

        # posterior variables
        self.success_features = None
        self.failure_features = None

    def update(self, success_feature, failure_feature):
        if (len(success_feature) > 0) and (self.success_features is None):
            self.success_features = success_feature
        elif len(success_feature) > 0:
            self.success_features = np.r_[self.success_features, success_feature]

        if (len(failure_feature) > 0) and (self.failure_features is None):
            self.failure_features = failure_feature
        elif len(failure_feature) > 0:
            self.failure_features = np.r_[self.failure_features, failure_feature]

    def posterior_sample(self, features, q_values):
        # if there are no samples, return None
        if len(q_values) == 0:
            print('Cannot posterior sample from empty samples')
            return None

        # get likelihood
        if self.success_features is None:
            n = np.zeros(features.shape[0])
        else:
            pos_dist = np.dot(features, self.success_features.T)
            n = np.count_nonzero(pos_dist > self.similarity_threshold, axis=1)
        if self.failure_features is None:
            m = np.zeros(features.shape[0])
        else:
            neg_dist = np.dot(features, self.failure_features.T)
            m = np.count_nonzero(neg_dist > self.similarity_threshold, axis=1)

        # get prior
        a = self.prior_strength*q_values + 1e-10
        b = self.prior_strength*(1.0 - q_values) + 1e-10

        # get posterior
        prob = np.random.beta(a + n, b + m)
        idx = np.argmax(prob)

        print('prios: {:.2f}, {:.2f} | likelihood: {:04.0f}, {:04.0f} | posterior: {:07.2f}, {:07.2f} | prob: {:.2f}'.format(
            a[idx], b[idx], n[idx], m[idx], (a+n)[idx], (b+m)[idx], prob[idx]))
        return idx

    def plan(self, depth_images, segmasks, verbose=False):
        # pool is not efficient for this task
        # get antipodal grasps from  depth images
        start_time = time.time()
        grasp_candidates = []
        for depth_image, segmask in zip(depth_images, segmasks):
            grasp_candidates.append(antipodal_sampler(depth_image, segmask, self.sample_config))
        if verbose:
            print('Get antipodal grasp time: {:.2f}s'.format(time.time() - start_time))

        # get grasp candidates
        start_time = time.time()
        im_tensors, depth_tensors = [], []
        grasp_to_tensor(depth_images[0], grasp_candidates[0])
        with Pool(16) as p:
            temp = p.starmap(grasp_to_tensor, zip(depth_images, grasp_candidates))
        for i in range(len(temp)):
            im_tensors.append(temp[i][0])
            depth_tensors.append(temp[i][1])
        if verbose:
            print('Convert grasp to tensor time: {:.2f}s'.format(time.time() - start_time))

        # get grasp quality
        start_time = time.time()
        features, q_values = [], []
        with torch.no_grad():
            for im_tensor, depth_tensor in zip(im_tensors, depth_tensors):
                im_tensor = torch.from_numpy(im_tensor).to('cuda').view(-1, 1, 32, 32)
                depth_tensor = torch.from_numpy(depth_tensor).to('cuda')
                output = self.model(im_tensor, depth_tensor)
                q_value = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
                feature = self.activations['(n-1)layer'].cpu().numpy()
                feature = feature/np.linalg.norm(feature, axis=1, keepdims=True)

                # # debug
                # r = np.random.randint(20, 50)
                # feature = feature[:r]
                # q_value = q_value[:r]

                features.append(feature)
                q_values.append(q_value)
        if verbose:
            print('GQ-CNN inference time: {:.2f}s'.format(time.time() - start_time))

        # do posterior sampling
        indices = []
        grasps = []
        for feature, q_value, grasp_candidate in zip(features, q_values, grasp_candidates):
            idx = self.posterior_sample(feature, q_value)
            indices.append(idx)
            if idx is not None:
                grasps.append(grasp_candidate[idx])
            else:
                grasps.append(None)

        features = np.array(features)
        q_values = np.array(q_values)
        indices = np.array(indices)

        # DEBUG: visualzie grasp in full scene
        if False:
            fig, ax = plt.subplots(1)
            ax.imshow(depth_images[0])
            g = grasps[0]
            assert isinstance(g, AntipodalGrasp)
            circ = Circle((g.contact_point1[1], g.contact_point1[0]), 10, color='red')
            ax.add_patch(circ)
            circ = Circle((g.contact_point2[1], g.contact_point2[0]), 10, color='red')
            ax.add_patch(circ)
            plt.show()

        return features, q_values, indices, grasps


if __name__ == "__main__":
    depth_images = np.load('depth_images.npy')
    depth_images = depth_images*-1
    segmasks = np.load('segmasks.npy')

    ts = ThompsonSamplingGraspPlanner()

    for k in range(100):
        print('======= Iter {} ========'.format(k))
        f, q, i, g = ts.plan(depth_images, segmasks)  # f (num_envs, num_grasps, 128)

        # update likelihood
        random_reward = np.random.randint(0, 2, len(i))
        success_idx = np.where(random_reward == 1)[0]
        failure_idx = np.where(random_reward == 0)[0]

        success_features = []
        failure_features = []
        for num_env_idx in range(len(f)):
            # if there is no grasp, skip
            if i[num_env_idx] is None:
                continue
            if random_reward[num_env_idx] == 1:
                success_features.append(f[num_env_idx][i[num_env_idx]])
            else:
                failure_features.append(f[num_env_idx][i[num_env_idx]])
        success_features = np.array(success_features)
        failure_features = np.array(failure_features)

        ts.update(success_features, failure_features)
