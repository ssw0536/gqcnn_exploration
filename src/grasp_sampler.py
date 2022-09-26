import cv2
import numpy as np
import matplotlib.pyplot as plt


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

    @property
    def feature_vec(self):
        """Get grasp feature vector.

        Returns:
            numpy.ndarray: grasp feature vector.
        """
        return np.r_[self.contact_point1, self.contact_point2, self.depth]

    @staticmethod
    def from_feature_vec(vec):
        """Create grasp from feature vector.

        Args:
            vec (`numpy.ndarray`): grasp feature vector.
        Returns:
            `AntipodalGrasp`: AntipodalGrasp instance.
        """
        contact_point1 = vec[0:2]
        contact_point2 = vec[2:4]
        depth = vec[4]
        axis_vector = (contact_point2 - contact_point1) / np.linalg.norm(contact_point2 - contact_point1)
        grasp = AntipodalGrasp(contact_point1, contact_point2, axis_vector, depth)
        return grasp

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


class AntipodalGraspSampler(object):
    def __init__(self, config):
        """Antipodal grasp sampler.

        Args:
            config (dict): configuration dictionary.
        """
        # convert max gripper width from m to pixel
        max_gripper_width = config['max_gripper_width']
        camera_matrix = np.array([[525.0, 0.0, 320.0], [0.0, 525.0, 240.0], [0.0, 0.0, 1.0]])
        max_grasp_point1_3d = [max_gripper_width, 0, 0.75]
        max_grasp_point2_3d = [0, 0, 0.75]
        max_grasp_point1_image = np.dot(camera_matrix, max_grasp_point1_3d) / 0.75
        max_grasp_point2_image = np.dot(camera_matrix, max_grasp_point2_3d) / 0.75
        max_gripper_width = np.linalg.norm(max_grasp_point2_image - max_grasp_point1_image)

        self.max_gripper_width = max_gripper_width
        self.depth_laplacian_threshold = config['depth_laplacian_threshold']
        self.antipodal_threshold_angle = config['antipodal_threshold_angle']
        self.max_num_sample = config['max_num_sample']
        self.num_depth_sample = config['num_depth_sample']
        self.depth_offset_min = config['depth_offset_min']
        self.depth_offset_max = config['depth_offset_max']

    def sample(self, depth_image, segmask_image):
        """Image based antipodal sampler.
        The performance of this fuction depends on the edge detection algorithm.

        Args:
            depth_image (numpy.ndarray): HxW depth images in [m].
            segmask_image (numpy.ndarray): HxW object segment mask. Value of the mask pixel is 1, others 0.

        Returns:
            lsit: Grasp candidates in list. Each element is `AntipodalGrasp` instance.
        """
        # deep copy the images
        depth_img = depth_image.copy()
        segmask_img = segmask_image.copy()

        # get edge using Laplacian with depth
        segmask_img = segmask_img.astype('uint8')
        depth_image_downsampled = cv2.resize(depth_img, (0, 0), fx=0.5, fy=0.5)
        segmask_image_downsampled = cv2.resize(segmask_img, (0, 0), fx=0.5, fy=0.5)
        depth_laplacian = cv2.Laplacian(depth_image_downsampled, cv2.CV_64F, ksize=5)
        depth_laplacian = np.abs(depth_laplacian)
        depth_edge_image_laplacian = np.zeros(depth_image_downsampled.shape).astype('uint8')
        depth_edge_image_laplacian[depth_laplacian > self.depth_laplacian_threshold] = 255

        # get edge pixels
        edge_image = depth_edge_image_laplacian * segmask_image_downsampled
        edge_image[edge_image > 0] = 255
        edge_pixel_y, edge_pixel_x = np.where(edge_image == 255)
        edge_pixel_y = edge_pixel_y * 2
        edge_pixel_x = edge_pixel_x * 2

        # # get edge using Laplacian with depth
        # depth_laplacian = cv2.Laplacian(depth_img, cv2.CV_64F, ksize=5)
        # depth_laplacian = np.abs(depth_laplacian)
        # depth_edge_image_laplacian = np.zeros(depth_img.shape).astype('uint8')
        # depth_edge_image_laplacian[depth_laplacian > self.depth_laplacian_threshold] = 255

        # # get edge pixels
        # edge_image = depth_edge_image_laplacian * segmask_img
        # edge_image[edge_image > 0] = 255
        # edge_pixel_y, edge_pixel_x = np.where(edge_image == 255)

        # get surface nomal using gradient
        grad_image = np.copy(depth_img)
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

        # get antipodal normals
        threshold_deg = self.antipodal_threshold_angle
        normal_inner_products = np.dot(unit_normals, np.transpose(unit_normals))
        distance_matrix = np.linalg.norm(
            edge_pixels[:, None, :] - edge_pixels[None, :, :],
            axis=-1,
            )

        candidate_index = np.where(
            (normal_inner_products < -np.cos(threshold_deg*np.pi/180.0)) &
            (distance_matrix < self.max_gripper_width)
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
        if candidate_index.shape[0] < self.max_num_sample:
            num_sample = candidate_index.shape[0]
        else:
            num_sample = self.max_num_sample
        num_sample_per_depth = int(np.ceil(num_sample / self.num_depth_sample))

        rand_indicies = np.random.choice(candidate_index, num_sample_per_depth, replace=False)
        for rand_index in rand_indicies:
            pair1_index = pair1[rand_index]
            pair2_index = pair2[rand_index]
            contact_pixel1 = edge_pixels[pair1_index, :]
            contact_pixel2 = edge_pixels[pair2_index, :]
            center_pixel = ((contact_pixel1 + contact_pixel2)/2.0).astype('uint16')
            raw_depth = depth_img[center_pixel[0], center_pixel[1]]

            depth_offset_samples = np.random.rand(self.num_depth_sample) * (self.depth_offset_max - self.depth_offset_min) + self.depth_offset_min
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
        return grasp_candidates
