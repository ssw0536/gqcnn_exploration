import time
import copy
from multiprocessing import Pool

from isaacgym import gymapi

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.mixture import GaussianMixture

from src.grasp_sampler import AntipodalGrasp


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


class CrossEntropyGraspingPolicy(object):
    def __init__(self, model, sampler, config):
        self.model = model
        self.sampler = sampler

        config = config['cross_entropy']
        self.num_seed_samples = config['num_seed_samples']
        self.num_gmm_samples = config['num_gmm_samples']
        self.num_iters = config['num_iters']
        self.gmm_refit_p = config['gmm_refit_p']
        self.gmm_component_frac = config['gmm_component_frac']
        self.gmm_reg_covar = config['gmm_reg_covar']
        self.feature_dim = config['feature_dim']
        self.gmm_max_resample_trial = self.num_gmm_samples * 10

        # re-parameterize the sampler
        self.sampler.max_num_sample = self.num_seed_samples

    def optimize_sample(self, depth_images, segmasks, verbose=False):
        depth_image_width = depth_images.shape[2]
        depth_image_height = depth_images.shape[1]

        # get batch size
        batch_size = depth_images.shape[0]

        # get initial seed samples
        start_time = time.time()
        grasp_candidates = []
        for i in range(batch_size):
            grasp_candidates.append(
                self.sampler.sample(depth_images[i], segmasks[i]))
        if verbose:
            print('Get antipodal grasp time: {:.2f}s'.format(time.time() - start_time))

        # debug
        print(len(grasp_candidates[i]))

        # get grasp candidates
        for _ in range(self.num_iters):
            if verbose:
                print('**Iteration: {}'.format(_))
                grasp_num = 0
                for g in grasp_candidates:
                    grasp_num += len(g)
                print('Average grasp candidates: {}'.format(grasp_num/batch_size))

            # convert grasp candidates to tensors
            start_time = time.time()
            im_tensors, depth_tensors = [], []
            grasp_to_tensor(depth_images[0], grasp_candidates[0])
            with Pool(12) as p:
                temp = p.starmap(grasp_to_tensor, zip(depth_images, grasp_candidates))
            for i in range(batch_size):
                im_tensors.append(temp[i][0])
                depth_tensors.append(temp[i][1])
            if verbose:
                print('Convert grasp to tensor time: {:.2f}s'.format(time.time() - start_time))

            # get grasp quality
            start_time = time.time()
            q_values = []
            with torch.no_grad():
                for i in range(batch_size):
                    q_value, _ = self.model(im_tensors[i], depth_tensors[i])
                    q_values.append(q_value)
                    if len(grasp_candidates[i]) == 0:
                        print('No grasp candidates on {}-th env'.format(i))
            if verbose:
                print('GQ-CNN inference time: {:.2f}s'.format(time.time() - start_time))

            # sort grasp candidates
            elite_grasp_candidates = []
            for i in range(batch_size):
                if len(grasp_candidates[i]) == 0:
                    elite_grasp_candidates.append([])
                    continue

                sorted_index = np.argsort(q_values[i])[::-1]
                num_elite = max(int(np.ceil(self.gmm_refit_p * len(grasp_candidates[i]))), 1)

                # debug
                if i == 0:
                    elite_q_value = q_values[i][sorted_index[:num_elite]]
                    print('Elite q values: {}'.format(elite_q_value[:5]))

                elite_index = sorted_index[:num_elite]
                elite_grasp = [grasp_candidates[i][j] for j in elite_index]
                elite_grasp_feature = np.array([g.feature_vec for g in elite_grasp])

                # normalize elite set
                elite_grasp_mean = np.mean(elite_grasp_feature, axis=0)
                elite_grasp_std = np.std(elite_grasp_feature, axis=0)
                elite_grasp_std[elite_grasp_std == 0] = 1e-6
                elite_grasp_feature = (elite_grasp_feature - elite_grasp_mean) / elite_grasp_std

                # fit GMM with elite samples
                num_components = max(int(np.ceil(self.gmm_component_frac * num_elite)), 1)
                uniform_weights = (1.0 / num_components) * np.ones(num_components)
                gmm = GaussianMixture(
                    n_components=num_components,
                    weights_init=uniform_weights,
                    reg_covar=self.gmm_reg_covar,
                    )
                gmm.fit(elite_grasp_feature)

                # sample from GMM
                elite_grasp_candidate = []
                num_try = 0
                while (len(elite_grasp_candidate) < self.num_gmm_samples) and (num_try < self.gmm_max_resample_trial):
                    grasp_features, _ = gmm.sample(n_samples=self.num_gmm_samples)
                    grasp_features = elite_grasp_std * grasp_features + elite_grasp_mean

                    # check bound
                    for grasp_feature in grasp_features:
                        grasp = AntipodalGrasp.from_feature_vec(grasp_feature)

                        # check grasp center is in the image bound
                        if (grasp.center_point[0] < 0) or (grasp.center_point[0] > depth_image_height):  # y, H
                            break
                        if (grasp.center_point[1] < 0) or (grasp.center_point[1] > depth_image_width):  # x, W
                            break

                        # if segmask is None keep the grasp(do no check bound)
                        if segmasks is None:
                            elite_grasp_candidate.append(grasp)
                        elif segmasks[i][grasp.center_point[0], grasp.center_point[1]] != 0:
                            elite_grasp_candidate.append(grasp)
                    num_try += 1
                elite_grasp_candidates.append(elite_grasp_candidate)

            # update grasp candidates
            grasp_candidates = elite_grasp_candidates
        return elite_grasp_candidates

    def action(self, depth_images, segmasks, verbose=False):
        """Return a list of grasp candidate for each env.

        Args:
            depth_images (numpy.ndarray): BxHxW depth image.
            segmasks (numpy.ndarray): BxHxW segmentation mask.
            verbose (bool, optional): Defaults to False.

        Returns:
            list: list of grasps. If no grasp is found, return `None` for corresponding env.
            list: list of grasp quality. If no grasp is found, return `0` for corresponding env.
            list: list of image tensors. If no grasp is found, return `zero image` for corresponding env.
            list: list of depth tensors. If no grasp is found, return `zero depth` for corresponding env.
            list: list of features. If no grasp is found, return `zero feature` for corresponding env.
        """
        batch_size = depth_images.shape[0]

        # get cem optimized grasp candidates
        grasp_candidates = self.optimize_sample(depth_images, segmasks, verbose=verbose)

        # get grasp candidates
        start_time = time.time()
        im_tensors, depth_tensors = [], []
        grasp_to_tensor(depth_images[0], grasp_candidates[0])
        with Pool(12) as p:
            temp = p.starmap(grasp_to_tensor, zip(depth_images, grasp_candidates))
        for i in range(batch_size):
            im_tensors.append(temp[i][0])
            depth_tensors.append(temp[i][1])
        if verbose:
            print('Convert grasp to tensor time: {:.2f}s'.format(time.time() - start_time))

        # get grasp quality
        start_time = time.time()
        q_values = []
        features = []
        with torch.no_grad():
            for i in range(batch_size):
                q_value, feature = self.model(im_tensors[i], depth_tensors[i])
                q_values.append(q_value)
                features.append(feature)
        if verbose:
            print('GQ-CNN inference time: {:.2f}s'.format(time.time() - start_time))

        # get best grasp index
        best_grasps = []
        best_q_values = []
        best_im_tensors = []
        best_depth_tensors = []
        best_features = []
        best_posteriors = []
        for i in range(batch_size):
            if len(im_tensors[i]) == 0:
                best_grasps.append(None)
                best_q_values.append(0)
                best_im_tensors.append(np.zeros((32, 32, 1)))
                best_depth_tensors.append(np.zeros((1,)))
                best_features.append(np.zeros((self.feature_dim,)))
                best_posteriors.append(0)
            else:
                best_index = np.argmax(q_values[i])
                best_grasps.append(grasp_candidates[i][best_index])
                best_q_values.append(q_values[i][best_index])
                best_im_tensors.append(im_tensors[i][best_index])
                best_depth_tensors.append(depth_tensors[i][best_index])
                best_features.append(features[i][best_index])
                best_posteriors.append(0)
        return (
            best_grasps,
            best_q_values,
            best_im_tensors,
            best_depth_tensors,
            best_features,
            best_posteriors)


class PosteriorGraspingPolicy(object):
    def __init__(self, model, sampler, config):
        self.model = model
        self.sampler = sampler

        config = config['posterior']
        self.prior_strength = config['prior_strength']
        self.prior_disc_coef = config['prior_discouraging_coef']
        self.likelihood_strength = config['likelihood_strength']
        self.cos_sim_thres = config['cosine_similarity_threshold']
        self.buffer_size = config['buffer_size']
        self.feature_dim = config['feature_dim']

        if self.buffer_size == 'inf':
            self.buffer_size = np.inf

        self.feature_history = np.zeros((0, self.feature_dim))
        self.reward_history = np.zeros((0, 1))

    def update(self, feature, reward):
        """Update the posterior policy.

        Args:
            feature (numpy.ndarray): (N, self.feature_dim) feature vector.
            reward (numpy.ndarray): (N, 1) reward.
        """
        if len(np.shape(feature)) == 1:
            feature = np.expand_dims(feature, axis=0)
            reward = np.expand_dims(reward, axis=0)

        if len(self.feature_history) == 0:
            self.feature_history = feature
            self.reward_history = reward
        else:
            self.feature_history = np.r_[self.feature_history, feature]
            self.reward_history = np.r_[self.reward_history, reward]

        # remove old samples
        if not self.buffer_size == np.inf:
            self.feature_history = self.feature_history[-self.buffer_size:]
            self.reward_history = self.reward_history[-self.buffer_size:]

    def posterior_sample(self, features, q_values):
        """Sample from the posterior policy.

        Args:
            features (numpy.ndarray): (N, self.feature_dim) feature vector.
            q_values (numpy.ndarray): (N, ) q value.
        """
        # if there are no samples, return None
        if len(q_values) == 0:
            return None, None

        # get success and failure features
        success_features = self.feature_history[self.reward_history == 1]
        failure_features = self.feature_history[self.reward_history == 0]

        # get likelihood
        if len(success_features) == 0:
            n = np.zeros(features.shape[0])
        else:
            pos_dist = np.dot(features, success_features.T)
            n = np.count_nonzero(pos_dist > self.cos_sim_thres, axis=1).astype(np.float32)
            n *= self.likelihood_strength
        if len(failure_features) == 0:
            m = np.zeros(features.shape[0])
        else:
            neg_dist = np.dot(features, failure_features.T)
            m = np.count_nonzero(neg_dist > self.cos_sim_thres, axis=1).astype(np.float32)
            m *= self.likelihood_strength

        # get prior
        # a = self.prior_strength*q_values + self.prior_strength*self.prior_disc_coef 
        # b = self.prior_strength*(1.0 - q_values) + self.prior_strength*self.prior_disc_coef
        q_values *= self.prior_disc_coef
        a = self.prior_strength*q_values + 1e-10
        b = self.prior_strength*(1.0 - q_values) + 1e-10

        # get posterior
        posterior = (a + n) / (a + b + n + m)
        random_posterior = np.random.beta(a + n, b + m)
        idx = np.argmax(random_posterior)

        # print('prios: {:.2f}, {:.2f} | likelihood: {:04.0f}, {:04.0f} | posterior: {:07.2f}, {:07.2f} | prob: {:.2f} | len: {}'.format(
        #     a[idx], b[idx], n[idx], m[idx], (a+n)[idx], (b+m)[idx], random_posterior[idx], len(self.feature_history)))
        return idx, posterior[idx]

    def action(self, depth_images, segmasks, verbose=False):
        """Return a list of grasp candidate for each env.

        Args:
            depth_images (numpy.ndarray): BxHxW depth image.
            segmasks (numpy.ndarray): BxHxW segmentation mask.
            verbose (bool, optional): Defaults to False.

        Returns:
            list: list of grasps. If no grasp is found, return `None` for corresponding env.
            list: list of grasp quality. If no grasp is found, return `0` for corresponding env.
            list: list of image tensors. If no grasp is found, return `zero image` for corresponding env.
            list: list of depth tensors. If no grasp is found, return `zero depth` for corresponding env.
            list: list of features. If no grasp is found, return `zero feature` for corresponding env.
        """
        batch_size = depth_images.shape[0]

        # pool is not efficient for this task
        # get antipodal grasps from  depth images
        start_time = time.time()
        grasp_candidates = []
        for i in range(batch_size):
            grasp_candidates.append(self.sampler.sample(depth_images[i], segmasks[i]))
        if verbose:
            print('Get antipodal grasp time: {:.2f}s'.format(time.time() - start_time))

        # get grasp candidates
        start_time = time.time()
        im_tensors, depth_tensors = [], []
        grasp_to_tensor(depth_images[0], grasp_candidates[0])
        with Pool(12) as p:
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
                q_value, feature = self.model(im_tensor, depth_tensor)
                features.append(feature)
                q_values.append(q_value)
        if verbose:
            print('GQ-CNN inference time: {:.2f}s'.format(time.time() - start_time))

        # get best grasp from the posterior
        best_grasps = []
        best_q_values = []
        best_im_tensors = []
        best_depth_tensors = []
        best_features = []
        best_posteriors = []
        for i in range(batch_size):
            # get features
            features_i = np.array(features[i])
            q_values_i = np.array(q_values[i])

            # get best grasp
            idx, posterior = self.posterior_sample(features_i, q_values_i)
            if idx is None:
                best_grasps.append(None)
                best_q_values.append(0)
                best_im_tensors.append(np.zeros((32, 32, 1)))
                best_depth_tensors.append(np.zeros((1,)))
                best_features.append(np.zeros((self.feature_dim,)))
                best_posteriors.append(0)
            else:
                best_grasps.append(grasp_candidates[i][idx])
                best_q_values.append(q_values_i[idx])
                best_im_tensors.append(im_tensors[i][idx])
                best_depth_tensors.append(depth_tensors[i][idx])
                best_features.append(features_i[idx])
                best_posteriors.append(posterior)
        return (
            best_grasps,
            best_q_values,
            best_im_tensors,
            best_depth_tensors,
            best_features,
            best_posteriors)


class CrossEntorpyPosteriorGraspingPolicy(object):
    def __init__(self, model, sampler, config):
        self.model = model
        self.sampler = sampler

        config = config['cross_entropy_posterior']
        # posterior parameters
        self.prior_strength = config['prior_strength']
        self.prior_disc_coef = config['prior_discouraging_coef']
        self.likelihood_strength = config['likelihood_strength']
        self.cos_sim_thres = config['cosine_similarity_threshold']
        self.buffer_size = config['buffer_size']
        self.feature_dim = config['feature_dim']
        self.random_sample = config['random_sample']

        # cross entropy parameters
        self.num_seed_samples = config['num_seed_samples']
        self.num_gmm_samples = config['num_gmm_samples']
        self.num_iters = config['num_iters']
        self.gmm_refit_p = config['gmm_refit_p']
        self.gmm_component_frac = config['gmm_component_frac']
        self.gmm_reg_covar = config['gmm_reg_covar']
        self.gmm_max_resample_trial = self.num_gmm_samples * 10

        # re-parameterize the sampler
        self.sampler.max_num_sample = self.num_seed_samples

        if self.buffer_size == 'inf':
            self.buffer_size = np.inf

        self.feature_history = np.zeros((0, self.feature_dim))
        self.reward_history = np.zeros((0, 1))

    def update(self, feature, reward):
        """Update the posterior policy.

        Args:
            feature (numpy.ndarray): (N, self.feature_dim) feature vector.
            reward (numpy.ndarray): (N, 1) reward.
        """
        if len(np.shape(feature)) == 1:
            feature = np.expand_dims(feature, axis=0)
            reward = np.expand_dims(reward, axis=0)

        if len(self.feature_history) == 0:
            self.feature_history = feature
            self.reward_history = reward
        else:
            self.feature_history = np.r_[self.feature_history, feature]
            self.reward_history = np.r_[self.reward_history, reward]

        # remove old samples
        if not self.buffer_size == np.inf:
            self.feature_history = self.feature_history[-self.buffer_size:]
            self.reward_history = self.reward_history[-self.buffer_size:]

    def posterior_sample(self, features, q_values):
        """Sample from the posterior policy.

        Args:
            features (numpy.ndarray): (N, self.feature_dim) feature vector.
            q_values (numpy.ndarray): (N, ) q value.
        """
        # if there are no samples, return None
        if len(q_values) == 0:
            return None, None

        # get success and failure features
        success_features = self.feature_history[self.reward_history == 1]
        failure_features = self.feature_history[self.reward_history == 0]

        # get likelihood
        # if len(success_features) == 0:
        #     n = np.zeros(features.shape[0])
        # else:
        #     pos_dist = np.dot(features, success_features.T) # (N, M)
        #     pos_dist = (pos_dist + 1)/2.0
        #     pos_dist = pos_dist ** 10.0
        #     n = np.sum(pos_dist, axis=1)
        #     n *= self.likelihood_strength
        #     # print('pos_dist', pos_dist)
        #     # print('n', n)
        # if len(failure_features) == 0:
        #     m = np.zeros(features.shape[0])
        # else:
        #     neg_dist = np.dot(features, failure_features.T)
        #     neg_dist = (neg_dist + 1)/2.0
        #     neg_dist = neg_dist ** 10.0
        #     m = np.sum(neg_dist, axis=1)
        #     m *= self.likelihood_strength
        #     # print('neg_dist', neg_dist)
        #     # print('m', m)

        # get likelihood
        if len(success_features) == 0:
            n = np.zeros(features.shape[0])
        else:
            pos_dist = np.dot(features, success_features.T)
            n = np.count_nonzero(pos_dist > self.cos_sim_thres, axis=1).astype(np.float32)
            n *= self.likelihood_strength
        if len(failure_features) == 0:
            m = np.zeros(features.shape[0])
        else:
            neg_dist = np.dot(features, failure_features.T)
            m = np.count_nonzero(neg_dist > self.cos_sim_thres, axis=1).astype(np.float32)
            m *= self.likelihood_strength

        # get prior
        q_values *= self.prior_disc_coef
        a = self.prior_strength*q_values + 1e-10
        b = self.prior_strength*(1.0 - q_values) + 1e-10

        # get posterior
        posterior = (a + n) / (a + b + n + m)
        if self.random_sample:
            random_posterior = np.random.beta(a + n, b + m)
            idx = np.argmax(random_posterior)
        else:
            idx = np.argmax(posterior)

        # print('prios: {:.2f}, {:.2f} | likelihood: {:04.0f}, {:04.0f} | posterior: {:07.2f}, {:07.2f} | prob: {:.2f} | len: {}'.format(
        #     a[idx], b[idx], n[idx], m[idx], (a+n)[idx], (b+m)[idx], posterior[idx], len(self.feature_history)))
        return idx, posterior[idx]

    def compute_posteriors(self, features, q_values):
        # if there are no samples, return None
        if len(q_values) == 0:
            return []

        # get success and failure features
        success_features = self.feature_history[self.reward_history == 1]
        failure_features = self.feature_history[self.reward_history == 0]

        # get likelihood
        if len(success_features) == 0:
            n = np.zeros(features.shape[0])
        else:
            pos_dist = np.dot(features, success_features.T)
            n = np.count_nonzero(pos_dist > self.cos_sim_thres, axis=1).astype(np.float32)
            n *= self.likelihood_strength
        if len(failure_features) == 0:
            m = np.zeros(features.shape[0])
        else:
            neg_dist = np.dot(features, failure_features.T)
            m = np.count_nonzero(neg_dist > self.cos_sim_thres, axis=1).astype(np.float32)
            m *= self.likelihood_strength

        # get prior
        # a = self.prior_strength*q_values + self.prior_strength*self.prior_disc_coef 
        # b = self.prior_strength*(1.0 - q_values) + self.prior_strength*self.prior_disc_coef
        q_values *= self.prior_disc_coef
        a = self.prior_strength*q_values + 1e-10
        b = self.prior_strength*(1.0 - q_values) + 1e-10

        posterior = (a + n) / (a + b + n + m)

        # idx = np.argmax(posterior)
        # print('prios: {:.2f}, {:.2f} | likelihood: {:04.0f}, {:04.0f} | posterior: {:07.2f}, {:07.2f} | prob: {:.2f} | len: {}'.format(
        #     a[idx], b[idx], n[idx], m[idx], (a+n)[idx], (b+m)[idx], posterior[idx], len(self.feature_history)))
        return posterior

    def optimize_sample(self, depth_images, segmasks, verbose=False):
        depth_image_width = depth_images.shape[2]
        depth_image_height = depth_images.shape[1]

        # get batch size
        batch_size = depth_images.shape[0]

        # get initial seed samples
        start_time = time.time()
        grasp_candidates = []
        for i in range(batch_size):
            grasp_candidates.append(
                self.sampler.sample(depth_images[i], segmasks[i]))
        if verbose:
            print('Get antipodal grasp time: {:.2f}s'.format(time.time() - start_time))

        # get grasp candidates
        for _ in range(self.num_iters):
            if verbose:
                print('**Iteration: {}'.format(_))
                grasp_num = 0
                for g in grasp_candidates:
                    grasp_num += len(g)
                print('Average grasp candidates: {}'.format(grasp_num/batch_size))

            # convert grasp candidates to tensors
            start_time = time.time()
            im_tensors, depth_tensors = [], []
            grasp_to_tensor(depth_images[0], grasp_candidates[0])
            with Pool(12) as p:
                temp = p.starmap(grasp_to_tensor, zip(depth_images, grasp_candidates))
            for i in range(batch_size):
                im_tensors.append(temp[i][0])
                depth_tensors.append(temp[i][1])
            if verbose:
                print('Convert grasp to tensor time: {:.2f}s'.format(time.time() - start_time))

            # get grasp quality
            start_time = time.time()
            q_values = []
            features = []
            with torch.no_grad():
                for i in range(batch_size):
                    q_value, feature = self.model(im_tensors[i], depth_tensors[i])
                    q_values.append(q_value)
                    features.append(feature)
                    if len(grasp_candidates[i]) == 0:
                        print('No grasp candidates on {}-th env'.format(i))
            if verbose:
                print('GQ-CNN inference time: {:.2f}s'.format(time.time() - start_time))

            # update posterior
            posteriors = []
            for i in range(batch_size):
                posterior = self.compute_posteriors(features[i], q_values[i])
                posteriors.append(posterior)

            # sort grasp candidates
            elite_grasp_candidates = []
            for i in range(batch_size):
                if len(grasp_candidates[i]) == 0:
                    elite_grasp_candidates.append([])
                    continue

                sorted_index = np.argsort(posteriors[i])[::-1]
                num_elite = max(int(np.ceil(self.gmm_refit_p * len(grasp_candidates[i]))), 1)

                # debug
                elite_posteriors = posteriors[i][sorted_index[:num_elite]]
                if i == 0:
                    print('Elite posterior: {}'.format(elite_posteriors[:5]))

                elite_index = sorted_index[:num_elite]
                elite_grasp = [grasp_candidates[i][j] for j in elite_index]
                elite_grasp_feature = np.array([g.feature_vec for g in elite_grasp])

                # normalize elite set
                elite_grasp_mean = np.mean(elite_grasp_feature, axis=0)
                elite_grasp_std = np.std(elite_grasp_feature, axis=0)
                elite_grasp_std[elite_grasp_std == 0] = 1e-6
                elite_grasp_feature = (elite_grasp_feature - elite_grasp_mean) / elite_grasp_std

                # fit GMM with elite samples
                num_components = max(int(np.ceil(self.gmm_component_frac * num_elite)), 1)
                uniform_weights = (1.0 / num_components) * np.ones(num_components)
                gmm = GaussianMixture(
                    n_components=num_components,
                    weights_init=uniform_weights,
                    reg_covar=self.gmm_reg_covar,
                    )
                gmm.fit(elite_grasp_feature)

                # sample from GMM
                elite_grasp_candidate = []
                num_try = 0
                while (len(elite_grasp_candidate) < self.num_gmm_samples) and (num_try < self.gmm_max_resample_trial):
                    grasp_features, _ = gmm.sample(n_samples=self.num_gmm_samples)
                    grasp_features = elite_grasp_std * grasp_features + elite_grasp_mean

                    # check bound
                    for grasp_feature in grasp_features:
                        grasp = AntipodalGrasp.from_feature_vec(grasp_feature)

                        # check grasp center is in the image bound
                        if (grasp.center_point[0] < 0) or (grasp.center_point[0] > depth_image_height):  # y, H
                            break
                        if (grasp.center_point[1] < 0) or (grasp.center_point[1] > depth_image_width):  # x, W
                            break

                        # if segmask is None keep the grasp(do no check bound)
                        if segmasks is None:
                            elite_grasp_candidate.append(grasp)
                        elif segmasks[i][grasp.center_point[0], grasp.center_point[1]] != 0:
                            elite_grasp_candidate.append(grasp)
                    num_try += 1
                elite_grasp_candidates.append(elite_grasp_candidate)

            # update grasp candidates
            grasp_candidates = elite_grasp_candidates
        return elite_grasp_candidates

    def action(self, depth_images, segmasks, verbose=False):
        """Return a list of grasp candidate for each env.

        Args:
            depth_images (numpy.ndarray): BxHxW depth image.
            segmasks (numpy.ndarray): BxHxW segmentation mask.
            verbose (bool, optional): Defaults to False.

        Returns:
            list: list of grasps. If no grasp is found, return `None` for corresponding env.
            list: list of grasp quality. If no grasp is found, return `0` for corresponding env.
            list: list of image tensors. If no grasp is found, return `zero image` for corresponding env.
            list: list of depth tensors. If no grasp is found, return `zero depth` for corresponding env.
            list: list of features. If no grasp is found, return `zero feature` for corresponding env.
        """
        batch_size = depth_images.shape[0]

        # get cem optimized grasp candidates
        start_time = time.time()
        grasp_candidates = self.optimize_sample(depth_images, segmasks, verbose=verbose)

        # get grasp candidates
        start_time = time.time()
        im_tensors, depth_tensors = [], []
        grasp_to_tensor(depth_images[0], grasp_candidates[0])
        with Pool(12) as p:
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
                q_value, feature = self.model(im_tensor, depth_tensor)
                features.append(feature)
                q_values.append(q_value)
        if verbose:
            print('GQ-CNN inference time: {:.2f}s'.format(time.time() - start_time))

        # get best grasp from the posterior
        best_grasps = []
        best_q_values = []
        best_im_tensors = []
        best_depth_tensors = []
        best_features = []
        best_posteriors = []
        for i in range(batch_size):
            # get features
            features_i = np.array(features[i])
            q_values_i = np.array(q_values[i])

            # get best grasp
            idx, posterior = self.posterior_sample(features_i, q_values_i)
            if idx is None:
                best_grasps.append(None)
                best_q_values.append(0)
                best_im_tensors.append(np.zeros((32, 32, 1)))
                best_depth_tensors.append(np.zeros((1,)))
                best_features.append(np.zeros((self.feature_dim,)))
                best_posteriors.append(0)
            else:
                best_grasps.append(grasp_candidates[i][idx])
                best_q_values.append(q_values_i[idx])
                best_im_tensors.append(im_tensors[i][idx])
                best_depth_tensors.append(depth_tensors[i][idx])
                best_features.append(features_i[idx])
                best_posteriors.append(posterior)
        return (
            best_grasps,
            best_q_values,
            best_im_tensors,
            best_depth_tensors,
            best_features,
            best_posteriors)


class QLearningGraspingPolicy(object):
    def __init__(self, posterior_model, prior_model, sampler, config):
        self.posterior_model = posterior_model
        self.sampler = sampler
        self.prior_model = prior_model

        config = config['q_learning']
        # posterior parameters
        self.num_seed_samples = config['num_seed_samples']
        self.num_gmm_samples = config['num_gmm_samples']
        self.num_iters = config['num_iters']
        self.gmm_refit_p = config['gmm_refit_p']
        self.gmm_component_frac = config['gmm_component_frac']
        self.gmm_reg_covar = config['gmm_reg_covar']
        self.feature_dim = config['feature_dim']
        self.gmm_max_resample_trial = self.num_gmm_samples * 10
        # re-parameterize the sampler
        self.sampler.max_num_sample = self.num_seed_samples

        # q-learing paramters
        # self.learning_rate = config['learning_rate']
        # self.target_update_interval = config['target_update_interval']
        self.mini_epochs = config['mini_epochs']
        self.buffer_size = config['buffer_size']
        self.batch_size = config['batch_size']

        # initialize
        self.image_tensors = None
        self.depth_tensors = None
        self.label_tensors = None

    def optimize_sample(self, depth_images, segmasks, verbose=False):
        depth_image_width = depth_images.shape[2]
        depth_image_height = depth_images.shape[1]

        # get batch size
        batch_size = depth_images.shape[0]

        # get initial seed samples
        start_time = time.time()
        grasp_candidates = []
        for i in range(batch_size):
            grasp_candidates.append(
                self.sampler.sample(depth_images[i], segmasks[i]))
        if verbose:
            print('Get antipodal grasp time: {:.2f}s'.format(time.time() - start_time))

        # get grasp candidates
        for _ in range(self.num_iters):
            if verbose:
                print('**Iteration: {}'.format(_))
                grasp_num = 0
                for g in grasp_candidates:
                    grasp_num += len(g)
                print('Average grasp candidates: {}'.format(grasp_num/batch_size))

            # convert grasp candidates to tensors
            start_time = time.time()
            im_tensors, depth_tensors = [], []
            grasp_to_tensor(depth_images[0], grasp_candidates[0])
            with Pool(12) as p:
                temp = p.starmap(grasp_to_tensor, zip(depth_images, grasp_candidates))
            for i in range(batch_size):
                im_tensors.append(temp[i][0])
                depth_tensors.append(temp[i][1])
            if verbose:
                print('Convert grasp to tensor time: {:.2f}s'.format(time.time() - start_time))

            # get grasp quality
            start_time = time.time()
            q_values = []
            with torch.no_grad():
                for i in range(batch_size):
                    q_value, _ = self.posterior_model(im_tensors[i], depth_tensors[i])
                    q_values.append(q_value)
                    if len(grasp_candidates[i]) == 0:
                        print('No grasp candidates on {}-th env'.format(i))
            if verbose:
                print('GQ-CNN inference time: {:.2f}s'.format(time.time() - start_time))

            # sort grasp candidates
            elite_grasp_candidates = []
            for i in range(batch_size):
                if len(grasp_candidates[i]) == 0:
                    elite_grasp_candidates.append([])
                    continue

                sorted_index = np.argsort(q_values[i])[::-1]
                num_elite = max(int(np.ceil(self.gmm_refit_p * len(grasp_candidates[i]))), 1)

                # debug
                if i == 0:
                    elite_q_value = q_values[i][sorted_index[:num_elite]]
                    print('Elite q values: {}'.format(elite_q_value[:5]))

                elite_index = sorted_index[:num_elite]
                elite_grasp = [grasp_candidates[i][j] for j in elite_index]
                elite_grasp_feature = np.array([g.feature_vec for g in elite_grasp])

                # normalize elite set
                elite_grasp_mean = np.mean(elite_grasp_feature, axis=0)
                elite_grasp_std = np.std(elite_grasp_feature, axis=0)
                elite_grasp_std[elite_grasp_std == 0] = 1e-6
                elite_grasp_feature = (elite_grasp_feature - elite_grasp_mean) / elite_grasp_std

                # fit GMM with elite samples
                num_components = max(int(np.ceil(self.gmm_component_frac * num_elite)), 1)
                uniform_weights = (1.0 / num_components) * np.ones(num_components)
                gmm = GaussianMixture(
                    n_components=num_components,
                    weights_init=uniform_weights,
                    reg_covar=self.gmm_reg_covar,
                    )
                gmm.fit(elite_grasp_feature)

                # sample from GMM
                elite_grasp_candidate = []
                num_try = 0
                while (len(elite_grasp_candidate) < self.num_gmm_samples) and (num_try < self.gmm_max_resample_trial):
                    grasp_features, _ = gmm.sample(n_samples=self.num_gmm_samples)
                    grasp_features = elite_grasp_std * grasp_features + elite_grasp_mean

                    # check bound
                    for grasp_feature in grasp_features:
                        grasp = AntipodalGrasp.from_feature_vec(grasp_feature)

                        # check grasp center is in the image bound
                        if (grasp.center_point[0] < 0) or (grasp.center_point[0] > depth_image_height):  # y, H
                            break
                        if (grasp.center_point[1] < 0) or (grasp.center_point[1] > depth_image_width):  # x, W
                            break

                        # if segmask is None keep the grasp(do no check bound)
                        if segmasks is None:
                            elite_grasp_candidate.append(grasp)
                        elif segmasks[i][grasp.center_point[0], grasp.center_point[1]] != 0:
                            elite_grasp_candidate.append(grasp)
                    num_try += 1
                elite_grasp_candidates.append(elite_grasp_candidate)

            # update grasp candidates
            grasp_candidates = elite_grasp_candidates
        return elite_grasp_candidates

    def action(self, depth_images, segmasks, verbose=False):
        batch_size = depth_images.shape[0]

        # get cem optimized grasp candidates
        grasp_candidates = self.optimize_sample(depth_images, segmasks, verbose=verbose)

        # get grasp candidates
        start_time = time.time()
        im_tensors, depth_tensors = [], []
        grasp_to_tensor(depth_images[0], grasp_candidates[0])
        with Pool(12) as p:
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
                q_value, feature = self.posterior_model(im_tensor, depth_tensor)
                features.append(feature)
                q_values.append(q_value)
        if verbose:
            print('GQ-CNN inference time: {:.2f}s'.format(time.time() - start_time))

        # get priors
        priors = []
        with torch.no_grad():
            for im_tensor, depth_tensor in zip(im_tensors, depth_tensors):
                q_value, _ = self.prior_model(im_tensor, depth_tensor)
                priors.append(q_value)

        # get best grasp index
        best_grasps = []
        best_q_values = []
        best_im_tensors = []
        best_depth_tensors = []
        best_features = []
        best_posteriors = []
        for i in range(batch_size):
            if len(im_tensors[i]) == 0:
                best_grasps.append(None)
                best_q_values.append(0)
                best_im_tensors.append(np.zeros((32, 32, 1)))
                best_depth_tensors.append(np.zeros((1,)))
                best_features.append(np.zeros((self.feature_dim,)))
                best_posteriors.append(0)
            else:
                best_index = np.argmax(q_values[i])
                best_grasps.append(grasp_candidates[i][best_index])
                best_q_values.append(priors[i][best_index])
                best_im_tensors.append(im_tensors[i][best_index])
                best_depth_tensors.append(depth_tensors[i][best_index])
                best_features.append(features[i][best_index])
                best_posteriors.append(q_values[i][best_index])
        return (
            best_grasps,
            best_q_values,
            best_im_tensors,
            best_depth_tensors,
            best_features,
            best_posteriors)

    def update(self, image_tensors, pose_tensors, rewards):
        # one-hot encoding
        label_tensors = np.zeros((len(rewards), 2), dtype=np.float32)
        label_tensors[:, 0] = 1 - rewards
        label_tensors[:, 1] = rewards

        if self.image_tensors is None:
            self.image_tensors = image_tensors
            self.pose_tensors = pose_tensors
            self.label_tensors = label_tensors
        else:
            self.image_tensors = np.concatenate((self.image_tensors, image_tensors), axis=0)
            self.pose_tensors = np.concatenate((self.pose_tensors, pose_tensors), axis=0)
            self.label_tensors = np.concatenate((self.label_tensors, label_tensors), axis=0)

        for epoch in range(self.mini_epochs):
            # shuffle
            indices = np.arange(self.image_tensors.shape[0])
            np.random.shuffle(indices)
            self.image_tensors = self.image_tensors[indices]
            self.pose_tensors = self.pose_tensors[indices]
            self.label_tensors = self.label_tensors[indices]

            # train
            avg_loss = 0
            for j in range(0, self.image_tensors.shape[0], self.batch_size):
                image_batch = self.image_tensors[j:j + self.batch_size]
                pose_batch = self.pose_tensors[j:j + self.batch_size]
                label_batch = self.label_tensors[j:j + self.batch_size]
                loss = self.posterior_model.fine_tune(image_batch, pose_batch, label_batch)
                avg_loss += loss
            avg_loss /= len(image_batch)
        print(avg_loss)
        return avg_loss

    def save(self, model_path):
        torch.save(self.posterior_model.model.state_dict(), model_path)
        print('Saved model to {}'.format(model_path))
