import os
import sys
from importlib.util import spec_from_file_location, module_from_spec

import numpy as np
import torch


class GQCNN(object):
    def __init__(self):
        # TODO: model dir as parameter
        # load gqcnn
        root_dir = '/media/sungwon/WorkSpace/projects/feature_space_analysis'
        model_name = '2022-09-20-1315'
        # model_name = '2022-09-19-2011'
        # model_name = '2022-08-16-2023'

        spec = spec_from_file_location('model.GQCNN', os.path.join(root_dir, 'models', model_name, 'model.py'))
        module = module_from_spec(spec)
        if 'model.GQCNN' in sys.modules:
            del sys.modules['model.GQCNN']
        sys.modules['model.GQCNN'] = module
        spec.loader.exec_module(module)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = module.GQCNN()
        self.model.load_state_dict(torch.load(os.path.join(root_dir, 'models', model_name, 'model.pt')))
        # self.model.load_state_dict(torch.load(os.path.join(root_dir, 'models', model_name, 'model_ft.pt')))
        self.model.to(self.device)

        self.im_mean = np.load(os.path.join(root_dir, 'models', model_name, 'im_mean.npy'))
        self.im_std = np.load(os.path.join(root_dir, 'models', model_name, 'im_std.npy'))
        self.pose_mean = np.load(os.path.join(root_dir, 'models', model_name, 'pose_mean.npy'))
        self.pose_std = np.load(os.path.join(root_dir, 'models', model_name, 'pose_std.npy'))

        # register activation hook
        self.activations = {}
        try:
            self.model.merge_stream[-2].register_forward_hook(self.get_activation(self.activations, '(n-1)layer'))
        except:
            self.model.feature_fc_stack[-2].register_forward_hook(self.get_activation(self.activations, '(n-1)layer'))

        # define optimizer and loss
        self.loss_fn = torch.nn.CrossEntropyLoss()
        params = list(self.model.merge_stream.parameters())
        self.optimizer = torch.optim.Adam(params, lr=0.0005)

    @staticmethod
    def get_activation(activations, name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    def __call__(self, image_tensor, pose_tensor):
        """Evaluate GQCNN on multiple images and poses.

        Args:
            image_tensor (numpy.ndarray): [N, H, W, 1], image tensor.
            pose_tensor (numpy.ndarray): [N, 1], pose tensor.

        Returns:
            numpy.ndarray: [N, 1], predicted quality scores.
            numpy.ndarray: [N, 128], extracted features.
        """
        self.model.train(False)

        # normalize image
        image_tensor = (image_tensor - self.im_mean) / self.im_std
        pose_tensor = (pose_tensor - self.pose_mean) / self.pose_std

        # numpy to tensor
        image_tensor = torch.from_numpy(image_tensor).to(self.device).view(-1, 1, 32, 32)
        pose_tensor = torch.from_numpy(pose_tensor).to(self.device)

        # get result
        with torch.no_grad():
            output = self.model(image_tensor, pose_tensor)
            q_value = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
            feature = self.activations['(n-1)layer'].cpu().numpy()
            feature = feature/np.linalg.norm(feature, axis=1, keepdims=True)

        return q_value, feature

    def fine_tune(self, image_tensor, pose_tensor, label_tensor):
        # normalize image
        image_tensor = (image_tensor - self.im_mean) / self.im_std
        pose_tensor = (pose_tensor - self.pose_mean) / self.pose_std

        # numpy to tensor
        image_tensor = torch.from_numpy(image_tensor).to(self.device).view(-1, 1, 32, 32)
        pose_tensor = torch.from_numpy(pose_tensor).to(self.device)
        label_tensor = torch.from_numpy(label_tensor).to(self.device)

        # train
        output = self.model(image_tensor, pose_tensor)
        loss = self.loss_fn(output, label_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


if __name__ == '__main__':
    gqcnn = GQCNN()
    batch_size = 10
    image_tensor = np.zeros((batch_size, 32, 32, 1), dtype=np.float32)
    pose_tensor = np.zeros((batch_size, 1), dtype=np.float32)
    label_tensor = np.zeros((batch_size, 2), dtype=np.float32)
    label_tensor[:, 1] = 1

    for i in range(100):
        loss = gqcnn.fine_tune(image_tensor, pose_tensor, label_tensor)
        print(loss)
        print(gqcnn.model.merge_stream[0].weight)

