# import isaacgym modules
from isaacgym import gymapi

# import 3rd party modules
import numpy as np
from scipy.spatial.transform import Rotation as R


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


def transform_to_gym(transform):
    """Convert numpy array to gymapi.Transform

    Args:
        transform (numpy.ndarray): transform to convert

    Returns:
        gymapi.Transform: transform in gymapi.Transform
    """
    # convert to gymapi.Transform
    t = gymapi.Transform()
    t.p = gymapi.Vec3(transform[0, 3], transform[1, 3], transform[2, 3])
    quat = R.from_matrix(transform[:3, :3]).as_quat()
    t.r = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])
    return t
