import cv2
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from gqcnn.grasping.grasp import Grasp2D, SuctionPoint2D
from gqcnn.grasping.policy import GraspAction


def depth_image_to_gray_image(depth_image):
    """Convert depth image to gray image.

    Args:
        depth_image (np.ndarray): The (H, W) depth image to be converted.

    Returns:
        np.ndarray: The converted (H, W, 3) gray image.
    """
    gray_image = depth_image.copy()

    # float depth image to grasy scale image
    min_depth = 0.5
    max_depth = 1.0
    gray_image = (gray_image - min_depth) / (max_depth - min_depth)
    gray_image = np.clip(gray_image, 0, 1)
    gray_image = (gray_image * 255).astype(np.uint8)
    gray_image = np.stack([gray_image] * 3, axis=2)
    return gray_image


def label_grasp_on_image(image, action):
    """Label grasp on the image.

    Args:
        image (np.ndarray): The (H, W, 3) image to be labeled.
        action (GraspAction): The grasp action to be labeled.

    Returns:
        np.ndarray: The labeled (H, W, 3) image.
    """
    if isinstance(action.grasp, Grasp2D):
        return _label_pj_grasp_on_image(image, action)
    elif isinstance(action.grasp, SuctionPoint2D):
        return _label_sc_grasp_on_image(image, action)
    else:
        raise ValueError('Unknown grasp type: {}'.format(type(action.grasp)))


def _label_pj_grasp_on_image(image, action):
    """Label grasp on the image.

    Args:
        image (np.ndarray): The (H, W, 3) image to be labeled.
        action (GraspAction): The grasp action to be labeled.

    Returns:
        np.ndarray: The labeled (H, W, 3) image.
    """
    assert isinstance(action, GraspAction)
    assert isinstance(action.grasp, Grasp2D)

    # get color for grasp
    color = plt.get_cmap('bwr')(action.q_value)
    color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

    # draw grasp
    center = action.grasp.center.data
    endpoints = action.grasp.endpoints
    endpoint1 = (
        int(endpoints[0][0]),
        int(endpoints[0][1]))
    endpoint2 = (
        int(endpoints[1][0]),
        int(endpoints[1][1]))
    cv2.circle(image, (int(center[0]), int(center[1])), 5, color, -1)
    cv2.line(image, endpoint1, endpoint2, color, 2)
    return image


def _label_sc_grasp_on_image(image, action):
    """Label grasp on the image.

    Args:
        image (np.ndarray): The (H, W, 3) image to be labeled.
        action (GraspAction): The grasp action to be labeled.

    Returns:
        np.ndarray: The labeled (H, W, 3) image.
    """
    assert isinstance(action, GraspAction)
    assert isinstance(action.grasp, SuctionPoint2D)

    # get color for grasp
    color = plt.get_cmap('bwr')(action.q_value)
    color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

    # draw grasp
    center = action.grasp.center.data
    cv2.circle(image, (int(center[0]), int(center[1])), 5, color, -1)
    return image


def visulaize_scene(grasp_pose, meshes, mesh_poses):
    mesh_world = []
    for mesh, mesh_pose in zip(meshes, mesh_poses):
        _mesh = mesh.copy()
        _mesh.apply_transform(mesh_pose)
        mesh_world.append(_mesh)

    scene = trimesh.Scene(mesh_world)
    scene.add_geometry(trimesh.primitives.Box(
        extents=[0.08, 0.005, 0.005], transform=grasp_pose))
    scene.add_geometry(trimesh.creation.axis(origin_size=0.01, transform=grasp_pose))
    scene.add_geometry(trimesh.creation.axis(origin_size=0.01))
    scene.show()


def draw_pj_grasp_on_mesh(mesh, grasp_poses, qualities, grasp_width=0.085):
    """Draw parallel jaw grasp on mesh.

    Args:
        mesh (trimesh.Trimesh): The mesh to be drawn.
        grasp_poses (np.ndarray): The (N, 4, 4) grasp pose.
        qualities (np.ndarray): The (N, ) grasp qualities.
        grasp_width (float, optional): The grasp width. Defaults to 0.085.

    Returns:
        trimesh.Scene: The scene with grasp.
    """
    grasp_path = []
    for i in range(len(grasp_poses)):
        grasp_axis = grasp_poses[i][:3, 0]
        grasp_center = grasp_poses[i][:3, 3]

        endpoint1 = grasp_center - grasp_width / 2 * grasp_axis
        endpoint2 = grasp_center + grasp_width / 2 * grasp_axis

        vector_path = np.array([endpoint1, endpoint2])
        vector_path = trimesh.load_path(vector_path)

        # set color
        cmap = plt.get_cmap('bwr')
        color = cmap(qualities[i])

        if i == 0:
            print('Warning: delete it')
            color = (0, 1, 0, 1)

        vector_path.colors = [(np.array(color) * 255).astype(np.uint8)]
        grasp_path.append(vector_path)
    scene = trimesh.Scene([mesh] + grasp_path)
    return scene


def draw_sc_grasp_on_mesh(mesh, grasp_poses, qualities):
    """Draw suction grasp on mesh.

    Args:
        mesh (trimesh.Trimesh): The mesh to be drawn.
        grasp_poses (np.ndarray): The (N, 4, 4) grasp pose.
        qualities (np.ndarray): The (N, ) grasp qualities.
        grasp_width (float, optional): The grasp width. Defaults to 0.085.

    Returns:
        trimesh.Scene: The scene with grasp.
    """
    grasp_path = []
    for i in range(len(grasp_poses)):
        approach_axis = grasp_poses[i][:3, 0]
        contact_point = grasp_poses[i][:3, 3]

        endpoint1 = contact_point
        endpoint2 = (contact_point - (approach_axis * 0.03))

        vector_path = np.array([endpoint1, endpoint2])
        vector_path = trimesh.load_path(vector_path)

        # set color
        cmap = plt.get_cmap('bwr')
        color = cmap(qualities[i])
        vector_path.colors = [(np.array(color) * 255).astype(np.uint8)]
        grasp_path.append(vector_path)
    scene = trimesh.Scene([mesh] + grasp_path)
    return scene
