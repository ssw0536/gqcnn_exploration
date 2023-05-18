from multiprocessing import Pool

import time
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import cvxopt
cvxopt.solvers.options['show_progress'] = False

from .suction_model import suction_cup_lib as sclib
from .suction_model import suction_cup_logic as scl


class RigidContactPoint(object):
    def __init__(self, center_of_mass, contact_point, contact_normal, contact_tangent,
                 friction_coef, num_edges, finger_radius, torque_scale):
        """Contact information for a single contact point.

        Args:
            center_of_mass (np.ndarray): (x, y, z) location of center of mass.
            contact_point (np.ndarray): (x, y, z) location of contact point.
            contact_normal (np.ndarray): (x, y, z) normal vector of contact point. Direction should be inward.
            contact_tangent (np.ndarray): (x, y, z) tangent vector of contact point.
            friction_coef (float): friction coefficient.
            num_edges (int): number of edges in friction cone.
            finger_radius (float): radius of finger to compute soft finger contact.
            torque_scale (float): scale factor for torque.
        """
        self.center_of_mass = center_of_mass
        self.contact_point = contact_point
        self.contact_noraml = contact_normal / np.linalg.norm(contact_normal)
        self.contact_tangent = contact_tangent / np.linalg.norm(contact_tangent)

        self.friction_coef = friction_coef
        self.num_edges = num_edges
        self.finger_radius = finger_radius
        self.torque_scale = torque_scale

    @property
    def friction_cone(self):
        """Get friction cone.

        Returns:
            np.ndarray: (num_edges, 3) friction cone.
        """
        # get tangent vectors
        tangent_u = self.contact_tangent
        tangent_v = np.cross(self.contact_noraml, tangent_u)
        tangent_v /= np.linalg.norm(tangent_v)

        # get friction cone
        wrenches = []
        for theta in np.linspace(0, 2 * np.pi, self.num_edges, endpoint=False):
            tangent = self.friction_coef * (tangent_u * np.cos(theta) + tangent_v * np.sin(theta))
            wrenches.append(self.contact_noraml + tangent)
        wrenches = np.array(wrenches)
        return wrenches

    @property
    def torque(self):
        """Torque at contact point.

        Returns:
            np.ndarray: (num_edges, 3) torque at contact point.
        """
        t = np.cross(
            self.contact_point - self.center_of_mass,
            self.friction_cone)
        t *= self.torque_scale
        return t

    @property
    def torsion(self):
        """Torsion at contact point.

        Returns:
            np.ndarray: (num_edges, 3) torsion at contact point.
        """
        area = np.pi * self.finger_radius**2
        t = self.friction_coef * area * self.contact_noraml
        t *= self.torque_scale
        return t

    @property
    def primitive_wrenches(self):
        """Get primitive wrenches.

        Returns:
            np.ndarray: (num_edges+2, 6) primitive wrenches. The last two wrenches are for torsion.
        """
        wrenches = np.zeros((self.num_edges + 2, 6))
        wrenches[:self.num_edges, :3] = self.friction_cone
        wrenches[:self.num_edges, 3:] = self.torque
        wrenches[self.num_edges, 3:] = self.torsion
        wrenches[self.num_edges + 1, 3:] = -self.torsion
        return wrenches

    def friction_cone_geomtery(self, scale=0.01):
        """Get friction cone geometry.

        Args:
            scale (float, optional): scale factor for friction cone. Defaults to 0.01.

        Returns:
            trimesh.path.Path3D: friction cone geometry.
        """
        friction_cone_lines = []
        for primitive_wrench in self.friction_cone:
            friction_cone_lines.append(
                [self.contact_point,
                 self.contact_point - primitive_wrench * scale])
        friction_cone_lines = trimesh.load_path(friction_cone_lines)
        return friction_cone_lines


class GraspQaulity(object):
    @staticmethod
    def ferrari_canny_l1(contacts):
        # get grasp wrenche space
        gws = []
        for contact in contacts:
            gws.append(contact.primitive_wrenches)
        gws = np.concatenate(gws, axis=0)

        # get convex hull
        gws_convex_hull = ConvexHull(gws)
        gws_convex_hull_idx = gws_convex_hull.vertices
        gws_convex_hull_points = gws[gws_convex_hull_idx]

        # check whether the origin is inside the convex hull
        # https://github.com/BerkeleyAutomation/dex-net/blob/cccf93319095374b0eefc24b8b6cd40bc23966d2/src/dexnet/grasping/quality.py#L670
        min_norm, _ = GraspQaulity.get_min_norm_point(gws_convex_hull_points)
        if min_norm > 1e-3:
            return 0.0

        # get the minimum norm
        min_norm = np.inf
        gws_convex_hull_faces = gws_convex_hull.simplices
        for i in range(len(gws_convex_hull_faces)):
            face = gws_convex_hull_faces[i]
            face_points = gws[face]
            dist, coef = GraspQaulity.get_min_norm_point(face_points)

            if dist < min_norm:
                min_norm = dist

        return min_norm

    @staticmethod
    def get_min_norm_point(points):
        # https://par.nsf.gov/servlets/purl/10251539
        # https://cvxopt.org/examples/tutorial/qp.html

        # number of vertices
        num_vertices = points.shape[0]

        # create labmda matrix
        # add small value to diagonal to avoid singular matrix
        lambda_matrix = points.dot(points.T)
        lambda_matrix = lambda_matrix + np.eye(num_vertices) * 1e-10

        # solve quadratic programming
        # min x'Qx + p'x
        # s.t. Gx <= h
        #      Ax = b
        Q = cvxopt.matrix(2 * lambda_matrix)
        p = cvxopt.matrix(np.zeros((num_vertices, 1)))
        G = cvxopt.matrix(-np.eye(num_vertices))
        h = cvxopt.matrix(np.zeros((num_vertices, 1)))
        A = cvxopt.matrix(np.ones((1, num_vertices)))
        b = cvxopt.matrix(np.ones(1))
        sol = cvxopt.solvers.qp(Q, p, G, h, A, b)

        # get min norm point
        coef = np.array(sol['x'])  # coefficients of the wrenches
        min_norm = np.sqrt(sol['primal objective'])
        return min_norm, coef


class ParallelJawGraspMetric(object):
    # constant parameters
    num_edges = 8
    finger_radius = 0.005
    torque_scale = 1000.0
    gripper_width = 0.085
    quality_method = 'ferrari_canny_l1'

    @staticmethod
    def _check_params():
        assert ParallelJawGraspMetric.num_edges is not None
        assert ParallelJawGraspMetric.finger_radius is not None
        assert ParallelJawGraspMetric.torque_scale is not None
        assert ParallelJawGraspMetric.gripper_width is not None
        assert ParallelJawGraspMetric.quality_method is not None

    @staticmethod
    def compute(mesh, grasp_pose, friction_coef, visualize=False, verbose=False):
        """Get the parallel jaw contact wrenches from a mesh and a grasp pose.

        Args:
            mesh (trimesh.Trimesh): Mesh to compute contact wrenches.
            grasp_pose (np.ndarray): 4x4 grasp pose in object frame. Grasp approach
                direction should be along the z-axis. Grasp axis should be along the x-axis.
            friction_coef (float): friction coefficient.
            visualize (bool): whether to visualize the contact wrenches.
            verbose (bool): whether to print debug information.

        Returns:
            float: grasp quality. None if the grasp is not feasible.
            list of ContactPoint: contact points. None if the grasp is not feasible.
        """
        # check params
        ParallelJawGraspMetric._check_params()

        # get params
        gripper_width = ParallelJawGraspMetric.gripper_width
        finger_radius = ParallelJawGraspMetric.finger_radius
        num_edges = ParallelJawGraspMetric.num_edges
        torque_scale = ParallelJawGraspMetric.torque_scale
        quality_method = ParallelJawGraspMetric.quality_method

        # smooth mesh
        # mesh = copy.deepcopy(mesh)
        # mesh = trimesh.smoothing.filter_humphrey(mesh, alpha=0.5, beta=0.5, iterations=10)
        # assert isinstance(mesh, trimesh.Trimesh)

        # get grasp pose
        grasp_center = grasp_pose[:3, 3]
        grasp_approach = grasp_pose[:3, 2]
        grasp_axis = grasp_pose[:3, 0]

        # ray casting
        rt = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

        # get jaw endpoints
        endpoint1 = grasp_center - grasp_axis * gripper_width / 2
        endpoint2 = grasp_center + grasp_axis * gripper_width / 2

        # visualize for debug
        if visualize is True:
            grasp_line = np.stack([endpoint1, endpoint2], axis=0)
            grasp_line = trimesh.load_path(grasp_line)
            geometry = [mesh, grasp_line]
            trimesh.Scene(geometry=geometry).show(smooth=False)

        # check collision before closing the gripper
        ray_origin = np.stack([endpoint1, endpoint2], axis=0)
        ray_direction = np.stack([-grasp_approach, -grasp_approach], axis=0)
        is_collision = rt.intersects_any(
            ray_origins=ray_origin,
            ray_directions=ray_direction)  # (2,)
        if np.any(is_collision):
            if verbose:
                print('collision before closing the gripper')
            return (None, None)

        # close the gripper and get contact points
        ray_origin = np.stack([endpoint1, endpoint2], axis=0)
        ray_direction = np.stack([grasp_axis, -grasp_axis], axis=0)

        contact_poinsts = []
        for i in range(2):
            # set ray
            cp_ray_origin = ray_origin[i]
            cp_ray_direction = ray_direction[i]

            # get contact point
            cp_loc, _, cp_tri_idx = rt.intersects_location(
                [cp_ray_origin],
                [cp_ray_direction])
            if len(cp_loc) == 0:
                if verbose:
                    print('no contact point found while closing the gripper')
                return (None, None)

            dist = np.linalg.norm(cp_loc - cp_ray_origin, axis=1)

            if np.min(dist) > gripper_width / 2:
                if verbose:
                    print('no contact point found while closing the gripper')
                return (None, None)
            intersect_first_idx = np.argmin(dist)

            cp_loc = cp_loc[intersect_first_idx]
            cp_tri_idx = cp_tri_idx[intersect_first_idx]
            cp_normal = -mesh.face_normals[cp_tri_idx]
            cp_tangent = mesh.triangles[cp_tri_idx][0] - mesh.triangles[cp_tri_idx][1]
            cp_tangent /= np.linalg.norm(cp_tangent)
            cp = RigidContactPoint(
                center_of_mass=mesh.center_mass,
                contact_point=cp_loc,
                contact_normal=cp_normal,
                contact_tangent=cp_tangent,
                friction_coef=friction_coef,
                num_edges=num_edges,
                finger_radius=finger_radius,
                torque_scale=torque_scale)
            contact_poinsts.append(cp)

        # visualize for debug
        if visualize is True:
            grasp_line = np.stack([endpoint1, endpoint2], axis=0)
            grasp_line = trimesh.load_path(grasp_line)

            friction_cone_lines = []
            for c in contact_poinsts:
                for primitive_wrench in c.friction_cone:
                    friction_cone_lines.append(
                        [c.contact_point, c.contact_point - primitive_wrench * 0.01])
            friction_cone_lines = trimesh.load_path(friction_cone_lines)

            geometry = [mesh, grasp_line, friction_cone_lines]
            trimesh.Scene(geometry=geometry).show(smooth=False)

        if quality_method == 'ferrari_canny_l1':
            quality = GraspQaulity.ferrari_canny_l1(contact_poinsts)

        return (quality, contact_poinsts)


def evaluate_pj_grasp(grasp_pose, meshes, mesh_poses):
    """Evaluate grasp quality for a parallel-jaw gripper on the scene.

    Args:
        grasp_pose (np.ndarray): 4x4 homogenous matrix of the grasp pose.
        meshes (list of trimesh.Trimesh): meshes to be evaluated.
        mesh_poses (np.ndarray): 4x4 homogenous matrix of the mesh poses.

    Returns:
        int: index of the target mesh
        float: grasp success
    """
    # get 3 close mesh from grasp pose
    mesh_pose_xyz = mesh_poses[:, :3, 3]
    grasp_pose_xyz = grasp_pose[:3, 3]
    dist = np.linalg.norm(mesh_pose_xyz - grasp_pose_xyz, axis=1)
    near_indices = np.argsort(dist)[:3]

    # meshes is list of trimesh.Trimesh
    # gather meshes with near_indices
    meshes = [meshes[i] for i in near_indices]
    mesh_poses = mesh_poses[near_indices]

    # gather inputs for parallel processing
    input_tuple = []
    for mesh, mesh_pose in zip(meshes, mesh_poses):
        grasp_pose_object = np.linalg.inv(mesh_pose) @ grasp_pose
        input_tuple.append((mesh, grasp_pose_object, 1.0, False, False))

    # parallel processing for grasp quality evaluation
    with Pool(3) as p:
        output = p.starmap(ParallelJawGraspMetric.compute, input_tuple)

    # gather outputs
    quality = []
    for q, _ in output:
        if q is None:
            quality.append(0.0)
        else:
            quality.append(q)

    # all quality is 0.0, return the first mesh
    if np.count_nonzero(quality) == 0:
        return near_indices[0], 0.0

    # get the best quality
    quality = quality[np.argmax(quality)]
    target_mesh_idx = near_indices[np.argmax(quality)]
    if quality > 0.002:
        return target_mesh_idx, 1.0
    else:
        return target_mesh_idx, 0.0


def evaluate_sc_grasp(grasp_pose, meshes, mesh_poses, visualize=False):
    """Evaluate grasp quality for a suction-cup gripper on the scene.

    Args:
        grasp_pose (np.ndarray): 4x4 homogenous matrix of the grasp pose.
        meshes (trimesh.Trimesh): meshes to be evaluated.
        mesh_poses (np.ndarray): 4x4 homogenous matrix of the mesh poses.

    Returns:
        int : target mesh index.
        float: grasp success.
    """
    # get 3 close mesh from grasp pose
    mesh_pose_xyz = mesh_poses[:, :3, 3]
    grasp_pose_xyz = grasp_pose[:3, 3]
    dist = np.linalg.norm(mesh_pose_xyz - grasp_pose_xyz, axis=1)
    near_indices = np.argsort(dist)[:3]

    # meshes is list of trimesh.Trimesh
    # gather meshes with near_indices
    meshes = [meshes[i] for i in near_indices]
    mesh_poses = mesh_poses[near_indices]

    # gather inputs for parallel processing
    input_tuple = []
    for mesh, mesh_pose in zip(meshes, mesh_poses):
        grasp_pose_object = np.linalg.inv(mesh_pose) @ grasp_pose
        input_tuple.append((mesh, [grasp_pose_object[:3, 3]]))

    # get target mesh
    with Pool(3) as p:
        output = p.starmap(trimesh.proximity.closest_point, input_tuple)
    contact_points = np.array([o[0] for o in output])
    dists = np.array([o[1][0] for o in output])
    target_mesh_idx = near_indices[np.argmin(dists)]

    # gather inputs for grasp quality evaluation
    contact_point = contact_points[np.argmin(dists)][0] * 1000.0  # convert to mm

    obj_model = sclib.ModelData(
        mesh=meshes[np.argmin(dists)],
        units=("meters", "millimeters"),
        subdivide=True)

    grasp_pose_object = np.linalg.inv(mesh_poses[np.argmin(dists)]) @ grasp_pose
    approach_vector = grasp_pose_object[:3, 0]

    # evaluate grasp quality
    contact_seal = scl.contact_test_seal(
        con_p=contact_point,
        obj_model=obj_model,
        a_v=approach_vector,
        noise_samples=0)

    # get score
    if contact_seal[2] is None:
        score = 0.0
    else:
        score = contact_seal[1]

    # visualize for debug
    if visualize is True:
        # get approach vector
        vector_path = np.array(
            [contact_point, contact_point - approach_vector * 10.0]) * 1e-3
        vector_path = trimesh.load_path(vector_path)

        # set color
        cmap = plt.get_cmap("bwr")
        vector_path.colors = [(np.array(cmap(score)) * 255).astype(np.uint8)]

        # visualize
        scene = trimesh.Scene()
        scene.add_geometry(meshes[np.argmin(dists)])
        scene.add_geometry(vector_path)
        scene.show()
    return target_mesh_idx, score
