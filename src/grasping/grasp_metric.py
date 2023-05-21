from multiprocessing import Pool

import scipy
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import cvxopt
cvxopt.solvers.options['show_progress'] = False

from .suction_model import suction_cup_lib as sclib
from .suction_model import suction_cup_logic as scl


def on_draw(viewer):
    camera_transform = viewer.camera_transform
    print(camera_transform)


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
        self.contact_normal = contact_normal / np.linalg.norm(contact_normal)
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
        tangent_v = np.cross(self.contact_normal, tangent_u)
        tangent_v /= np.linalg.norm(tangent_v)

        # get friction cone
        wrenches = []
        for theta in np.linspace(0, 2 * np.pi, self.num_edges, endpoint=False):
            tangent = self.friction_coef * (tangent_u * np.cos(theta) + tangent_v * np.sin(theta))
            wrenches.append(self.contact_normal + tangent)
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
        t = self.friction_coef * area * self.contact_normal
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
        lambda_matrix = lambda_matrix + np.eye(num_vertices) * 1e-6

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


# If I use `Pool` for ParallelJawGraspMetric, it will cause some weird error
# /home/sungwon/ws/projects/rl/exercise/venv/lib/python3.6/site-packages/
# trimesh/ray/ray_triangle.py:398: RuntimeWarning: overflow encountered in
# true_divide
# The error is caused by rtree is not thread safe
class ParallelJawGraspMetric(object):
    # constant parameters
    num_edges = 8
    finger_radius = 0.005
    torque_scale = 1000.0
    gripper_width = 0.085
    # gripper_height = 0.002
    gripper_height = 0.1
    quality_method = 'ferrari_canny_l1'

    @staticmethod
    def _check_params():
        assert ParallelJawGraspMetric.num_edges is not None
        assert ParallelJawGraspMetric.finger_radius is not None
        assert ParallelJawGraspMetric.torque_scale is not None
        assert ParallelJawGraspMetric.gripper_width is not None
        assert ParallelJawGraspMetric.quality_method is not None

    @staticmethod
    def distance_to_line(points, line_point, line_direction):
        """
        Calculate the distance between a point and a line

        Args:
            points: (N, 3) numpy array of points
            line_point: (3,) numpy array of a point on the line
            line_direction: (3,) numpy array specifying the line direction

        Returns:
            (N,) numpy array of distances
        """
        return np.linalg.norm(np.cross(points - line_point, points - line_point - line_direction), axis=1) / np.linalg.norm(line_direction)

    @staticmethod
    def get_contact_points(mesh, grasp_pose, friction_coef, visualize=False, verbose=False):
        """Get the parallel jaw contact points from a mesh and a grasp pose.

        Args:
            mesh (trimesh.Trimesh): Mesh to compute contact wrenches.
            grasp_pose (np.ndarray): 4x4 grasp pose in object frame. Grasp approach
                direction should be along the z-axis. Grasp axis should be along the x-axis.
            friction_coef (float): friction coefficient.
            visualize (bool): whether to visualize the contact wrenches.
            verbose (bool): whether to print debug information.

        Returns:
            list of ContactPoint: contact points. None if the grasp is not feasible.
        """
        # check params
        ParallelJawGraspMetric._check_params()

        # get params
        gripper_width = ParallelJawGraspMetric.gripper_width
        gripper_height = ParallelJawGraspMetric.gripper_height
        finger_radius = ParallelJawGraspMetric.finger_radius
        num_edges = ParallelJawGraspMetric.num_edges
        torque_scale = ParallelJawGraspMetric.torque_scale

        # get grasp pose
        grasp_center = grasp_pose[:3, 3]
        grasp_approach = grasp_pose[:3, 2]
        grasp_axis = grasp_pose[:3, 0]

        # ray casting
        rt = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

        # get jaw endpoints
        endpoint1 = grasp_center - grasp_axis * gripper_width / 2
        endpoint2 = grasp_center + grasp_axis * gripper_width / 2

        # check collision before closing the gripper
        ray_origin = np.stack([endpoint1, endpoint2], axis=0)
        ray_direction = np.stack([-grasp_approach, -grasp_approach], axis=0)
        collision_intersects = rt.intersects_any(
            ray_origins=ray_origin,
            ray_directions=ray_direction)  # (2,)
        is_collision = np.any(collision_intersects)

        # close the gripper
        is_contact = False
        if not is_collision:
            grid_size = 0.001
            num_rays = int(gripper_height / grid_size)
            ray_origin = grasp_center - grasp_axis * gripper_width / 2
            ray_origins = []
            for i in range(num_rays):
                ray_origins.append(ray_origin - grasp_approach * grid_size * i)
            ray_directions = [grasp_axis] * num_rays
            interesect_loc, _, interesect_tri_idx = rt.intersects_location(
                ray_origins, ray_directions)

            if len(interesect_loc) > 1:
                # compute distance between contact points and grasp_axis
                dists = ParallelJawGraspMetric.distance_to_line(
                    points=interesect_loc,
                    line_point=ray_origin,
                    line_direction=grasp_approach)

                # remove points that are too far away(< gripper width)
                interesect_loc = interesect_loc[dists < gripper_width]
                interesect_tri_idx = interesect_tri_idx[dists < gripper_width]
                dists = dists[dists < gripper_width]

            # update is_contact
            is_contact = (len(interesect_loc) > 1)
        else:
            if verbose:
                print('collision before closing the gripper')

        # get contact points
        contact_poinsts = None
        if (not is_collision) and is_contact:
            # get contact points
            interesect_tri_idx1 = np.argmin(dists)
            dists[dists > gripper_width] = -np.inf
            interesect_tri_idx2 = np.argmax(dists)

            # get contact points
            contact_poinsts = []
            for intersect_idx in [interesect_tri_idx1, interesect_tri_idx2]:
                cp_loc = interesect_loc[intersect_idx]
                cp_tri_idx = interesect_tri_idx[intersect_idx]

                # Use smooth normal
                bary = trimesh.triangles.points_to_barycentric(
                    [mesh.triangles[cp_tri_idx]], [cp_loc])
                vertex_normals = mesh.vertex_normals[mesh.faces[cp_tri_idx]]
                cp_normal = -np.sum(vertex_normals * bary.reshape(3, 1), axis=0)
                cp_tangent = trimesh.unitize(np.random.rand(3))
                while np.abs(cp_normal.dot(cp_tangent)) > 1e-4:
                    cp_tangent = np.cross(cp_normal, np.random.rand(3))
                    cp_tangent /= np.linalg.norm(cp_tangent)

                # visualize vertex normal
                # vertex_vectors = []
                # vertices = mesh.vertices[mesh.faces[cp_tri_idx]]
                # vertex_normals = mesh.vertex_normals[mesh.faces[cp_tri_idx]]
                # for v, vn in zip(vertices, vertex_normals):
                #     vertex_vectors.append([v, v + vn * 0.01])
                # vertex_vectors = trimesh.load_path(vertex_vectors)
                # vertex_vectors.colors = [(0, 0, 255, 200)] * len(vertex_vectors.entities)

                # normal_vector = [[cp_loc, cp_loc + cp_normal * 0.01]]
                # normal_vector = trimesh.load_path(normal_vector)
                # normal_vector.colors = [(255, 0, 0, 200)] * len(normal_vector.entities)
                # scene = trimesh.Scene(geometry=[mesh, vertex_vectors, normal_vector])
                # scene.show()

                # Use mesh noraml
                # cp_normal = -mesh.face_normals[cp_tri_idx]
                # cp_tangent = mesh.triangles[cp_tri_idx][0] - mesh.triangles[cp_tri_idx][1]
                # cp_tangent /= np.linalg.norm(cp_tangent)

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
        else:
            if verbose:
                print('intersect points found while closing the gripper is not 2')

        # visualize for debug
        if visualize is True:
            # set camera transform
            camera_transform = np.array([
                [ 0.99328917,  0.09502994, -0.06592377, -0.01804732],
                [ 0.09646525, -0.36625786,  0.92549967,  0.30213444],
                [ 0.06380508, -0.92564815, -0.37296705, -0.14623659],
                [ 0.,          0.,          0.,          1.        ]])

            # grasp line
            grasp_line = np.stack([endpoint1, endpoint2], axis=0)
            grasp_line = trimesh.load_path(grasp_line)
            grasp_line.colors = [(0, 255, 0, 200)]

            # friction cone
            friction_cone_lines = []
            if contact_poinsts is not None:
                for c in contact_poinsts:
                    for primitive_wrench in c.friction_cone:
                        friction_cone_lines.append(
                            [c.contact_point, c.contact_point - primitive_wrench * 0.01])
                friction_cone_lines = trimesh.load_path(friction_cone_lines)
                friction_cone_lines.colors = [(255, 0, 0, 200)] * len(friction_cone_lines.entities)

            # jaw geometry
            jaw1 = trimesh.primitives.Box([0.001, 0.001, gripper_height])  # x, y, z
            jaw2 = trimesh.primitives.Box([0.001, 0.001, gripper_height])  # x, y, z
            jaw1.apply_translation([gripper_width/2, 0.0, -gripper_height/2])
            jaw2.apply_translation([-gripper_width/2, 0.0, -gripper_height/2])
            jaw1.apply_transform(grasp_pose)
            jaw2.apply_transform(grasp_pose)

            # geometry
            geometry = [mesh, grasp_line, friction_cone_lines, jaw1, jaw2]
            scene = trimesh.Scene(geometry=geometry)
            scene.camera_transform = camera_transform

            # save as image
            # random_hash = np.random.randint(0, 1000)
            # img = scene.save_image(resolution=(1920, 1080), visible=True)
            # with open('figs/robust_wrench/img_{:03d}.png'.format(random_hash), 'wb') as f:
            #     f.write(img)
            scene.show(smooth=False)

        return contact_poinsts

    @staticmethod
    def compute(contact_points):
        quality_method = 'ferrari_canny_l1'
        if quality_method == 'ferrari_canny_l1':
            quality = GraspQaulity.ferrari_canny_l1(contact_points)
        return quality


def add_noise_to_transform(tf, std_r, std_t):
    """Add noise to the transform.

    Args:
        tf (np.ndarray): 4x4 homogenous matrix.
        std_r (float): standard deviation of rotation noise.
        std_t (float): standard deviation of translation noise.

    Returns:
        np.ndarray: 4x4 homogenous matrix.
    """
    so3 = np.random.normal(0, std_r, size=(3,))
    noise_r = scipy.linalg.expm(np.cross(np.eye(3), so3))
    noise_t = np.random.normal(0, std_t, size=(3, 1))
    tf[:3, :3] = noise_r @ tf[:3, :3]
    tf[:3, 3] += noise_t.flatten()
    return tf


# Previous multiprocessing makes wrong result.(2023-05-19)
# parallel processing for grasp quality evaluation
# Fixed on 2023-05-20
def evaluate_pj_grasp(grasp_pose, meshes, mesh_poses, visualize=False):
    """Evaluate grasp quality for a parallel-jaw gripper on the scene.

    Args:
        grasp_pose (np.ndarray): 4x4 homogenous matrix of the grasp pose in world.
        meshes (list of trimesh.Trimesh): meshes to be evaluated.
        mesh_poses (np.ndarray): 4x4 homogenous matrix of the mesh poses in world.

    Returns:
        int: index of the target mesh
        float: grasp successp
    """
    # get closest mesh from grasp center point
    grasp_pose_t = grasp_pose[:3, 3]
    min_dists = []
    for mesh in meshes:
        min_dists.append(
            mesh.nearest.on_surface([grasp_pose_t])[1][0])
    target_mesh_idx = np.argmin(min_dists)
    target_mesh = meshes[target_mesh_idx]
    target_mesh_pose = mesh_poses[target_mesh_idx]

    # get grasp pose in object frame
    grasp_pose_object = np.linalg.inv(target_mesh_pose) @ grasp_pose

    # get contact points
    contact_points = ParallelJawGraspMetric.get_contact_points(
        target_mesh, grasp_pose_object, 1.0, visualize, False)

    # compute grasp quality
    if contact_points is not None:
        quality = ParallelJawGraspMetric.compute(contact_points)
    else:
        quality = 0.0
    quality = 1.0 if quality > 0.002 else 0.0

    return target_mesh_idx, quality


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


def evaluate_robust_pj_grasp(grasp_pose, meshes, mesh_poses):
    """Evaluate grasp quality for a parallel-jaw gripper on the scene.

    Args:
        grasp_pose (np.ndarray): 4x4 homogenous matrix of the grasp pose.
        meshes (list of trimesh.Trimesh): meshes to be evaluated.
        mesh_poses (np.ndarray): 4x4 homogenous matrix of the mesh poses.

    Returns:
        int: index of the target mesh
        float: grasp successp
    """
    # get closest mesh from grasp center point
    grasp_pose_t = grasp_pose[:3, 3]
    min_dists = []
    for mesh in meshes:
        min_dists.append(
            mesh.nearest.on_surface([grasp_pose_t])[1][0])
    target_mesh_idx = np.argmin(min_dists)
    target_mesh = meshes[target_mesh_idx]
    target_mesh_pose = mesh_poses[target_mesh_idx]

    # from `Learning ambidextrous robot grasping policies``
    # can be found on `Supplementary Material`
    num_samples = 10
    grasp_r_std = 0.01
    grasp_t_std = 0.0025
    input_tuple = []
    for _ in range(num_samples):
        grasp_pose_object = np.linalg.inv(target_mesh_pose) @ grasp_pose
        grasp_pose_object = add_noise_to_transform(
            grasp_pose_object, grasp_r_std, grasp_t_std)
        input_tuple.append((target_mesh, grasp_pose_object, 1.0, False, False))

    # parallel processing for grasp quality evaluation
    # If I use `Pool` here, it will cause some weird error
    # /home/sungwon/ws/projects/rl/exercise/venv/lib/python3.6/site-packages/
    # trimesh/ray/ray_triangle.py:398: RuntimeWarning: overflow encountered in
    # true_divide
    # The error is caused by rtree is not thread safe
    contact_points = []
    for i in range(num_samples):
        cp = ParallelJawGraspMetric.get_contact_points(*input_tuple[i])
        if cp is None:
            continue
        contact_points.append([cp])

    # get grasp quality
    with Pool(num_samples) as p:
        quality = p.starmap(ParallelJawGraspMetric.compute, contact_points)

    # get robust quality
    quality = [1.0 if q > 0.002 else 0.0 for q in quality]
    robust_quality = np.sum(quality) / num_samples

    # convert to grasp success
    robust_quality = 1.0 if robust_quality >= 0.5 else 0.0
    return target_mesh_idx, robust_quality
