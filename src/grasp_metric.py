import copy
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
import cvxopt
cvxopt.solvers.options['show_progress'] = False


class ContactPoint(object):
    def __init__(self, contact_point, contact_normal, contact_tangent,
                 friction_coef, num_edges, finger_radius, torque_scale):
        """Contact information for a single contact point.

        Args:
            contact_point (np.ndarray): (x, y, z) location of contact point.
            contact_normal (np.ndarray): (x, y, z) normal vector of contact point. Direction should be inward.
            contact_tangent (np.ndarray): (x, y, z) tangent vector of contact point.
            friction_coef (float): friction coefficient.
            num_edges (int): number of edges in friction cone.
            finger_radius (float): radius of finger to compute soft finger contact.
            torque_scale (float): scale factor for torque.
        """
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
        t = np.cross(self.contact_point, self.friction_cone)
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
        wrenches = np.zeros((self.num_edges+2, 6))
        wrenches[:self.num_edges, :3] = self.friction_cone
        wrenches[:self.num_edges, 3:] = self.torque
        wrenches[self.num_edges, 3:] = self.torsion
        wrenches[self.num_edges+1, 3:] = -self.torsion
        return wrenches


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
    num_edges = None
    finger_radius = None
    torque_scale = None
    gripper_width = None
    quality_method = None

    def __init__(self, contact_points):
        # check params are set
        self._check_params()
        self._contact_points = contact_points

    def _check_params(self):
        assert self.num_edges is not None
        assert self.finger_radius is not None
        assert self.torque_scale is not None
        assert self.gripper_width is not None
        assert self.quality_method is not None

    def quality(self):
        if self._contact_points is None:
            return 0.0

        if self.quality_method == 'ferrari_canny_l1':
            return GraspQaulity.ferrari_canny_l1(self._contact_points)
        else:
            raise NotImplementedError

    @staticmethod
    def from_mesh(mesh, grasp_pose, friction_coef):
        """Get the parallel jaw contact wrenches from a mesh and a grasp pose.

        Args:
            mesh (trimesh.Trimesh): Mesh to compute contact wrenches.
            grasp_pose (np.ndarray): 4x4 grasp pose in object frame. Grasp approach
                direction should be along the z-axis. Grasp axis should be along the x-axis.
            friction_coef (float): friction coefficient.

        Returns:
            obj::ParallelJawContact: Parallel jaw contact wrenches.
        """
        # get params
        gripper_width = ParallelJawGraspMetric.gripper_width
        finger_radius = ParallelJawGraspMetric.finger_radius
        num_edges = ParallelJawGraspMetric.num_edges
        torque_scale = ParallelJawGraspMetric.torque_scale

        # smooth mesh
        mesh = copy.deepcopy(mesh)
        mesh = trimesh.smoothing.filter_humphrey(mesh, alpha=0.5, beta=0.5, iterations=10)
        assert isinstance(mesh, trimesh.Trimesh)

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
        if False:
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
            print('collision before closing the gripper')
            return ParallelJawGraspMetric(None)

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
                print('no contact point found while closing the gripper')
                return ParallelJawGraspMetric(None)

            dist = np.linalg.norm(cp_loc - cp_ray_origin, axis=1)
            intersect_first_idx = np.argmin(dist)
            cp_loc = cp_loc[intersect_first_idx]
            cp_tri_idx = cp_tri_idx[intersect_first_idx]
            cp_normal = -mesh.face_normals[cp_tri_idx]
            print('cp_normal: ', cp_normal)
            cp_tangent = mesh.triangles[cp_tri_idx][0] - mesh.triangles[cp_tri_idx][1]
            cp_tangent /= np.linalg.norm(cp_tangent)
            cp = ContactPoint(
                contact_point=cp_loc,
                contact_normal=cp_normal,
                contact_tangent=cp_tangent,
                friction_coef=friction_coef,
                num_edges=num_edges,
                finger_radius=finger_radius,
                torque_scale=torque_scale)
            contact_poinsts.append(cp)

        # visualize for debug
        if True:
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

        return ParallelJawGraspMetric(contact_poinsts)
