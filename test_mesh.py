import glob
import trimesh


target_dir = 'assets/urdf/adv_obj_urdf'
mesh_files = glob.glob(target_dir + '/*/*.obj')

for f in mesh_files:
    mesh = trimesh.load(f)

    smoothed_mesh = mesh.copy()
    # subdivide mesh
    smoothed_mesh.subdivide()
    smoothed_mesh = trimesh.smoothing.filter_humphrey(smoothed_mesh, iterations=5)

    # show meshes to compare
    smoothed_mesh.apply_transform(trimesh.transformations.translation_matrix([0, 0, 0.1]))
    scene = trimesh.Scene([mesh, smoothed_mesh])
    scene.show()
