import os
import numpy as np
import trimesh
import trimesh.viewer
import pickle
from src.visualizer import (draw_pj_grasp_on_mesh,
                            draw_sc_grasp_on_mesh)


def on_draw(viewer):
    camera_transform = viewer.camera_transform
    print(camera_transform)


if __name__ == "__main__":
    # set save dir
    run_history_dir = os.path.join(
        'run_history',
        '2023-05-04-18-09-51')

    # load meshes
    object_files = np.load(os.path.join(run_history_dir, 'object_files.npy'))
    meshes = []
    obejct_names = []

    for file in object_files:
        object_name = file.split('/')[-1].split('.')[0]
        mesh_file = file + '/' + object_name + '.obj'
        mesh = trimesh.load(mesh_file)
        meshes.append(mesh)
        obejct_names.append(object_name)

    # viower camera transform
    camera_transform = np.array([
        [ 0.61777526, -0.44732442,  0.64672606,  0.22649504],
        [ 0.78130769,  0.44220193, -0.44047218, -0.15957632],
        [-0.08894955,  0.77740486,  0.62267942,  0.22679349],
        [ 0.,          0.,          0.,          1.        ]])

    # create save dir based on mesh name
    pj_save_dir = os.path.join(run_history_dir, 'grasps_on_mesh', 'pj')
    for i in range(len(obejct_names)):
        mesh = meshes[i]
        mesh_name = obejct_names[i]

        # load grasps
        grasp_poses = pickle.load(open(os.path.join(pj_save_dir, mesh_name, 'grasp_poses.pkl'), 'rb'))
        grasp_successes = pickle.load(open(os.path.join(pj_save_dir, mesh_name, 'successes.pkl'), 'rb'))
        priors = pickle.load(open(os.path.join(pj_save_dir, mesh_name, 'priors.pkl'), 'rb'))
        posteriors = pickle.load(open(os.path.join(pj_save_dir, mesh_name, 'posteriors.pkl'), 'rb'))

        # make numpy array
        grasp_poses = np.array(grasp_poses)
        grasp_successes = np.array(grasp_successes)
        priors = np.array(priors)
        posteriors = np.array(posteriors)

        # # max sample num = 50
        grasp_poses = grasp_poses[:100]
        grasp_successes = grasp_successes[:100]
        priors = priors[:100]
        posteriors = posteriors[:100]

        # draw grasp on mesh
        gt_scene = draw_pj_grasp_on_mesh(mesh, grasp_poses, grasp_successes)
        prior_scene = draw_pj_grasp_on_mesh(mesh, grasp_poses, priors)
        posterior_scene = draw_pj_grasp_on_mesh(mesh, grasp_poses, posteriors)

        # set camera transform
        gt_scene.camera_transform = camera_transform
        prior_scene.camera_transform = camera_transform
        posterior_scene.camera_transform = camera_transform

        # save image
        img = gt_scene.save_image(resolution=(1280, 720), visible=True)
        with open(os.path.join(pj_save_dir, mesh_name, 'gt.png'), 'wb') as f:
            f.write(img)
        img = prior_scene.save_image(resolution=(1280, 720), visible=True)
        with open(os.path.join(pj_save_dir, mesh_name, 'prior.png'), 'wb') as f:
            f.write(img)
        img = posterior_scene.save_image(resolution=(1280, 720), visible=True)
        with open(os.path.join(pj_save_dir, mesh_name, 'posterior.png'), 'wb') as f:
            f.write(img)

    # create save dir based on mesh name
    sc_save_dir = os.path.join(run_history_dir, 'grasps_on_mesh', 'sc')
    for i in range(len(obejct_names)):
        mesh = meshes[i]
        mesh_name = obejct_names[i]

        # load grasps
        grasp_poses = pickle.load(open(os.path.join(sc_save_dir, mesh_name, 'grasp_poses.pkl'), 'rb'))
        grasp_successes = pickle.load(open(os.path.join(sc_save_dir, mesh_name, 'successes.pkl'), 'rb'))
        priors = pickle.load(open(os.path.join(sc_save_dir, mesh_name, 'priors.pkl'), 'rb'))
        posteriors = pickle.load(open(os.path.join(sc_save_dir, mesh_name, 'posteriors.pkl'), 'rb'))

        # make numpy array
        grasp_poses = np.array(grasp_poses)
        grasp_successes = np.array(grasp_successes)
        priors = np.array(priors)
        posteriors = np.array(posteriors)

        # # max sample num = 50
        grasp_poses = grasp_poses[:100]
        grasp_successes = grasp_successes[:100]
        priors = priors[:100]
        posteriors = posteriors[:100]

        # draw grasp on mesh
        gt_scene = draw_sc_grasp_on_mesh(mesh, grasp_poses, grasp_successes)
        prior_scene = draw_sc_grasp_on_mesh(mesh, grasp_poses, priors)
        posterior_scene = draw_sc_grasp_on_mesh(mesh, grasp_poses, posteriors)

        # set camera transform
        gt_scene.camera_transform = camera_transform
        prior_scene.camera_transform = camera_transform
        posterior_scene.camera_transform = camera_transform

        # save image
        img = gt_scene.save_image(resolution=(1280, 720), visible=True)
        with open(os.path.join(sc_save_dir, mesh_name, 'gt.png'), 'wb') as f:
            f.write(img)
        img = prior_scene.save_image(resolution=(1280, 720), visible=True)
        with open(os.path.join(sc_save_dir, mesh_name, 'prior.png'), 'wb') as f:
            f.write(img)
        img = posterior_scene.save_image(resolution=(1280, 720), visible=True)
        with open(os.path.join(sc_save_dir, mesh_name, 'posterior.png'), 'wb') as f:
            f.write(img)
