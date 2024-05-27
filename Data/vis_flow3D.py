import mayavi.mlab as mlab
import numpy as np
import os


def genflow(pc1, pc2, mask1_flow, mask2_flow, flow):
    n1 = len(pc1)
    n2 = len(pc2)

    full_mask1 = np.arange(n1)
    full_mask2 = np.arange(n2)
    mask1_noflow = np.setdiff1d(full_mask1, mask1_flow, assume_unique=True)
    mask2_noflow = np.setdiff1d(full_mask2, mask2_flow, assume_unique=True)

    num_points = 8192
    nonrigid_rate = 0.8
    rigid_rate = 0.2
    if n1 >= num_points:
        if int(num_points * nonrigid_rate) > len(mask1_flow):
            num_points1_flow = len(mask1_flow)
            num_points1_noflow = num_points - num_points1_flow
        else:
            num_points1_flow = int(num_points * nonrigid_rate)
            num_points1_noflow = int(num_points * rigid_rate) + 1

        try:  # ANCHOR: argoverse has some cases without nonrigid flows.
            sample_idx1_noflow = np.random.choice(mask1_noflow, num_points1_noflow, replace=False)
        except:
            sample_idx1_noflow = np.random.choice(mask1_noflow, num_points1_noflow, replace=True)
        sample_idx1_flow = np.random.choice(mask1_flow, num_points1_flow, replace=False)
        sample_idx1 = np.hstack((sample_idx1_flow, sample_idx1_noflow))

        pc1_ = pc1[sample_idx1, :]
        flow_ = flow[sample_idx1, :]

        pc1 = pc1_.astype('float32')
        flow = flow_.astype('float32')

    if n2 >= num_points:
        if int(num_points * nonrigid_rate) > len(mask2_flow):
            num_points2_flow = len(mask2_flow)
            num_points2_noflow = num_points - num_points2_flow
        else:
            num_points2_flow = int(num_points * nonrigid_rate)
            num_points2_noflow = int(num_points * rigid_rate) + 1
            
        try:  # ANCHOR: argoverse has some cases without nonrigid flows.
            sample_idx2_noflow = np.random.choice(mask2_noflow, num_points2_noflow, replace=False)
        except:
            sample_idx2_noflow = np.random.choice(mask2_noflow, num_points2_noflow, replace=True)
        sample_idx2_flow = np.random.choice(mask2_flow, num_points2_flow, replace=False)
        sample_idx2 = np.hstack((sample_idx2_flow, sample_idx2_noflow))

        pc2_ = pc2[sample_idx2, :]
        pc2 = pc2_.astype('float32')
    return pc1, pc2, flow

def visualize_point_clouds_and_flow(file_list):
    # Set the background to black
    mlab.options.background = (0, 0, 0)

    for file_path in file_list:
        # Load the .npz file
        data = np.load(file_path)
        pc1 = data['pc1']
        pc2 = data['pc2']
        flow = data['flow'] if 'flow' in data else pc2 - pc1
        mask1_flow = data['mask1_tracks_flow']
        mask2_flow = data['mask2_tracks_flow']

        pc1, pc2, flow = genflow(pc1, pc2, mask1_flow, mask2_flow, flow)
        # Visualize the point clouds
        # pc1 in red
        mlab.points3d(pc1[:, 0], pc1[:, 1], pc1[:, 2], color=(0.8, 0, 0), scale_factor=0.5)
        # pc2 in green
        mlab.points3d(pc2[:, 0], pc2[:, 1], pc2[:, 2], color=(0, 0.8, 0), scale_factor=0.5)

        # pc3 in blue
        pc3 = pc1+flow
        mlab.points3d(pc3[:, 0], pc3[:, 1], pc3[:, 2], color=(0, 0, 0.9), scale_factor=0.5)

        # Flow vectors in blue
        # quiver = mlab.quiver3d(
        #     pc1[:, 0], pc1[:, 1], pc1[:, 2],
        #     flow[:, 0], flow[:, 1], flow[:, 2],
        #     color=(0, 0, 1), scale_factor=1.0, mode='arrow'
        # )

        # # Access the glyph object to modify the arrows
        # quiver.glyph.glyph_source.glyph_source = quiver.glyph.glyph_source.glyph_dict['arrow_source']

        # # Now you can safely set the shaft and tip radius
        # quiver.glyph.glyph_source.glyph_source.shaft_radius = 0.03
        # quiver.glyph.glyph_source.glyph_source.tip_radius = 0.10


        mlab.show()

def find_npz_files(directory):
    npz_files_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.npz'):
                full_path = os.path.join(root, file)
                npz_files_list.append(full_path)
    return npz_files_list

search_directory = '/sceneflow_eval_dataset/argoverse/withNormal'
npz_files = find_npz_files(search_directory)

# Call the visualization function
visualize_point_clouds_and_flow(npz_files)