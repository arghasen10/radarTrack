import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
import os
from matplotlib.animation import FuncAnimation
from sklearn.cluster import DBSCAN
from helper import generate_pcd_and_speed
fig = plt.figure()
ax = fig.add_subplot(111)

scat = ax.scatter([], [], s=50)


def compute_centroids(cluster_labels, current_data):
    unique_labels = np.unique(cluster_labels)
    centroids = []
    for label in unique_labels:
        cluster_points = current_data[cluster_labels == label, :2]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
    return np.array(centroids)


def update_only_points(frame,raw_poincloud_data_for_plot):
    ax.clear()  # Clear the previous frame
    ax.set_xlim(-10, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    current_data = raw_poincloud_data_for_plot[frame]
    doppler_shifts = current_data[:,2]
    normalized_doppler_shifts = (doppler_shifts-doppler_shifts.min())/(doppler_shifts.max()-doppler_shifts.min())
    scat = ax.scatter(current_data[:, 0], current_data[:, 1], c=normalized_doppler_shifts, cmap='viridis', marker='o')
    std_x = np.std(current_data[:, 0])
    std_y = np.std(current_data[:, 1])
    std_dev_str = f'Stdev X: {std_x:.2f}, Y: {std_y:.2f}'
    ax.legend(loc='upper center', ncol=3, fontsize='small')
    ax.set_title(f'2D Scatter Plot Animation (Frame {frame})\n {std_dev_str}')
    fig.tight_layout()
    return scat,


def update_points_with_clusters(frame, raw_pointcloud_data_for_plot):
    ax.clear() 
    ax.set_xlim(-10, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    
    # Get current point cloud data for the frame
    current_data = raw_pointcloud_data_for_plot[frame]
    
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=1, min_samples=4).fit(current_data[:, :2])
    cluster_labels = clustering.labels_
    # Compute centroids of clusters
    centroids = compute_centroids(cluster_labels, current_data)
    
    # Normalize doppler shifts for color mapping
    doppler_shifts = current_data[:, 2]
    normalized_doppler_shifts = (doppler_shifts - doppler_shifts.min()) / (doppler_shifts.max() - doppler_shifts.min())
    
    # Scatter plot for points, colored by doppler shifts
    scatter = ax.scatter(current_data[:, 0], current_data[:, 1], c=cluster_labels, cmap='viridis', marker='o', label="Points")
    
    # Plot the centroids
    if centroids.size > 0:
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
    
    # Add legend and title
    ax.legend(loc='upper center', ncol=3, fontsize='small')
    ax.set_title(f'2D Scatter Plot with Clusters (Frame {frame})')
    
    fig.tight_layout()
    return scatter,



if __name__ == "__main__":
    cluster_plot = True
    files = glob.glob('../mmPhase/datasets/stick*_traj_cse_*.bin')
    for file in files:
        print(file)
        pickle_save_name = file.split('/')[-1].split('.')[0] + 'pcd_vb_data.pkl'
        if os.path.exists(pickle_save_name):
            with open(pickle_save_name, 'rb') as f:
                data = pickle.load(f)
            pcd, velocity = data['pcd'], data['velocity']
        else:
            pcd, velocity = generate_pcd_and_speed(file)
            with open(pickle_save_name, 'wb') as f:
                pickle.dump({'pcd': pcd, 'velocity': velocity}, f)

                
        if cluster_plot:
            anim = FuncAnimation(fig, update_points_with_clusters, frames=pcd.shape[0], interval=50, blit=True, fargs=(pcd,))
            anim_savefig_name = f'2d_scatter_animation_clusters_{pickle_save_name.split('.')[0]}.gif'
            anim.save(anim_savefig_name, writer='ffmpeg', fps=10)
        else:
            anim = FuncAnimation(fig, update_only_points, frames=pcd.shape[0], interval=50, blit=True, fargs=(pcd,))
            anim_savefig_name = f'2d_scatter_animation_points_{pickle_save_name.split('.')[0]}.gif'
            anim.save(anim_savefig_name, writer='ffmpeg', fps=10)
