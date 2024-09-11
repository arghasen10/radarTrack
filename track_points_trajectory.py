from helper import *
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import glob
from pcd_generator import generate_pcd
import pickle


def get_traj(P1, P2, v_b, t, prev_point):
    C1 = (v_b * t) - P1[1] + (P1[0]**2 / (v_b * t - P1[1]))
    B1 = P2[0] + (P2[1] * P1[0] / (v_b * t - P1[1]))
    translation_magnitude = v_b*t #eclid(P1,P2)#v_b*t
    angle = np.arcsin((-B1) / C1)
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    prev_point=np.array(prev_point).reshape(-1,1)
    rotated_point = np.dot(rotation_matrix, prev_point)
    translation_vector = translation_magnitude * np.array([[np.sin(angle)],
                                                           [np.cos(angle)]])
    current_point = rotated_point + translation_vector
    return tuple(current_point.flatten())


def eclid(p1,p2):
    return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5


if __name__ == "__main__":
    files = glob.glob('../mmPhase/datasets/stick*_traj_cse_*.bin')
    for file in files:
        pcd, velocity = generate_pcd_and_speed(file)
        with open('data.pkl', 'wb') as f:
            pickle.dump({'pcd': pcd, 'velocity': velocity}, f)
        print(pcd.shape)
        print(velocity.shape)
        plt.cla()
        total_data = []
        total_ids = []
        total_frames=0
        prev_pointclouds = None
        prev_traj_point = (0,0)
        final_trajectory = [prev_traj_point]
        final_traj_point = prev_traj_point 
        frame_no = 0
        for points, v_b in zip(pcd, velocity):
            traj1 = []
            if prev_pointclouds is None:
                prev_pointclouds = points
                continue
            for point1 in points:
                for point2 in prev_pointclouds:
                    distance = eclid(point1, point2)
                    if distance < 0.05:
                        new_traj_point = get_traj(point1, point2,v_b, 0.2, prev_traj_point)
                        traj1.append(new_traj_point)
            if len(traj1) > 1:
                final_traj_point = (np.median([point[0] for point in traj1]), np.median([point[1] for point in traj1]))
                print(final_traj_point)
            else:
                print("Skipped frame_no: ", frame_no)
            # print("final_traj_point: ", final_traj_point)
            prev_traj_point = final_traj_point
            final_trajectory.append(prev_traj_point)
            prev_pointclouds = points
            frame_no += 1
            
            
        plt.plot([e[0] for e in final_trajectory],[e[1] for e in final_trajectory])
        save_fig = f"trajectory_plot_{traj_file.split('.')[0]}.png"
        plt.savefig(save_fig)
        plt.close()
        print(save_fig)
        break