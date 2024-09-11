import struct
import pickle
import numpy as np
import configuration as cfg
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
# import tensorflow as tf
import seaborn as sns
import argparse
import pandas as pd
import subprocess
import statistics
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from mmwave.dataloader import DCA1000
import mmwave as mm
import mmwave.dsp as dsp
from mmwave.dataloader import DCA1000
from mmwave.tracking import EKF
from mmwave.tracking import gtrack_visualize
import matplotlib.pyplot as plt
import cv2
import os
import time
import glob

plt.rcParams.update({'font.size': 24})
plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
mode_velocities = []
def read8byte(x):
    return struct.unpack('<hhhh', x)

def frame_reshape(frames, NUM_FRAMES):
    adc_data = frames.reshape(NUM_FRAMES, -1)
    print(adc_data.shape)
    all_data = np.apply_along_axis(DCA1000.organize, 1, adc_data, num_chirps=cfg.NUM_CHIRPS*cfg.NUM_TX, num_rx=cfg.NUM_RX, num_samples=cfg.NUM_ADC_SAMPLES)
    return all_data

def phase_unwrapping(phase_len,phase_cur_frame):
    i=1
    new_signal_phase = phase_cur_frame
    for k,ele in enumerate(new_signal_phase):
        if k==len(new_signal_phase)-1:
            continue
        if new_signal_phase[k+1] - new_signal_phase[k] > 1.5*np.pi:
            new_signal_phase[k+1:] = new_signal_phase[k+1:] - 2*np.pi*np.ones(len(new_signal_phase[k+1:]))
    return np.array(new_signal_phase)


def get_args():
    parser=argparse.ArgumentParser(description="Run the phase_generation script")
    parser.add_argument('-f','--file_name',help="Get the .bin file to process")
    args=parser.parse_args()
    return args


def get_info(file_name):
    dataset=pd.read_csv('dataset.csv')
    filtered_row=dataset[dataset['filename']==file_name]
    info_dict={}
    for col in dataset.columns:
        info_dict[col]=filtered_row[col].values
    if len(info_dict['filename'])==0:
        print('Oops! File not found in database. Cross check the file name')
    return info_dict


def print_info(info_dict):
    print('***************************************************************')
    print('Printing the file profile')
    print(f'--filename: {"only_sensor"+info_dict["filename"][0]}')
    print(f'--Length(L in cm): {info_dict[" L"][0]}')
    print(f'--Radial_Length(R in cm): {info_dict[" R"][0]}')
    print(f'--PWM Value: {info_dict[" PWM"][0]}')
    print(f'--A brief desciption: {info_dict[" Description"][0]}')
    print('***************************************************************')


def iterative_range_bins_detection(rangeResult,pointcloud_processcfg):
    if pointcloud_processcfg.enableStaticClutterRemoval:
        rangeResult = clutter_removal(rangeResult, axis=2)
    range_result_absnormal_split=[]
    for i in range(pointcloud_processcfg.frameConfig.numTxAntennas):
        for j in range(pointcloud_processcfg.frameConfig.numRxAntennas):
            r_r=np.abs(rangeResult[i][j])
            r_r[:,0:10]=0
            min_val = np.min(r_r)
            max_val = np.max(r_r)
            r_r_normalise = (r_r - min_val) / (max_val - min_val) * (1000 - 0) + 0
            range_result_absnormal_split.append(r_r_normalise)
    
    range_abs_combined_nparray=np.zeros((pointcloud_processcfg.frameConfig.numLoopsPerFrame,pointcloud_processcfg.frameConfig.numADCSamples))
    for ele in range_result_absnormal_split:
        range_abs_combined_nparray+=ele
    range_abs_combined_nparray/=(pointcloud_processcfg.frameConfig.numTxAntennas*pointcloud_processcfg.frameConfig.numRxAntennas)
    
    range_abs_combined_nparray_collapsed=np.sum(range_abs_combined_nparray,axis=0)/pointcloud_processcfg.frameConfig.numLoopsPerFrame
    peaks_min_intensity_threshold = np.argsort(range_abs_combined_nparray_collapsed)[::-1][:5]
    max_range_index=np.argmax(range_abs_combined_nparray_collapsed)
    return max_range_index, peaks_min_intensity_threshold


def iterative_doppler_bins_selection(dopplerResult,pointcloud_processcfg,range_peaks, max_range_index):
    doppler_result_absnormal_split=[]
    for i in range(pointcloud_processcfg.frameConfig.numTxAntennas):
        for j in range(pointcloud_processcfg.frameConfig.numRxAntennas):
            d_d=np.abs(dopplerResult[i][j])
            d_d[:,0:10]=0
            min_val = np.min(d_d)
            max_val = np.max(d_d)
            d_d_normalise = (d_d - min_val) / (max_val - min_val) * (1000 - 0) + 0
            doppler_result_absnormal_split.append(d_d_normalise)
    
    doppler_abs_combined_nparray=np.zeros((pointcloud_processcfg.frameConfig.numLoopsPerFrame,pointcloud_processcfg.frameConfig.numADCSamples))
    for ele in doppler_result_absnormal_split:
        doppler_abs_combined_nparray+=ele
    doppler_abs_combined_nparray/=(pointcloud_processcfg.frameConfig.numTxAntennas*pointcloud_processcfg.frameConfig.numRxAntennas)
    
    vel_idx=[]
    for peak in range_peaks:
        vel_idx.append(np.argmax(doppler_abs_combined_nparray[:,peak])-91)
    max_doppler_index = np.argmax(doppler_abs_combined_nparray[:,max_range_index])-91
    return max_doppler_index, vel_idx


def get_phase(r,i):
    if r==0:
        if i>0:
            phase=np.pi/2
        else :
            phase=3*np.pi/2
    elif r>0:
        if i>=0:
            phase=np.arctan(i/r)
        if i<0:
            phase=2*np.pi - np.arctan(-i/r)
    elif r<0:
        if i>=0:
            phase=np.pi - np.arctan(-i/r)
        else:
            phase=np.pi + np.arctan(i/r)
    return phase


def solve_equation(phase_cur_frame,info_dict):
    phase_diff=[]
    for soham in range (1,len(phase_cur_frame)):
        phase_diff.append(phase_cur_frame[soham]-phase_cur_frame[soham-1])
    Tp=cfg.Tp
    Tc=cfg.Tc
    L=info_dict[' L'][0]/100
    r0=info_dict[' R'][0]/100
    roots_of_frame=[]
    for i,val in enumerate(phase_diff):
        c=(phase_diff[i]*0.001/3.14)/(3*(Tp+Tc))
        t=3*(i+1)*(Tp+Tc)
        c1=t*t
        c2=-2*L*t
        c3=L*L-c*c*t*t
        c4=2*L*c*c*t
        c5=-r0*r0*c*c
        coefficients=[c1, c2, c3, c4, c5]
        root=min(np.abs(np.roots(coefficients)))
        roots_of_frame.append(root)
    median_root=np.median(roots_of_frame)
    final_roots=[]
    for root in roots_of_frame:
        if root >0.9*median_root and root<1.1*median_root:
            final_roots.append(root)
    return np.mean(final_roots)


def get_velocity_antennawise(range_FFT_,peak, info_dict):
        phase_per_antenna=[]
        vel_peak=[]
        for k in range(0,cfg.LOOPS_PER_FRAME):
            r = range_FFT_[k][peak].real
            i = range_FFT_[k][peak].imag
            phase=get_phase(r,i)
            phase_per_antenna.append(phase)
        phase_cur_frame=phase_unwrapping(len(phase_per_antenna),phase_per_antenna)
        cur_vel=solve_equation(phase_cur_frame,info_dict)
        return cur_vel


def get_velocity(rangeResult,range_peaks,info_dict):
    vel_array_frame=[]
    for peak in range_peaks:
        vel_arr_all_ant=[]
        for i in range(0,cfg.NUM_TX):
            for j in range(0,cfg.NUM_RX):
                cur_velocity=get_velocity_antennawise(rangeResult[i][j],peak,info_dict)
                vel_arr_all_ant.append(cur_velocity)
        vel_array_frame.append(vel_arr_all_ant)
    return vel_array_frame


def find_peaks_in_range_data(rangeResult, pointcloud_processcfg, intensity_threshold):
    range_result_absnormal_split = []
    for i in range(pointcloud_processcfg.frameConfig.numTxAntennas):
        for j in range(pointcloud_processcfg.frameConfig.numRxAntennas):
            r_r = np.abs(rangeResult[i][j])
            r_r[:,0:10] = 0
            min_val = np.min(r_r)
            max_val = np.max(r_r)
            r_r_normalise = (r_r - min_val) / (max_val - min_val) * 1000
            range_result_absnormal_split.append(r_r_normalise)

    range_abs_combined_nparray = np.zeros((pointcloud_processcfg.frameConfig.numLoopsPerFrame, pointcloud_processcfg.frameConfig.numADCSamples))
    for ele in range_result_absnormal_split:
        range_abs_combined_nparray += ele
    range_abs_combined_nparray /= (pointcloud_processcfg.frameConfig.numTxAntennas * pointcloud_processcfg.frameConfig.numRxAntennas)
    
    range_abs_combined_nparray_collapsed = np.sum(range_abs_combined_nparray, axis=0) / pointcloud_processcfg.frameConfig.numLoopsPerFrame
    peaks, _ = find_peaks(range_abs_combined_nparray_collapsed)

    peaks_min_intensity_threshold = []
    for indices in peaks:
        if range_abs_combined_nparray_collapsed[indices] > intensity_threshold:
            peaks_min_intensity_threshold.append(indices)
    
    return peaks_min_intensity_threshold

def check_consistency_of_frame(previous_peaks, current_peaks, threshold):
    if not any(any(abs(c - p) <= threshold for c in current_peaks) for p in previous_peaks):
        return False
    return True

def get_consistent_peaks(previous_peaks, current_peaks, threshold):
    consistent_peaks = [current_peaks[i] for i, val in enumerate(any(abs(c-p) <= threshold for p in previous_peaks) for c in current_peaks) if val]
    return consistent_peaks

def run_data_read_only_sensor(info_dict):
    filename = 'datasets/'+info_dict["filename"][0]
    command =f'python data_read_only_sensor.py {filename} {info_dict[" Nf"][0]}'
    process = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout = process.stdout
    stderr = process.stderr

def call_destructor(info_dict):
    file_name="datasets/only_sensor"+info_dict["filename"][0]
    command =f'rm {file_name}'
    process = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout = process.stdout
    stderr = process.stderr


def get_mode_velocity(velocity_array_framewise):
    vel_array_all = []
    for velocity_all_antennas in velocity_array_framewise:
        for velocity in velocity_all_antennas:
            vel_array_all.append(velocity)
    vel_mode = statistics.mode(vel_array_all)
    return vel_mode