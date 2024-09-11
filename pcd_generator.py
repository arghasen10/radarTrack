import mmwave as mm
import mmwave.dsp as dsp
from mmwave.dataloader import DCA1000
from mmwave.tracking import EKF
from mmwave.tracking import gtrack_visualize
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import time
import configuration as cfg
import glob
from helper import *

def generate_pcd(filename):
    file_name = file.split('/')[-1]
    info_dict = get_info(file_name)
    NUM_FRAMES = info_dict['Nf'][0]
    with open(file, 'rb') as ADCBinFile: 
        frames = np.frombuffer(ADCBinFile.read(cfg.FRAME_SIZE*4*NUM_FRAMES), dtype=np.uint16)
    all_data = frame_reshape(frames, NUM_FRAMES)
    range_azimuth = np.zeros((cfg.ANGLE_BINS, cfg.BINS_PROCESSED))
    num_vec, steering_vec = dsp.gen_steering_vec(cfg.ANGLE_RANGE, cfg.ANGLE_RES, cfg.VIRT_ANT)
    tracker = EKF()
    count = 0
    pcd_datas = []
    for adc_data in all_data:
        count+=1
        radar_cube = dsp.range_processing(adc_data)
        mean = radar_cube.mean(0)                 
        radar_cube = radar_cube - mean  
        # --- capon beamforming
        beamWeights   = np.zeros((cfg.VIRT_ANT, cfg.BINS_PROCESSED), dtype=np.complex128)
        radar_cube = np.concatenate((radar_cube[0::3,...], radar_cube[1::3,...], radar_cube[2::3,...]), axis=1)
        # Note that when replacing with generic doppler estimation functions, radarCube is interleaved and
        # has doppler at the last dimension.
        for i in range(cfg.BINS_PROCESSED):
            range_azimuth[:,i], beamWeights[:,i] = dsp.aoa_capon(radar_cube[:, :, i].T, steering_vec, magnitude=True)
        
        """ 3 (Object Detection) """
        heatmap_log = np.log2(range_azimuth)
        
        # --- cfar in azimuth direction
        first_pass, _ = np.apply_along_axis(func1d=dsp.ca_,
                                            axis=0,
                                            arr=heatmap_log,
                                            l_bound=1.5,
                                            guard_len=4,
                                            noise_len=16)
        
        # --- cfar in range direction
        second_pass, noise_floor = np.apply_along_axis(func1d=dsp.ca_,
                                                    axis=0,
                                                    arr=heatmap_log.T,
                                                    l_bound=2.5,
                                                    guard_len=4,
                                                    noise_len=16)

        # --- classify peaks and caclulate snrs
        noise_floor = noise_floor.T
        first_pass = (heatmap_log > first_pass)
        second_pass = (heatmap_log > second_pass.T)
        peaks = (first_pass & second_pass)
        peaks[:cfg.SKIP_SIZE, :] = 0
        peaks[-cfg.SKIP_SIZE:, :] = 0
        peaks[:, :cfg.SKIP_SIZE] = 0
        peaks[:, -cfg.SKIP_SIZE:] = 0
        pairs = np.argwhere(peaks)
        azimuths, ranges = pairs.T
        snrs = heatmap_log[pairs[:,0], pairs[:,1]] - noise_floor[pairs[:,0], pairs[:,1]]

        """ 4 (Doppler Estimation) """

        # --- get peak indices
        # beamWeights should be selected based on the range indices from CFAR.
        dopplerFFTInput = radar_cube[:, :, ranges]
        beamWeights  = beamWeights[:, ranges]

        # --- estimate doppler values
        # For each detected object and for each chirp combine the signals from 4 Rx, i.e.
        # For each detected object, matmul (numChirpsPerFrame, numRxAnt) with (numRxAnt) to (numChirpsPerFrame)
        dopplerFFTInput = np.einsum('ijk,jk->ik', dopplerFFTInput, beamWeights)
        if not dopplerFFTInput.shape[-1]:
            continue
        dopplerEst = np.fft.fft(dopplerFFTInput, axis=0)
        dopplerEst = np.argmax(dopplerEst, axis=0)
        dopplerEst[dopplerEst[:]>=cfg.NUM_CHIRPS/2] -= cfg.NUM_CHIRPS
        
        """ 5 (Extended Kalman Filter) """

        # --- convert bins to units
        ranges = ranges * cfg.RANGE_RESOLUTION
        azimuths = (azimuths - (cfg.ANGLE_BINS // 2)) * (np.pi / 180)
        dopplers = dopplerEst * cfg.DOPPLER_RESOLUTION
        snrs = snrs
        
        # --- put into EKF
        tracker.update_point_cloud(ranges, azimuths, dopplers, snrs)
        targetDescr, tNum = tracker.step()
        print(f"targetDescr.shape: {targetDescr.shape}, num of targets: {tNum}")
        frame_pcd = np.zeros((len(tracker.point_cloud),6))
        for point_cloud, idx in zip(tracker.point_cloud, range(len(tracker.point_cloud))):
            frame_pcd[idx,0] = -np.sin(point_cloud.angle) * point_cloud.range
            frame_pcd[idx,1] = np.cos(point_cloud.angle) * point_cloud.range
            frame_pcd[idx,2] = point_cloud.doppler 
            frame_pcd[idx,3] = point_cloud.snr
            frame_pcd[idx,4] = point_cloud.range
            frame_pcd[idx,5] = point_cloud.angle
        pcd_datas.append(frame_pcd)
    return np.array(pcd_datas)
    
    
if __name__ == "__main__":
    files = glob.glob('../mmPhase/datasets/stick*_traj_cse_*.bin')
    for file in files:
        pcd = generate_pcd(file)
        print(pcd.shape)
        break
