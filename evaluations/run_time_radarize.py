from mmwave.dataloader import DCA1000
import time

import os
import sys
import numpy as np

parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.insert(0, parent)
from helper import frame_reshape
from radarize_ae.radarize.utils import dsp
import torch
from radarize_ae.radarize.flow import model
from radarize_ae.radarize.flow.dataloader import FlowDataset
from tqdm import tqdm
import glob
from torch.utils.data import Dataset, DataLoader

def create_radar_doppler(
    all_data,
    resize_shape=(181, 60),
    radar_buffer_len=3,
    range_subsampling_factor=1,
    angle_res=1,
    angle_range=90,
    normalization_range=[10.0, 25.0],
):
    """Create doppler-azimuth heatmaps."""

    # Convert radar msg to radar cube.
    radar_cube = all_data
    radar_cube = np.concatenate((radar_cube[0::3,...], radar_cube[1::3,...], radar_cube[2::3,...]), axis=1)
    # Accumulate radar cubes in buffer.
        
    radar_cube_h = radar_cube[::1]
    heatmap_h = dsp.preprocess_1d_radar_1843(
        radar_cube_h,
        angle_res,
        angle_range,
        range_subsampling_factor,
        normalization_range[0],
        normalization_range[1],
        resize_shape,
    )
    heatmap_h = np.fliplr(heatmap_h)

    heatmap = np.stack(
        [
            heatmap_h,
        ]
    )

    return heatmap


def create_radar_doppler_elevation(
    all_data,
    resize_shape=(181, 60),
    radar_buffer_len=3,
    range_subsampling_factor=1,
    angle_res=1,
    angle_range=90,
    normalization_range=[10.0, 25.0],
):
    """Create doppler-azimuth heatmaps with elevation beamforming."""
    radar_cube = all_data
    radar_cube = np.concatenate((radar_cube[0::3,...], radar_cube[1::3,...], radar_cube[2::3,...]), axis=1)
    # Accumulate radar cubes in buffer.
    # Do elevation beamforming.
    radar_cube_e = (
        radar_cube[:, 2:6, :] + radar_cube[:, 8:12, :]
    ) / 2  # [2,3,4,5,8,9,10,11]

    radar_cube_h = radar_cube_e[::1]
    heatmap_h = dsp.preprocess_1d_radar_1843(
        radar_cube_h,
        angle_res,
        angle_range,
        range_subsampling_factor,
        normalization_range[0],
        normalization_range[1],
        resize_shape,
    )
    heatmap_h = np.fliplr(heatmap_h)

    heatmap = np.stack(
        [
            heatmap_h,
        ]
    )
    
    return heatmap
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved_model = torch.load('/home/jetson/radartrack/radarize_ae/output_main_0/transnet18/transnet18.pth')
    model_name = saved_model["model_name"]
    model_type = saved_model["model_type"]
    model_kwargs = saved_model["model_kwargs"]
    state_dict = saved_model["model_state_dict"]
    net = getattr(model, model_type)(**model_kwargs).to(device)
    dca = DCA1000()
    with torch.no_grad():
        while 1:
            start_time = time.time()
            adc_data = dca.read(num_frames=1)
            adc_data = frame_reshape(adc_data, 1)
            print(f"adc_data: {adc_data.shape}")
            heatmap_d = torch.from_numpy(np.expand_dims(create_radar_doppler(adc_data[0]), axis=0))
            heatmap_de = torch.from_numpy(np.expand_dims(create_radar_doppler_elevation(adc_data[0]), axis=0))
            print(heatmap_d.shape, heatmap_de.shape)
            x = torch.cat([heatmap_d, heatmap_de], axis=1).to(
                torch.float32
            )
            flow_pred = net(x.to(device))
            end_time = time.time()
            print('Total time taken: ', (end_time-start_time))
