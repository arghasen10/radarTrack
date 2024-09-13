import glob
from helper import *
from collections import deque, defaultdict
from tqdm import tqdm
from radarize_ae.radarize.utils import dsp
import os

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

    heatmap_msgs = []

    radar_buffer = deque(maxlen=radar_buffer_len)

    for i, msg in tqdm(enumerate(all_data)):
        # Convert radar msg to radar cube.
        radar_cube = msg
        radar_cube = np.concatenate((radar_cube[0::3,...], radar_cube[1::3,...], radar_cube[2::3,...]), axis=1)
        # Accumulate radar cubes in buffer.
        radar_buffer.append(radar_cube)
        if len(radar_buffer) < radar_buffer.maxlen:
            continue
        radar_cube = np.concatenate(radar_buffer, axis=0)

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

        heatmap_msgs.append(heatmap)

    return heatmap_msgs


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

    heatmap_msgs = []

    radar_buffer = deque(maxlen=radar_buffer_len)

    for i, msg in tqdm(enumerate(all_data)):
        # Convert radar msg to radar cube.
        radar_cube = msg
        radar_cube = np.concatenate((radar_cube[0::3,...], radar_cube[1::3,...], radar_cube[2::3,...]), axis=1)
        # Accumulate radar cubes in buffer.
        radar_buffer.append(radar_cube)
        if len(radar_buffer) < radar_buffer.maxlen:
            continue
        radar_cube = np.concatenate(radar_buffer, axis=0)
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

        heatmap_msgs.append(heatmap)
        
    return heatmap_msgs


def create_dataset(file):
    gb = get_bin_file(file)
    if gb is None:
        return None
    bin_filename, info_dict = gb
    NUM_FRAMES = info_dict['Nf'][0]
    with open(bin_filename, 'rb') as ADCBinFile: 
        frames = np.frombuffer(ADCBinFile.read(cfg.FRAME_SIZE*4*NUM_FRAMES), dtype=np.uint16)
    all_data = frame_reshape(frames, NUM_FRAMES)
    da_heatmaps = create_radar_doppler(all_data)
    da_e_heatmaps = create_radar_doppler_elevation(all_data)
    frames = defaultdict(lambda: [])
    frames["radar_d"] = [da_heatmaps]  # doppler heatmap
    frames["radar_de"] = [da_e_heatmaps]
    frames = {k: v for k, v in frames.items() if len(v[0]) > 0}
    npz_file = os.path.splitext(file)[0] + ".npz"
    np.savez(npz_file, **frames)
    return npz_file
    

if __name__ == "__main__":
    files = glob.glob("../mmPhase/datasets/*.bin")
    files = [file for file in files if not file.split('/')[-1].startswith("only_sensor")]
    for file in files:
        print("Processing file ", file)
        npz_file = create_dataset(file)
        if npz_file is None:
            print("Skipped file", file)
            continue