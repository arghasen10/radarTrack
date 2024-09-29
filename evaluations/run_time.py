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



dca = DCA1000()
#if len(sys.argv) > 1:
#    num_frames = sys.argv[1]
start_time = time.time()
while 1:
    adc_data = dca.read(num_frames=1)
    adc_data = frame_reshape(adc_data, 1)
    print(f"adc_data: {adc_data.shape}")
    heatmap_d = create_radar_doppler(adc_data[0])
    heatmap_de = create_radar_doppler_elevation(adc_data[0])
    print(heatmap_d.shape, heatmap_de.shape)
    break   
#end_time = time.time()
#total_time = end_time-start_time
#print("Total time: ", total_time)
#print("FPS:", int(num_frames)/total_time) 
#print(adc_data.shape)



class FlowDatasetOur(Dataset):
    """Flow dataset."""

    topics = ["radar_d", "radar_de", "velocity"]

    def __init__(self, path, subsample_factor=1, transform=None):
        # Load files from .npz.
        self.path = path
        print(path)
        with np.load(path) as data:
            self.files = [k for k in data.files if k in self.topics]
            self.dataset = {k: data[k][::subsample_factor] for k in self.files}
            if "velocity" in self.dataset:
                self.dataset["velocity"] = self.dataset["velocity"][:,1:]
            
            if "velocity" in self.dataset:
                velocity = self.dataset["velocity"]
                # Identify indices where velocity is NaN
                nan_indices = np.isnan(velocity)
                # Filter out corresponding values from radar_d and radar_de
                if "radar_d" in self.dataset:
                    self.dataset["radar_d"] = self.dataset["radar_d"][~nan_indices]
                if "radar_de" in self.dataset:
                    self.dataset["radar_de"] = self.dataset["radar_de"][~nan_indices]
                self.dataset["velocity"] = np.expand_dims(self.dataset["velocity"][~nan_indices], axis=1)
                print("radar_d: ",self.dataset['radar_d'].shape)
                print("radar_de:", self.dataset['radar_de'].shape)
                print("velocity:", self.dataset['velocity'].shape)

        # Check if lengths are the same.
        for k in self.files:
            print(k, self.dataset[k].shape, self.dataset[k].dtype)
        lengths = [self.dataset[k].shape[0] for k in self.files]
        assert len(set(lengths)) == 1
        self.num_samples = lengths[0]

        # Save transforms.
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = {
            k: (
                torch.from_numpy(self.dataset[k][idx])
                if type(self.dataset[k][idx]) is np.ndarray
                else self.dataset[k][idx]
            )
            for k in self.files
        }
        if self.transform:
            sample = self.transform(sample)
        return sample
    

if __name__ == "__main__":
    files = glob.glob("../mmPhase/datasets/*.bin")
    files = [file for file in files if not file.split('/')[-1].startswith("only_sensor")]
    files = [file for file in files if 'stick_' not in file]
    files = [file for file in files if 'drone' not in file]
    files = [file for file in files if 'dynamic' not in file]
    npz_paths = [os.path.splitext(file)[0] + ".npz" for file in files]
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    saved_model = torch.load('/home/stick/radarTrack/radarize_ae/output_main_0/transnet18/transnet18.pth')
    model_name = saved_model["model_name"]
    model_type = saved_model["model_type"]
    model_kwargs = saved_model["model_kwargs"]
    state_dict = saved_model["model_state_dict"]
    net = getattr(model, model_type)(**model_kwargs).to(device)

    mean_pred_mae = []
    mean_pred_rmse = []

    for path in npz_paths:
        print(f"Processing {path}...")
        dataset = FlowDatasetOur(
            path,
            subsample_factor=1,
            transform=None,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0
        )

        flow_pred_xs, flow_pred_ys = [], []
        flow_gt_xs, flow_gt_ys = [], []
        flow_pred_final = []
        flow_gt_final = []

        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader)):
                for k, v in batch.items():
                    batch[k] = v.to(device)
                    x = torch.cat([batch["radar_d"], batch["radar_de"]], axis=1).to(
                        torch.float32
                    )
                    flow_gt = batch["velocity"].cpu()

                    flow_pred = net(x)
                    flow_pred = flow_pred.cpu()
                    flow_pred = torch.squeeze(flow_pred, dim=1)

                    flow_x, flow_y = flow_pred[:, 0].numpy(), flow_pred[:, 1].numpy()
                    final_flow = np.sqrt(flow_x*flow_x+flow_y*flow_y)
                    flow_pred_final.append(final_flow)
                    flow_gt_val = flow_gt.numpy()
                    flow_gt_final.append(flow_gt_val)

            flow_pred_final, flow_gt_final = np.array(flow_pred_final) , np.array(flow_gt_final)
            print(f"MAE: {np.mean(np.abs(flow_pred_final - flow_gt_final)):.3f}")

            print(f"RMSE: {np.sqrt(np.mean((flow_pred_final - flow_gt_final)**2)):.3f}")

            print(f"err_mean: {np.mean((flow_pred_final - flow_gt_final)):.3f}")
            print(f"err_std:  {np.std((flow_pred_final - flow_gt_final)):.3f}")

            pred_mae = (
                np.mean(np.abs(flow_pred_final - flow_gt_final)))
            pred_rmse = (
                np.sqrt(np.mean((flow_pred_final - flow_gt_final) ** 2)))

        mean_pred_mae.append(pred_mae)
        mean_pred_rmse.append(pred_rmse)
        d = {
                "flow_pred_xs": flow_pred_final,
                "flow_pred_ys": flow_gt_final,
            }
        np.savez(
                os.path.join(
                    "results/radarize/", path.split('/')[-1]),
                **d,
            )


