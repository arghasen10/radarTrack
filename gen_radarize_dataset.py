import glob
from helper import *

if __name__ == "__main__":
    files = glob.glob("../mmPhase/datasets/*.bin")
    files = [file for file in files if not file.split('/')[-1].startswith("only_sensor")]
    for file in files:
        info_dict = get_info(file.split("/")[-1])
        run_data_read_only_sensor('/'.join(files[0].split("/")[0:-1]),info_dict)

# heatmap_ts, heatmap_msgs = [], []

# radar_buffer = deque(maxlen=radar_buffer_len)

# for i, (topic, msg, ts) in tqdm(
#     enumerate(bag.read_messages([radar_topic])),
#     total=bag.get_message_count(radar_topic),
# ):

#     # Convert radar msg to radar cube.
#     radar_cube = dsp.reshape_frame(msg)   #this shape is 32,12,96 => 32 is num of chirps, 12 is virt ant, 96 is range bin

#     # Accumulate radar cubes in buffer.
#     radar_buffer.append(radar_cube)
#     if len(radar_buffer) < radar_buffer.maxlen:
#         continue
#     radar_cube = np.concatenate(radar_buffer, axis=0)

#     radar_cube_h = radar_cube[::1]
#     heatmap_h = dsp.preprocess_1d_radar_1843(
#         radar_cube_h,
#         angle_res,
#         angle_range,
#         range_subsampling_factor,
#         normalization_range[0],
#         normalization_range[1],
#         resize_shape,
#     )
#     heatmap_h = np.fliplr(heatmap_h)

#     heatmap = np.stack(
#         [
#             heatmap_h,
#         ]
#     )

#     heatmap_msgs.append(heatmap)
#     heatmap_ts.append(ts.secs + 1e-9 * ts.nsecs)

# heatmap_ts = np.array(heatmap_ts)
# return heatmap_ts, heatmap_msgs