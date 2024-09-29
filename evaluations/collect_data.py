from mmwave.dataloader import DCA1000
import time

import sys

dca = DCA1000()
if len(sys.argv) > 1:
    num_frames = sys.argv[1]
start_time = time.time()
adc_data = dca.read(num_frames=int(num_frames))
end_time = time.time()
total_time = end_time-start_time
print("Total time: ", total_time)
print("FPS:", int(num_frames)/total_time) 
print(adc_data.shape)