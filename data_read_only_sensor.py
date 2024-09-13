import numpy as np
import struct
import sys
import os

FRAMES = 50

dca_name = sys.argv[1]
n_frames = int(sys.argv[2])

annotated_fname = '/'.join(dca_name.split("/")[0:-1])+"/only_sensor"+dca_name.split("/")[-1]
FRAMES = n_frames+1

ADC_PARAMS = {'chirps': 182, 
              'rx': 4,
              'tx': 3,
              'samples': 256,
              'IQ': 2,
              'bytes': 2}

array_size = ADC_PARAMS['chirps'] * ADC_PARAMS['rx'] * ADC_PARAMS['tx'] * ADC_PARAMS['IQ'] * ADC_PARAMS['samples']
element_size = ADC_PARAMS['bytes']


def read_and_print_dca_file(filename, packet_size):
    rows = FRAMES
    cols = (728 * 1536)  
    timestamp_infos = []
    frame_array = np.zeros((rows, cols), dtype=np.uint16)
    frame_time_array=np.zeros(FRAMES,dtype=np.float64)
    dirty_array=np.zeros(FRAMES)
    index=0 
    with open(filename,'rb') as file:
        last_packet_num=0
        while True:
            timestamp_data=file.read(8)
            if not timestamp_data:
                break
            timestamp=struct.unpack('q',timestamp_data)[0]
            data=file.read(packet_size)
            packet_num=struct.unpack('<1l',data[:4])[0]
            last_packet_num=packet_num
            if (packet_num%(1536))==0:
                print("packet_num%(1536))==0", packet_num)
                break
        
        packet_idx_in_frame=0
        while True:
            
            timestamp_data=file.read(8)
            if not timestamp_data:
                break
            timestamp=struct.unpack('q',timestamp_data)[0]
            data=file.read(packet_size) 
            if not data:
                break
            packet_num=struct.unpack('<1l',data[:4])[0]
            if packet_num==last_packet_num+1:
                last_packet_num=packet_num
              
                frame_array[index][packet_idx_in_frame:packet_idx_in_frame+728]= np.frombuffer(data[10:], dtype=np.uint16)
                packet_idx_in_frame+=728
                if packet_idx_in_frame==728*1536:
                    frame_time_array[index]=timestamp 
                    packet_idx_in_frame=0
                    index+=1
                continue
            elif packet_num>last_packet_num+1:
                dirty_array[index]=1
                frame_array[index][packet_idx_in_frame:packet_idx_in_frame+728]=np.zeros(728)
                packet_idx_in_frame+=728
                last_packet_num=packet_num
                if packet_idx_in_frame==728*1536:
                    frame_time_array[index]=timestamp
                    packet_idx_in_frame=0
                    index+=1
                continue
        for i in range(FRAMES):
            if dirty_array[i]==1:
                if i==0:
                    j=i
                    while(dirty_array[j]==0):
                        j+=1
                    frame_array[i]=frame_array[j]
                else:
                    j=i
                    while(j>=0 and dirty_array[j]==0):
                        j-=1 
                    frame_array[i]=frame_array[j]

    return frame_array,frame_time_array

def annotate(dca_array,frames):

    if os.path.exists(annotated_fname):
        os.remove(annotated_fname)
    annotation_file = open(annotated_fname, "ab")
    for i in range (frames):
        annotation_file.write(dca_array[i])
    annotation_file.close()


def annotate_time_stamp(dca_time_array,frames):
    time_fname = '/'.join(annotated_fname.split('/')[0:-2])+'/time_stamps/time'+dca_name.split('/')[-1]
    print("writing time stamps in ", time_fname)
    if os.path.exists(time_fname):
        os.remove(time_fname)
    annotation_file = open(time_fname, "ab")
    for i in range (frames):
        annotation_file.write(dca_time_array[i])
    annotation_file.close()



dca_array,dca_time_array=read_and_print_dca_file(dca_name,1466)
print("writing data in ", annotated_fname)
annotate(dca_array,FRAMES)
annotate_time_stamp(dca_time_array, FRAMES)
