import os
import glob
import pickle
import pandas as pd
from scipy import stats as st

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ''

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

from os.path import join
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, os.path.dirname(currentdir))
import argparse
import numpy as np
import json
import time
import math

from utility import plot_util
from utility.data_loader import load_data_multi_timestamp
from utility.test_util import convert_rel_to_44matrix, iround
from utility.networks import build_model_cross_att
from datetime import datetime
# keras
from keras import backend as K

K.set_image_dim_ordering('tf')
K.set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)))  #
K.set_learning_phase(0)  # Run testing mode

SCALER = 1.0  # scale label: 1, 100, 10000
RADIUS_2_DEGREE = 180.0 / math.pi
IMU_LENGTH = 20
convert_to_datetime = lambda timestamp: datetime.utcfromtimestamp(timestamp[0])
def predictionF(x_mm_t, x_imu_t, network_model, sess):
    len_x_i = x_mm_t.shape[0]

    pred_transform_t_1 = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])  # initial pose for prediction

    
    out_pred_array = []  # format (x,y) gt and (x,y) prediction

    for i in range(0, (len_x_i - 1)):
        x_mm_1 = x_mm_t[i]
        x_mm_2 = x_mm_t[i + 1]

        x_imu = x_imu_t[i]
        
        predicted = sess.run([network_model.outputs],
                                feed_dict={network_model.inputs[0]: x_mm_1,
                                        network_model.inputs[1]: x_mm_2,
                                        network_model.inputs[2]: x_imu})
        # print("predicted.shape: ", predicted.shape)
        pred_pose = predicted[0][0][0][0]

        pred_transform_t = convert_rel_to_44matrix(0, 0, 0, pred_pose)
        abs_pred_transform = np.dot(pred_transform_t_1, pred_transform_t)

        out_pred_array.append(
            [abs_pred_transform[0, 0], abs_pred_transform[0, 1], abs_pred_transform[0, 2], abs_pred_transform[0, 3],
                abs_pred_transform[1, 0], abs_pred_transform[1, 1], abs_pred_transform[1, 2], abs_pred_transform[1, 3],
                abs_pred_transform[2, 0], abs_pred_transform[2, 1], abs_pred_transform[2, 2],
                abs_pred_transform[2, 3]])

        pred_transform_t_1 = abs_pred_transform

    out_pred_array = np.array(out_pred_array)
    pred_x, pred_y = out_pred_array[:, 3], out_pred_array[:, 7]
    vel_x = np.diff(pred_x)/len_x_i
    vel_y = np.diff(pred_y)/len_x_i
    return vel_x, vel_y


def collect_ra_imu(f,sess):
    print("filename: ", f)
    imu_df = pd.DataFrame(columns=['time', 'z', 'y', 'x', 'yaw', 'pitch', 'roll'])
    ra_df = pd.DataFrame(columns=['time', 'ra'])
    time_difference = pd.Timedelta(seconds=6.276132)
    
    with open(f, 'rb') as file:
        loaded_arrays = pickle.load(file)
    ra_heatmap = loaded_arrays[0]
    imu_data = loaded_arrays[1]
    ra_time = loaded_arrays[2][:-1]
    imu_time = loaded_arrays[3]
    datetime_imu = list(map(convert_to_datetime, imu_time))
    datetime_ra = list(map(convert_to_datetime, ra_time))
    shifted_datetime_imu = [dt - time_difference for dt in datetime_imu]
    for imu, t in zip(imu_data, shifted_datetime_imu):
        imu_df = imu_df.append(pd.Series([t, *imu], index=imu_df.columns), ignore_index=True)
        if t > datetime_ra[-1]:
            break
    for r, t in zip(ra_heatmap, datetime_ra):
        ra_df = ra_df.append({'time': t, 'ra': r}, ignore_index=True)
    imu_df['time'] = imu_df['time'].dt.round('S')
    l=[]
    for g,df in imu_df.groupby('time'):
        while df.shape[0]<20:
            df.loc[df.index[-1]+1]=[df.time.values[0],*[0]*6]
        l.append(df[:20])
    df=pd.concat(l).sort_values(by='time',kind='mergesort').reset_index(drop=True)

    l=[]
    for g,df1 in ra_df.groupby('time'):
        while df1.shape[0]<3:
            df1.loc[df1.index[-1]+1]=[df1.time.values[0],df1.ra.values[-1]]
        l.append(df1[:3])
    df1=pd.concat(l).sort_values(by='time',kind='mergesort').reset_index(drop=True)
    ts=[];imu=[];ra=[]
    imu_groups=df.groupby('time')
    for gr,data_ra in df1.groupby('time'):
        try:
            imu.append(imu_groups.get_group(gr).drop(columns=['time']).values)
            ra.append(data_ra.drop(columns=['time']).values.tolist())
            ts.append(gr)
        except:
            pass
    imu = np.array(imu)
    imu = np.expand_dims(imu, axis=1)
    ra = np.array(ra)
    ra = np.expand_dims(np.transpose(ra, (0,2,4,3,1)),axis=1)
    ra = (ra-ra.min())/(ra.max()-ra.min())
    nn_opt_path = join('/home/argha/github/mmPhase/milliEgo/models', "cross-mio_turtle_v1", 'nn_opt.json')
    with open(nn_opt_path) as handle:
        nn_opt = json.loads(handle.read())
    network_model = build_model_cross_att(join('/home/argha/github/mmPhase/milliEgo/models', "cross-mio_turtle_v1", str(10)),
                                        imu_length=IMU_LENGTH, mask_att=nn_opt['cross_att_type'], istraining=False)
    

    vel_x, vel_y = predictionF(ra, imu, network_model, sess)
    abs_vel = np.sqrt(vel_x**2 + vel_y**2)/(0.0000573059281*2)  # Error min
    return np.abs(abs_vel.mean())


if __name__ == "__main__":
    erros = []
    dict_list = []
    os.system("hostname")
    files = glob.glob("datasets/*.pickle")
    files.remove("datasets/2024-03-29_vicon_135.pickle")
    files.remove("datasets/2024-03-29_vicon_210.pickle")

    with K.get_session() as sess:
        for f in files:
            val = collect_ra_imu(f, sess)
            data_dict = {'filename':f, 'estimate': val}
            erros.append(val)
            dict_list.append(data_dict)

    with open('milliegoestimate.json', 'w') as file:
        json.dump(dict_list, file)
