import pandas as pd
import numpy as np
import json
import os
import scipy.io.wavfile
import matplotlib.pyplot as plt
from IPython.display import Audio
import IPython.display as ipd
import scipy.misc
from scipy.stats import skew, kurtosis
import librosa

import librosa.display
from sklearn.utils import shuffle


def feat_mel_freq(y, hop_length, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mel_freq_features = np.round(
        np.array([np.mean(mfcc[0]), np.std(mfcc[0]), np.amin(mfcc[0]), np.amax(mfcc[0]), np.median(mfcc[0]),
                  np.mean(mfcc[1]), np.std(mfcc[1]), np.amin(mfcc[1]), np.amax(mfcc[1]), np.median(mfcc[1]),
                  np.mean(mfcc[2]), np.std(mfcc[2]), np.amin(mfcc[2]), np.amax(mfcc[2]), np.median(mfcc[2]),
                  np.mean(mfcc[3]), np.std(mfcc[3]), np.amin(mfcc[3]), np.amax(mfcc[3]), np.median(mfcc[3]),
                  np.mean(mfcc[4]), np.std(mfcc[4]), np.amin(mfcc[4]), np.amax(mfcc[4]), np.median(mfcc[4]),
                  np.mean(mfcc[5]), np.std(mfcc[5]), np.amin(mfcc[5]), np.amax(mfcc[5]), np.median(mfcc[5]),
                  np.mean(mfcc[6]), np.std(mfcc[6]), np.amin(mfcc[6]), np.amax(mfcc[6]), np.median(mfcc[6]),
                  np.mean(mfcc[7]), np.std(mfcc[7]), np.amin(mfcc[7]), np.amax(mfcc[7]), np.median(mfcc[7]),
                  np.mean(mfcc[8]), np.std(mfcc[8]), np.amin(mfcc[8]), np.amax(mfcc[8]), np.median(mfcc[8]),
                  np.mean(mfcc[9]), np.std(mfcc[9]), np.amin(mfcc[9]), np.amax(mfcc[9]), np.median(mfcc[9]),
                  np.mean(mfcc[10]), np.std(mfcc[10]), np.amin(mfcc[10]), np.amax(mfcc[10]), np.median(mfcc[10]),
                  np.mean(mfcc[11]), np.std(mfcc[11]), np.amin(mfcc[11]), np.amax(mfcc[11]), np.median(mfcc[11]),
                  np.mean(mfcc[12]), np.std(mfcc[12]), np.amin(mfcc[12]), np.amax(mfcc[12]), np.median(mfcc[12]),
                  np.mean(mfcc_delta[0]), np.std(mfcc_delta[0]), np.amin(mfcc_delta[0]), np.amax(mfcc_delta[0]),
                  np.median(mfcc_delta[0]),
                  np.mean(mfcc_delta[1]), np.std(mfcc_delta[1]), np.amin(mfcc_delta[1]), np.amax(mfcc_delta[1]),
                  np.median(mfcc_delta[1]),
                  np.mean(mfcc_delta[2]), np.std(mfcc_delta[2]), np.amin(mfcc_delta[2]), np.amax(mfcc_delta[2]),
                  np.median(mfcc_delta[2]),
                  np.mean(mfcc_delta[3]), np.std(mfcc_delta[3]), np.amin(mfcc_delta[3]), np.amax(mfcc_delta[3]),
                  np.median(mfcc_delta[3]),
                  np.mean(mfcc_delta[4]), np.std(mfcc_delta[4]), np.amin(mfcc_delta[4]), np.amax(mfcc_delta[4]),
                  np.median(mfcc_delta[4]),
                  np.mean(mfcc_delta[5]), np.std(mfcc_delta[5]), np.amin(mfcc_delta[5]), np.amax(mfcc_delta[5]),
                  np.median(mfcc_delta[5]),
                  np.mean(mfcc_delta[6]), np.std(mfcc_delta[6]), np.amin(mfcc_delta[6]), np.amax(mfcc_delta[6]),
                  np.median(mfcc_delta[6]),
                  np.mean(mfcc_delta[7]), np.std(mfcc_delta[7]), np.amin(mfcc_delta[7]), np.amax(mfcc_delta[7]),
                  np.median(mfcc_delta[7]),
                  np.mean(mfcc_delta[8]), np.std(mfcc_delta[8]), np.amin(mfcc_delta[8]), np.amax(mfcc_delta[8]),
                  np.median(mfcc_delta[8]),
                  np.mean(mfcc_delta[9]), np.std(mfcc_delta[9]), np.amin(mfcc_delta[9]), np.amax(mfcc_delta[9]),
                  np.median(mfcc_delta[9]),
                  np.mean(mfcc_delta[10]), np.std(mfcc_delta[10]), np.amin(mfcc_delta[10]), np.amax(mfcc_delta[10]),
                  np.median(mfcc_delta[10]),
                  np.mean(mfcc_delta[11]), np.std(mfcc_delta[11]), np.amin(mfcc_delta[11]), np.amax(mfcc_delta[11]),
                  np.median(mfcc_delta[11]),
                  np.mean(mfcc_delta[12]), np.std(mfcc_delta[12]), np.amin(mfcc_delta[12]), np.amax(mfcc_delta[12]),
                  np.median(mfcc_delta[12])]), 4)

    return mel_freq_features


def feat_f0(y):
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0_features = np.round(
        np.array([np.amin(f0), np.amax(f0), np.mean(f0), np.std(f0), np.median(f0), kurtosis(f0), skew(f0)]), 4)
    return (f0_features)


def read_path(path):
    return os.listdir(path)


def read_audio(path, sr):
    y, sr = librosa.load(path, sr=sr, mono=True, offset=0.8, duration=None)
    lib_data = {y:y, sr:sr}
    return lib_data


def gen_feat(sex, path, hop_length):
    data = {"f0": [], "mfcc": [], "spec": [], "gender": sex}
    datanames = read_path(path)
    for dataname in datanames:
        if os.path.splitext(dataname)[1] == ('.m4a' or '.wav'):
            audio_path = path + "/" + dataname
            lib_data = read_audio(audio_path, sr)
            y, sr = lib_data['y'], lib_data['sr']
            f0 = feat_f0(y)
            mfcc = feat_mel_freq(y, hop_length, sr)
            data['f0'].append(f0)
            data['mfcc'].append(mfcc)
    return data


def gen_df(data_dic):
    d = pd.DataFrame(data_dic)
    m1 = d['f0'].apply(pd.Series,
                       index=['f0_min', 'f0_max', 'f0_mean', 'f0_std', 'f0_median', 'f0_kurtosis', 'f0_skew'])
    m2 = d['mfcc'].apply(pd.Series)
    m2['gender'] = d['gender']
    df = pd.concat([m1, m2], axis=1)
    return df


def df2csv(df):
    df_csv = df.to_csv(index=False)
    return df_csv


def merge_df(df_list):
    df = pd.Dataframe()
    for i in len(df_list):
        df.append(df_list[i])
    return df


def feat_engineering(path, sex, hop_length):
    feats = gen_feat(sex, path, hop_length)
    df = gen_df(feats)
    df_shuffled = shuffle(df)
    return df_shuffled


#
#
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# import xgboost as xgb
# from sklearn import metrics
#
# X = df.iloc[:, :-1]
# y = df.iloc[:, -1]
# scaler = StandardScaler()
# scaler.fit(X)
# X = scaler.transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
#
# # xgboost model
# gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_train, y_train)
# y_pred = gbm.predict(X_test)
#
# # Test Accuracy (xgboost)
# print(metrics.accuracy_score(y_test, y_pred))
#
# importance_all = pd.DataFrame()
# for importance_type in ('weight', 'gain', 'cover', 'total_gain', 'total_cover'):
#     importance = gbm.get_booster().get_score(importance_type=importance_type)
#     keys = list(importance.keys())
#     values = list(importance.values())
#     df_importance = pd.DataFrame(data=values, index=keys, columns=[importance_type])
#     importance_all = pd.concat([importance_all, df_importance], axis=1)
# importance_all
#
# from sklearn.svm import SVC
#
# svc = SVC()
# svc.fit(X_train, y_train)
# y_pred = svc.predict(X_test)
# metrics.accuracy_score(y_test, y_pred)
#
# # @title NN
# import tensorflow as tf
#
# inputs = tf.keras.Input(shape=(X.shape[1],))
#
# x = tf.keras.layers.Dense(64, activation='relu')(inputs)
# x = tf.keras.layers.Dense(64, activation='relu')(x)
#
# outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
# model = tf.keras.Model(inputs, outputs)
#
# model.summary()
#
# model.compile(
#     optimizer='adam',
#     loss='binary_crossentropy',
#     metrics=[
#         'accuracy',
#         tf.keras.metrics.AUC(name='auc')
#     ]
# )
# history = model.fit(
#     X_train,
#     y_train,
#     validation_split=0.2,
#     batch_size=32,
#     epochs=100,
#     callbacks=[
#         tf.keras.callbacks.EarlyStopping(
#             monitor='val_loss',
#             patience=3,
#             restore_best_weights=True
#         )
#     ]
# )
#
# model.evaluate(X_test, y_test)
#
# # @title NN dropped out
# inputs = tf.keras.Input(shape=(X.shape[1],))
#
# x = tf.keras.layers.Dense(128, activation='relu')(inputs)
# x = tf.keras.layers.Dense(128, activation='relu')(x)
# x = tf.keras.layers.Dropout(.2, input_shape=(128,))(x)
#
# x = tf.keras.layers.Dense(64, activation='relu')(x)
# x = tf.keras.layers.Dropout(.2, input_shape=(64,))(x)
#
# outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
# model = tf.keras.Model(inputs, outputs)
#
# model.summary()
#
# model.compile(
#     optimizer='adam',
#     loss='binary_crossentropy',
#     metrics=[
#         'accuracy',
#         tf.keras.metrics.AUC(name='auc')
#     ]
# )
# history = model.fit(
#     X_train,
#     y_train,
#     validation_split=0.3,
#     batch_size=32,
#     epochs=100,
#     callbacks=[
#         tf.keras.callbacks.EarlyStopping(
#             monitor='val_loss',
#             patience=3,
#             restore_best_weights=True
#         )
#     ]
# )
#
# model.evaluate(X_test, y_test)
#
# # y 对应时间的amplitude
# import matplotlib.pyplot as plt
#
# fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
# D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
# img = librosa.display.specshow(D, y_axis='linear', x_axis='time',
#                                sr=sr, ax=ax[0])
# ax[0].set(title='Linear-frequency power spectrogram')
# ax[0].label_outer()
#
# hop_length = 1024
# D = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length)), ref=np.max)
# librosa.display.specshow(D, y_axis='log', sr=sr, hop_length=hop_length, x_axis='time', ax=ax[1])
# ax[1].set(title='Log-frequency power spectrogram')
# ax[1].label_outer()
# fig.colorbar(img, ax=ax, format="%+2.f dB")
#
# D
#
#
# # @title freqs, peak freq
# def extract_peak_frequency(data, sampling_rate):
#     fft_data = np.fft.fft(data)
#     freqs = np.fft.fftfreq(len(data))
#     print(np.mean(freqs), np.std(freqs), np.amin(freqs), np.amax(freqs), np.median(freqs), np.percentile(freqs, 25),
#           np.percentile(freqs, 75), (np.percentile(freqs, 75) - np.percentile(freqs, 25)), skew(freqs), kurtosis(freqs))
#     peak_coefficient = np.argmax(np.abs(fft_data))
#     print(peak_coefficient)
#     peak_freq = freqs[peak_coefficient]
#     print(peak_freq)
#
#     return abs(peak_freq * sampling_rate)
#
#
# extract_peak_frequency(y, sr)
