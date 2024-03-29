import pandas as pd
import numpy as np
import os
from scipy.stats import skew, kurtosis
import librosa
import librosa.display
import warnings
warnings.filterwarnings('ignore')

def feat_mel_freq(y, hop_length, sr):
    """generate mfcc relevant features"""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13, n_fft=441)
    # mfcc_delta = librosa.feature.delta(mfcc)
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
                  np.mean(mfcc[12]), np.std(mfcc[12]), np.amin(mfcc[12]), np.amax(mfcc[12]), np.median(mfcc[12])]), 4)
    return mel_freq_features


def feat_f0(y):
    """generate f0 relevant features"""
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0_features = np.round(
        np.array([np.amin(f0), np.amax(f0), np.mean(f0), np.std(f0), np.median(f0), kurtosis(f0), skew(f0)]), 4)
    return (f0_features)


def read_path(path):
    """read files as a list under path"""
    return os.listdir(path)


def read_audio(path, sr):
    """read audio from path using librosa, only remain first 6 seconds"""
    y, sr = librosa.load(path, sr=sr, mono=True, duration=6.0)
    lib_data = {"y": y, "sr": sr}
    return lib_data


def read_audio_female(path, sr):
    """read audio with windowing, remain all"""
    length = librosa.get_duration(filename=path)
    ys = []
    if length < 7:
        y, sr = librosa.load(path, sr=sr, mono=True, duration=6.0)
        ys.append(y)
    else:
        for i in np.arange(0.0, length, 0.2):
            y, sr = librosa.load(path, sr=sr, mono=True, duration=6.0, offset=i)
            ys.append(y)
    return ys


def gen_feat(path, hop_length, sr):
    """generate all features"""
    folders = read_path(path)
    data_fin = {"f0": [], "mfcc": [], "gender": []}

    for type in folders:
        current_path = path + "/" + type
        datanames = read_path(current_path)
        for dataname in datanames:
            if os.path.splitext(dataname)[1] == ('.m4a' or '.wav' or '.mp3'):
                audio_path = current_path + "/" + dataname
                if type == "females":
                    female_ys = read_audio_female(audio_path, sr)
                    for y in female_ys:
                        f0 = feat_f0(y)
                        mfcc = feat_mel_freq(y, hop_length, sr)
                        data_fin['f0'].append(f0)
                        data_fin['mfcc'].append(mfcc)
                        data_fin['gender'].append(0)

                    # print(f'length in female total:{len(data["ys"])}') #162

                if type == "males":
                    data_fin['gender'].append(1)
                    lib_data = read_audio(audio_path, sr)
                    y = lib_data['y']
                    # data['ys'].append(y)
                    f0 = feat_f0(y)
                    mfcc = feat_mel_freq(y, hop_length, sr)
                    data_fin['f0'].append(f0)
                    data_fin['mfcc'].append(mfcc)



    # for y in data:
    #     print(y)
        # f0 = feat_f0(y)
        # mfcc = feat_mel_freq(y, hop_length, sr)
        # data_fin['f0'].append(f0)
        # data_fin['mfcc'].append(mfcc)
        # data_fin['gender'].append(sex)

    return data_fin


def gen_df(data_dic):
    """transform features to dataframe"""
    d = pd.DataFrame(data_dic)
    m1 = d['f0'].apply(pd.Series,
                       index=['f0_min', 'f0_max', 'f0_mean', 'f0_std', 'f0_median', 'f0_kurtosis', 'f0_skew'])
    mfcc_indice = [x for x in range(13)]
    stats = ['mean', 'std', 'min', 'max', 'median']
    cols = []
    for i in mfcc_indice:
        for s in stats:
            cols.append("mfcc_" + str(i) + "_" + s)
    m2 = d['mfcc'].apply(pd.Series, index=cols)
    m2['gender'] = d['gender']
    df = pd.concat([m1, m2], axis=1)
    return df


def df2csv(df):
    """transform dataframe to csv"""
    df.to_csv('./df.csv')


def feat_engineering(path, hop_length=220, sr=22050):
    """initial function"""
    data = gen_feat(path, hop_length, sr)
    df = gen_df(data)
    df2csv(df)


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 0:
        print("please run code")
    elif len(sys.argv) == 1:
        print("please input project path")
    elif len(sys.argv) == 2:
        feat_engineering(sys.argv[1])
    elif len(sys.argv) == 3:
        feat_engineering(sys.argv[1], int(sys.argv[2]))
    elif len(sys.argv) == 4:
        feat_engineering(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    else:
        print("wrong input")



