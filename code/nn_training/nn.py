import time

import numpy as np
import os
import pandas as pd
import json 
import math

from keras.callbacks import ModelCheckpoint, EarlyStopping, LambdaCallback
import keras.backend as K
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras import optimizers
import tensorflowjs as tfjs
import tensorflow as tf

from nftutils import file_age_in_seconds, pickle_dump, pickle_load
from params import *


def matthews_correlation(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


if use_magnitude:
    num_features += 1
if use_angles:
    num_features = num_features + 2
if use_derivatives:
    if use_derivatives_angles: num_features = num_features + 3
    else: num_features = num_features + 6

learning_rates = [0.001, 0.01, 0.1, 1.0, 0.0001, 1e-5, 1e-6, 1e-7]
best_corcoeff = -1.0
max_iteration = 10

sampling_rate = 50
train_path = dataset_path + 'train/'
test_path = dataset_path + 'test/'
export_tflite_filename = "tflite_play__.tflite"
weights_untrained_filename = 'weights_untrained.h5'

col_names = ["time", "acc_x", "acc_y", "acc_z"]
X_tmp = None

def evaluate_results(predictions, target, print_mode = False):
    compare = np.empty((len(predictions), 2), dtype=object)
    compare[:, 0] = np.atleast_2d(predictions).T
    compare[:, 1] = target.T
    if print_mode: print('Predictions vs target', compare)

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for i in range(len(predictions)):
        if compare[i, 0] > 0.5 and compare[i, 1] == 1:
            true_positives += 1

        if compare[i, 0] < 0.5 and compare[i, 1] == 0:
            true_negatives += 1

        if compare[i, 0] < 0.5 and compare[i, 1] == 1:
            false_negatives += 1

        if compare[i, 0] > 0.5 and compare[i, 1] == 0:
            false_positives += 1

    positives = sum(target)
    negatives = len(predictions) - positives

    print('----------')
    print('true_positives', true_positives, positives)
    print('false_negatives', false_negatives, positives)
    print()
    print('true_negatives', true_negatives, negatives)
    print('false_positives', false_positives, negatives)
    print('----------')

def import_preprocessed_folder(path_folder):
    global X_tmp
    mat_x = np.empty(shape=(seq_length, num_features),dtype='object')

    vett_y = np.zeros(1)
    files_list = [f for f in os.listdir(path_folder) if "_pre.txt" in f]

    for file in files_list:
            #print(os.path.join(path_folder, file))

            read_file_pre = pd.read_csv(os.path.join(path_folder, file), delim_whitespace=True, header=None, names=col_names)
            read_file_post = pd.read_csv(os.path.join(path_folder, file.replace("_pre","_post")), delim_whitespace=True, header=None, names=col_names)

            k = read_file_pre.to_numpy()
            k2 = read_file_post.to_numpy()
            k = np.vstack((k, k2[:int(additional_time * sampling_rate), :]))

            if use_magnitude: k = np.hstack((k, np.atleast_2d(np.linalg.norm(k[:, 1:], axis=1)).T))

            roll = np.arctan2(k[:, 2], k[:, 3]) * 180/math.pi
            pitch = np.arctan2(-k[:,1], np.sqrt(k[:,2]*k[:,2] + k[:,3]*k[:,3])) * 180/math.pi

            roll = np.reshape(roll, (roll.shape[0],1))
            pitch = np.reshape(pitch, (roll.shape[0],1))

            if use_angles:
                k = np.hstack((k, roll))
                k = np.hstack((k, pitch))

            if use_derivatives:
                if use_derivatives_angles:
                    k = np.hstack((k, np.expand_dims(np.diff(np.append(k[:,5],0)),axis = 1) ))
                    k = np.hstack((k, np.expand_dims(np.diff(np.append(k[:,6],0)),axis = 1) ))
                    k = np.hstack((k, np.expand_dims(np.diff(np.append(k[:,7],0)),axis = 1) ))
                else:
                    k = np.hstack((k, np.expand_dims(np.diff(np.append(k[:,1],0)),axis = 1) ))
                    k = np.hstack((k, np.expand_dims(np.diff(np.append(k[:,2],0)),axis = 1) ))
                    k = np.hstack((k, np.expand_dims(np.diff(np.append(k[:,3],0)),axis = 1) ))

                    k = np.hstack((k, np.expand_dims(np.diff(np.append(k[:,-3],0)),axis = 1) ))
                    k = np.hstack((k, np.expand_dims(np.diff(np.append(k[:,-3],0)),axis = 1) ))
                    k = np.hstack((k, np.expand_dims(np.diff(np.append(k[:,-3],0)),axis = 1) ))

                k = k[:-2, :]

            X0 = k[0:k.shape[0] - istanti_finali_scartati, 1:]
            n_rows = X0.shape[0]

            X_fin = X0[n_rows-seq_length:,  :]
            X_past = X0[:n_rows-seq_length, :]

            if not rnn:
                mat_x = np.vstack((mat_x, X_past))
                vett_y = np.concatenate((vett_y, np.zeros(X_past.shape[0])))

                if mat_x.shape[0] == seq_length + X_past.shape[0]:
                    mat_x = mat_x[seq_length:, :]
                    vett_y = vett_y[1:]

                mat_x = np.vstack((mat_x, X_fin))
                vett_y = np.concatenate((vett_y, np.ones(X_fin.shape[0])), axis=0)

            if rnn:
                n_blocks = X_past.shape[0]//seq_length
                counter = 0

                for i in range(0, n_blocks+1):
                    block_data = X_past[X_past.shape[0]-seq_length*i-seq_length: X_past.shape[0]-seq_length*i, :]

                    if block_data.shape[0] < seq_length and block_data.shape[0] != 0:
                        continue

                    if block_data.shape[0] == 0:
                        pass

                    else:
                        mat_x = np.dstack((mat_x, block_data))
                        counter = counter + 1

                Y = np.zeros(counter)
                vett_y = np.concatenate((vett_y, Y))

                mat_x = np.dstack((mat_x, X_fin))
                vett_y = np.concatenate((vett_y, [1]))

            if X_tmp is None: X_tmp = X_fin

    return mat_x, vett_y

# fix random seed for reproducibility
np.random.seed(7)
experiment_log = "experiment.log"
with open(experiment_log, "w") as myfile:
    myfile.truncate()

def save_params():
    params = {'rnn':rnn, 'lr':learn_rate, 'additional_time':additional_time, 'seq_length':seq_length, 'dense_units':dense_units, 'lstm_units':lstm_units}
    pickle_dump(params, mc_file+".params")
    return params

def save_params_with_new_best(logs):
    if file_age_in_seconds(mc_file) < 1:
        with open(experiment_log, "a") as file:
            file.write(str(logs['val_matthews_correlation']) + ": " + json.dumps(save_params())+"\n")
        print("Saved params")

# create the model
mc_file = 'best_model_mcc.h5'
mc = ModelCheckpoint(mc_file, monitor='val_matthews_correlation', mode='max', verbose=1, save_best_only=True)
lc = LambdaCallback(
    on_epoch_end=lambda epoch, logs: save_params_with_new_best(logs)
)

iteration = 0
while iteration < max_iteration:
    print("=== Training "+str(iteration)+"th model ===")
    seq_length = np.random.randint(30, 100)
    additional_time = np.random.uniform(0.10, 0.80) * seq_length * 1.0 / sampling_rate
    lstm_units = np.random.randint(1, 10)
    dense_units = np.random.randint(2, 50)

    mat_x_train, vett_y_train = import_preprocessed_folder(train_path)

    if rnn:
        # Removing dummy none plane
        mat_x_train = mat_x_train[:, :, 1:]
        vett_y_train = vett_y_train[1:vett_y_train.shape[0]]
        x_training = mat_x_train.transpose(2, 0, 1)
    else:
        x_training = mat_x_train

    train_positive = sum(vett_y_train)
    train_negative = len(vett_y_train) - train_positive

    mat_x_test, vett_y_test = import_preprocessed_folder(test_path)

    if rnn:
        mat_x_test = mat_x_test[:, :, 1:]
        vett_y_test = vett_y_test[1:vett_y_test.shape[0]]
        x_test = mat_x_test.transpose(2, 0, 1)
    else:
        x_test = mat_x_test

    if data_visualization:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        neg_features = mat_x_train[vett_y_train == 0, :]
        pos_features = mat_x_train[vett_y_train == 1, :]

        ax.scatter(neg_features[:, 0], neg_features[:, 1], neg_features[:, 2], c='green')
        ax.scatter(pos_features[:, 0], pos_features[:, 1], pos_features[:, 2], c='red')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    cclasses = np.unique(vett_y_train)
    # x_training = x_training.reshape([-1,x_training.shape[0], x_training.shape[1],x_training.shape[2]])
    cweights = class_weight.compute_class_weight('balanced',
                                                 cclasses,
                                                 vett_y_train)

    model = Sequential()

    if rnn:
        model.add(LSTM(units = lstm_units, input_shape = (seq_length, num_features), return_sequences = False))
        model.add(Dense(dense_units))
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(dense_units, input_shape = (x_training.shape[1],), activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))

    model.save_weights(weights_untrained_filename)

    for learn_rate in learning_rates:
        opt = optimizers.Adam(learn_rate)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[matthews_correlation])
        model.load_weights(weights_untrained_filename)

        a = np.asarray(x_training).astype(np.float32)
        b = np.atleast_2d(np.asarray(vett_y_train).astype(np.float32)).T
        a_test = np.asarray(x_test).astype(np.float32)
        b_test = np.atleast_2d(np.asarray(vett_y_test).astype(np.float32)).T

        es = EarlyStopping(monitor='val_matthews_correlation', mode='max', verbose=1, patience=30)
        model.fit(a, b, epochs=200, shuffle=True, validation_data=(a_test, b_test), callbacks=[mc,es,lc], class_weight={cclasses[0]: cweights[0], cclasses[1]: cweights[1]})
        time.sleep(2)

        # Final evaluation of the model
        scores = model.evaluate(x_test.astype(np.float32), np.atleast_2d(vett_y_test).T, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))
    iteration += 1


print("Now, reload best model and export!")
best_params = pickle_load(mc_file+".params")
seq_length = best_params['seq_length']
loaded_model = Sequential()
if rnn:
    loaded_model.add(LSTM(units = best_params['lstm_units'], input_shape = (seq_length, num_features), return_sequences = False))
    loaded_model.add(Dense(best_params['dense_units']))
    loaded_model.add(Dense(1, activation='sigmoid'))
else:
    model.add(Dense(dense_units, input_shape=(best_params['dense_units'],), activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

# load weights into new model
loaded_model.load_weights(mc_file)
print("Loaded model from disk")

loaded_model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=[matthews_correlation])

converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
converter.allow_custom_ops=True
converter.experimental_new_converter =True

with open(export_tflite_filename, "wb+") as ff:
    ff.write(converter.convert())
