#
# Created on 9/17/2019
#
# @author Seyed
#
# Email: mousavikahaki@gmail.com
#

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
from keras.constraints import maxnorm
import seaborn as sns
import scipy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.layers import (Concatenate, Conv3D, Dropout, Input, BatchNormalization, Flatten,
                          Dense, MaxPooling3D, UpSampling3D, Activation, Reshape, Lambda,
                           Permute)
from keras.callbacks import TensorBoard
import datetime
import scipy.io as sio
import AT_Classes as Classes
from keras.callbacks import LearningRateScheduler
import math


def step_decay(epoch, lr):
    drop = 0.5
    epochs_drop = 2.0
    lrate = lr * math.pow(drop,math.floor((1+epoch)/epochs_drop))
    return lrate

lrate = LearningRateScheduler(step_decay)


# Define Convolutions Block
def convolutionblock(x, name, fms, params):
    x = Conv3D(filters=fms, **params, name=name+"_conv0")(x)
    x = BatchNormalization(name=name+"_bn0")(x)
    x = Activation("relu", name=name+"_relu0")(x)

    x = Conv3D(filters=fms, **params, name=name+"_conv1")(x)
    x = BatchNormalization(name=name+"_bn1")(x)
    x = Activation("relu", name=name)(x)
    return x



import tensorflow as tf

# from tensorflow.keras import optimizers
#
# initial_learning_rate = 0.1
# lr_schedule = optimizers.schedules.ExponentialDecay(
#     initial_learning_rate,
#     decay_steps=100000,
#     decay_rate=0.96,
#     staircase=True)




lst_useImage = ['True']             # Using the second tower or not, Default = ['True','False']
useEndpointFeatures = 'True'        # Use the proposed endpoint features
kernel_initializer='orthogonal'     # Kernel Initializer, Methods: orthogonal, HeNormal -> ReLu, Identity, LecunUniform, LecunNormal, Xavier - >sigmoid
rotation_degrees = [0,90,180,270]
flips = ['right']
UseConv = True
maxNumPoints = 12                   # Limit the maximum number of merging points

root_dir = 'E:/AutomatedTracing/TraceProofreading/TraceProofreading'

rotated_IMs = np.zeros([13,13,13,len(rotation_degrees)])

# Using the second tower or not
for UseIMage in lst_useImage:
    # Using one-leave-out strategy,
    # Choose the stack number to test
    # It will ignore the test stack in the Training
    for ImagetoTest in [1]:
        # Run multiple times to get variation in results
        for run in range(2):

            epoch  = 150             # number of epochs
            batch_size = 50         # batch size, Defualt = 50
            verbose= 1              # verbose=1 will show you an animated progress bar
            # doSMOTE = False         # Replicate data using SMOTE method, calculate k-nearest neighbors, N examples randomly selected from its knn, generate new data between samples
            learning_Rate = 0.001   # Learning Rate, # Default  =0.001

            # Define now time for unique names in the model
            T = datetime.datetime.today()
            nowTimeDate = T.strftime("%b_%d_%H_%M")
            # PltNAme = 'AT_XYZ_is5_'+str(useXYZ_Positions)+'_'+str(ImagetoTest)+'_run=2'+nowTimeDate

            PltNAme = 'TEMP22_Drop_20_3LAyerHalf81616_kernel555_S1_2_SmUnet1TEST20INV_FEATURES_CONV=' + str(UseConv) + '_LR=' + str(learning_Rate) + '_100_sce_' + str(
                kernel_initializer) + '_IM=' + str(ImagetoTest) + 'bchSiz=' + str(batch_size) + '_Use_IM=' + str(
                UseIMage) + '_Epoch=' + str(
                epoch) + '_run=' + str(run + 1) + '_' +nowTimeDate
            print(PltNAme)

            # # Use this for Dataset 1
            # or use (should be the same)
            # E:\AutomatedTracing\TraceProofreading\TraceProofreading\data\datafeed\IMonce_limit100scen_NEW_Inv_FEATURES.mat
            # filepath =
            # 'E:\AutomatedTraceResults\DataForConnectingTraining\Data_For_AE_BranchScenarios\IMonce_limit100scen_NEW_Inv_FEATURES.mat'

            # # Use this for Dataset 2
            # filepath =
            # 'E:\AutomatedTraceResults\DataForConnectingTraining\Data_For_AE_BranchScenarios\S2B_IMonce_100_scen_NEW_Inv_FEATURES_User=SK.mat'

            # Use this for combination of Datasets 1 and 2
            filepath = \
                'E:\AutomatedTraceResults\DataForConnectingTraining\Data_For_AE_BranchScenarios\S1and2_IMonce_100_scen_NEW_Inv_FEATURES_User=SK.mat'

            # Load scienaros information
            ScenariosData = sio.loadmat(filepath)

            # Get Image numbers
            IMnums = ScenariosData['IMnum']
            # IMnum = IMnums[0,IMnums.shape[1]-1]
            # IMnums.shape

            # Get Features
            Features = ScenariosData['NewFeatures']
            # Feature = Features[0,Features.shape[1]-1]
            # Feature.shape

            # Get Image Intensities Data
            IMs = ScenariosData['IMs']
            # IMtmp = IMs[0,IMs.shape[1]-1]
            # IMtmp.shape

            # Get Scenarios Information
            Scenarios = ScenariosData['Scenarios']
            # Scenarios.shape
            # Scenarios[0,Scenarios.shape[1]-1]

            # Get Labels for Training
            Labels = ScenariosData['Labels']
            # Labels[0,Labels.shape[1]-1]

            # Clear Data for Each Run
            IMsTrain = []
            FeatureTrain = []
            LabelsTrain = []
            ScenariosTrain = []
            IMsTest = []
            FeatureTest = []
            LabelsTest = []
            ScenariosTest = []

            # Use Uper triangle in matrix,
            # the scienario matrix is symmetric matrix but you may use all data
            UseUpper = False
            numScenarios = Scenarios.shape
            counter = 0

            # Generating Scienaros and Feature Data
            for i in range(numScenarios[1]):
                scenario = Scenarios[0,i]

                IM = IMs[0,i]
                # print(IM.shape)
                Feature = Features[0, i]

                # if scenario.shape[0] == 3:
                #     maxNumPoints = 3

                # for r in range(len(rotation_degrees)):
                #     degree = rotation_degrees[r]
                #     print(degree)
                #     rotated_IMs[:,:,:,r] = scipy.ndimage.interpolation.rotate(IM, degree, mode='nearest', reshape=False)
                #     # IM_Proj = Classes.IM3D.Z_Projection(rotated_IMs[:,:,:,r])
                #     # Classes.IM3D.plt(IM_Proj)

                Label = Labels[0, i]
                # print(scenarios.shape)
                # if scenario.any():
                # scenarios.shape[2]

                # Geting Scenario data from matrix (upper or all)
                S = Classes.cl_scenario(maxNumPoints, scenario.shape[0],scenario,0)
                if UseUpper:
                    scenario_arr = S.getUpperArr()
                else:
                    scenario_arr = S.getWholeArr()

                # Exclude the Test Image in Training
                # Generating Training and Test Data
                if IMnums[0, i] != ImagetoTest:
                    ScenariosTrain.append(scenario_arr)
                    IMsTrain.append(IM)
                    FeatureTrain.append(Feature)
                    LabelsTrain.append(Label)
                else:
                    ScenariosTest.append(scenario_arr)
                    IMsTest.append(IM)
                    FeatureTest.append(Feature)
                    LabelsTest.append(Label)

            # Numpy array creation
            ScenariosTrain = np.asarray(ScenariosTrain, dtype=np.float)
            IMsTrain = np.asarray(IMsTrain, dtype=np.float)
            IMsTrain3D = IMsTrain
            FeatureTrain = np.asarray(FeatureTrain, dtype=np.float)
            FeatureTrain = FeatureTrain[:,0,:]
            LabelsTrain = np.asarray(LabelsTrain, dtype=np.float)
            LabelsTrain = LabelsTrain[:,0]
            LabelsTrain = LabelsTrain[:,0]
            IMTrain = np.reshape(IMsTrain, [IMsTrain.shape[0],np.product(IMsTrain[0,:,:,:].shape)])

            # Shuffling Data
            indices = np.arange(len(ScenariosTrain))
            np.random.shuffle(indices)
            ScenariosTrain = ScenariosTrain[indices]
            IMsTrain = IMsTrain[indices]
            FeatureTrain = FeatureTrain[indices]

            LabelsTrain = LabelsTrain[indices]

            print(IMTrain.shape)
            print(FeatureTrain.shape)
            print(ScenariosTrain.shape)
            print(LabelsTrain.shape)

            # Plot data count
            # z_train = Counter(yIMs_train)
            # sns.countplot(yIMs_train)

            # to ignore image data
            if UseIMage == False:
                IMTrain = np.zeros(IMTrain.shape)
                IMsTest = np.zeros(IMsTest.shape)
                print('Not Using Image')

            # this goes for using Conv or sequential method for image featuers
            if UseConv:
                # input1 = Input(shape=(13, 13, 13, 1))
                # x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(
                #     input1)  # change to leakyRelu to avoid dead neurons
                # x = Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
                # x = MaxPooling3D((2, 2, 2))(x)
                # x2 = Flatten()(x)


                input1 = Input(shape=(13, 13, 13, 1), name="inputs")

                params = dict(kernel_size=(5, 5, 5), activation=None,
                              padding="same", kernel_initializer=kernel_initializer)

                # Transposed convolution parameters
                # params_trans = dict(kernel_size=(2, 2, 2), strides=(1, 1, 1), padding="same")

                # BEGIN - Encoding path
                # encodeA = ConvolutionBlock(input1, "encodeA", fms, params)

                # First Encoder
                name = "encodeA"
                x = Conv3D(filters=8, **params, name=name + "_conv0")(input1) # ** to use a dictionary instead of keyword in the function
                x = BatchNormalization(name=name + "_bn0")(x)
                x = Dropout(.20)(x)
                x = Activation("relu", name=name + "_relu0")(x)
                # x = Conv3D(filters=4, **params, name=name + "_conv1")(x)
                # x = BatchNormalization(name=name + "_bn1")(x)
                # encodeA = Activation("relu", name=name)(x)
                poolA = MaxPooling3D(name="poolA", pool_size=(2, 2, 2))(x)

                # Second Encoder
                name = "encodeB"
                x = Conv3D(filters=16, **params, name=name + "_conv0")(poolA)
                x = BatchNormalization(name=name + "_bn0")(x)
                x = Dropout(.20)(x)
                x = Activation("relu", name=name + "_relu0")(x)
                # x = Conv3D(filters=8, **params, name=name + "_conv1")(x)
                # x = BatchNormalization(name=name + "_bn1")(x)
                # encodeB = Activation("relu", name=name)(x)
                poolB = MaxPooling3D(name="poolB", pool_size=(2, 2, 2))(x)

                # Third Encoder
                name = "encodeC"
                x = Conv3D(filters=16, **params, name=name + "_conv0")(poolB)
                x = BatchNormalization(name=name + "_bn0")(x)
                x = Activation("relu", name=name + "_relu0")(x)
                # # x = Conv3D(filters=16, **params, name=name + "_conv1")(x)
                # # x = BatchNormalization(name=name + "_bn1")(x)
                # # encodeC = Activation("relu", name=name)(x)
                # poolC = MaxPooling3D(name="poolC", pool_size=(2, 2, 2))(x)

                # # Fourth Encoder
                # name = "encodeD"
                # x = Conv3D(filters=32, **params, name=name + "_conv0")(poolC)
                # x = BatchNormalization(name=name + "_bn0")(x)
                # x = Activation("relu", name=name + "_relu0")(x)
                # # x = Conv3D(filters=32, **params, name=name + "_conv1")(x)
                # # x = BatchNormalization(name=name + "_bn1")(x)
                # # encodeC = Activation("relu", name=name)(x)
                x2 = Flatten()(x) #was PoolA
            else:
                # # Relu
                input1 = keras.layers.Input(shape=(IMTrain.shape[1],))
                # ,kernel_regularizer=keras.regularizers.l2(l=0.2)
                x1 = keras.layers.Dense(32, input_dim=IMTrain.shape[1], activation='relu')(input1)
                x2 = keras.layers.Dense(16, input_dim=IMTrain.shape[1], activation='relu')(x1)

            # Second Tower
            input3 = keras.layers.Input(shape=(FeatureTrain.shape[1],))
            xxx0 = keras.layers.Dense(32, input_dim=FeatureTrain.shape[1], activation='relu')(input3)
            xxx1 = keras.layers.Dense(16, input_dim=FeatureTrain.shape[1], activation='relu')(xxx0)
            # xxx2 = keras.layers.Dense(8, input_dim=XFeature_train.shape[1], activation='relu')(xxx1)
            # xxx2 = keras.layers.Dense(16, activation='relu')(xxx1)
            # xxx3 = keras.layers.Dense(8, activation='relu')(xxx2)

            # Combine Towers
            combined = keras.layers.concatenate([x2, xxx1])
            # Relu
            out = keras.layers.Dense(4, activation='sigmoid')(combined)
            # Leaky Relu
            # out = keras.layers.Dense(4)(added)
            # out = keras.layers.LeakyReLU(alpha=0.05)(out)

            out1 = keras.layers.Dense(1, activation='sigmoid')(out)
            model = keras.models.Model(inputs=[input1, input3], outputs=out1)

            # Customized Loss functions
            import keras.backend as K

            def keras_loss_1(y_actual, y_predicted):
                loss_value = K.mean(K.sum(K.square((y_actual-y_predicted)/0.5)))
                return loss_value

            def mean_squared_logarithmic_error(y_true, y_pred):
                first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
                second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
                return K.mean(K.square(first_log - second_log), axis=-1)

            # Compile model
            optimizer = keras.optimizers.Adam(lr=learning_Rate)
            model.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'],
                          )

            # Tensor board Data
            tensorboard = TensorBoard(log_dir="E:/AutomatedTracing/AutomatedTracing/Python/logs/"+PltNAme)
            # tensorboard --logdir=E:\AutomatedTracing\AutomatedTracing\Python\logs
            # python -m tensorboard.main --logdir = E:\AutomatedTracing\AutomatedTracing\Python\logs
            # tensorboard --logdir=C:\Users\Seyed\Documents\TraceProofreading\data\logs
            # http://localhost:6006/#scalars&run=AT_All3&runSelectionState=eyJBVF9Ob0ltYWdlIjpmYWxzZSwiQVRfTm9JbWFnZTEiOmZhbHNlLCJBVF9XaXRoX0ltYWdlMSI6ZmFsc2UsIkFUX1dpdGhfSW1hZ2UyIjpmYWxzZSwiQVRfTm9fSW1hZ2UyIjpmYWxzZSwiQVRfV2l0aF9JbWFnZTMiOmZhbHNlLCJBVF9Ob19JbWFnZTMiOmZhbHNlLCJBVF9Ob19GZWF0dXJlczMiOmZhbHNlLCJBVF9TY2llbmFyaW9Pbmx5IjpmYWxzZSwiQVRfQWxsMyI6ZmFsc2UsIkFUX0FsbF9zaHVmZmxlZCI6ZmFsc2UsIkFUX0FsbF9zaHVmZmxlZDEiOmZhbHNlLCJBVF9BbGxfNSI6ZmFsc2UsIkFUX05vSW1hZ2Vfc2h1ZmZsZWQiOmZhbHNlLCJBVF9JbWFnZV9zaHVmZmxlZCI6ZmFsc2UsIkFUX0ltYWdlX1NodWZmbGVkXyI6ZmFsc2UsIkFUX0ltYWdlX1NodWZmbGVkXzYiOmZhbHNlLCJBVF9JbWFnZV9TaHVmZmxlZF82XzEiOmZhbHNlLCJBVF9JbWFnZV9TaHVmZmxlZF83IjpmYWxzZSwiQVRfSW1hZ2VfU2h1ZmZsZWRfRXh0XzEiOmZhbHNlLCJBVF9JbWFnZV9TaHVmZmxlZF9FeHRfMiI6ZmFsc2UsIkFUX0ltYWdlX1NodWZmbGVkX0V4dF8zIjpmYWxzZSwiQVRfSW1hZ2VfU2h1ZmZsZWRfRXh0XzQiOmZhbHNlLCJBVF9JbWFnZV9TaHVmZmxlZF9FeHRfNSI6ZmFsc2UsIkFUX0ltYWdlX1NodWZmbGVkX0V4dF82IjpmYWxzZSwiQVRfSW1hZ2VfU2h1ZmZsZWRfRXh0XzZuZXciOmZhbHNlLCJBVF9JbWFnZV9TaHVmZmxlZF9FeHRfNm4iOmZhbHNlLCJBVF9JbWFnZV9TaHVmZmxlZF9FeHRfNW4iOmZhbHNlLCJBVF9JbWFnZV9TaHVmZmxlZF9FeHRfNG4iOmZhbHNlLCJBVF9JbWFnZV9TaHVmZmxlZF9FeHRfM24iOmZhbHNlLCJBVF9JbWFnZV9TaHVmZmxlZF9FeHRfMm4iOmZhbHNlLCJBVF9JbWFnZV9TaHVmZmxlZF9FeHRfMW4iOmZhbHNlLCJBVF9iZWZvcmVMb3NzXzEiOmZhbHNlLCJBVF9rZXJhc19sb3NzXzFfMSI6ZmFsc2UsIkFUX2tlcmFzX2xvc3NfMV8xX0ZlYl8yNl8xM180MCI6ZmFsc2UsIkFUX2tlcmFzX2xvc3NfMV8xX0ZlYl8yNl8xM181MCI6ZmFsc2UsIkFUX21lYW5fc3F1YXJlZF9sb2dhcml0aG1pY19lcnJvcl8xX0ZlYl8yNl8xNF8wMSI6ZmFsc2UsIkFUX2tlcmFzX2xvc3NfMV8xX0ZlYl8yNl8xNF8xMyI6ZmFsc2UsIkFUX2JlZm9yZVhZWl8xX0ZlYl8yN18xM18yOCI6ZmFsc2UsIkFUX1hZWl8xX0ZlYl8yN18xM180NCI6ZmFsc2UsIkFUX1hZWl8xX0ZlYl8yN18xNF8xOCI6ZmFsc2UsIkFUX1hZWl8xX0ZlYl8yN18xNF8yMSI6ZmFsc2UsIkFUX1hZWl8xX0ZlYl8yN18xNl81OCI6ZmFsc2UsIkFUX1hZWl8xX0ZlYl8yOF8xMF8xNiI6ZmFsc2UsIkFUX1hZWl8xX0ZlYl8yOF8xMF8yNiI6ZmFsc2UsIkFUX1hZWl9pc19UcnVlXzFfRmViXzI4XzEwXzQxIjpmYWxzZSwiQVRfWFlaX2lzX1RydWVfMV9GZWJfMjhfMTBfNDgiOmZhbHNlLCJBVF9YWVpfaXNfVHJ1ZV8xX0ZlYl8yOF8xMF81NiI6ZmFsc2UsIkFUX1hZWl9pc19GYWxzZV8xX0ZlYl8yOF8xMV8wNyI6ZmFsc2UsIkFUX1hZWl9pc19GYWxzZV8xX0ZlYl8yOF8xMV8xNiI6ZmFsc2UsIkFUX1hZWl9pc19GYWxzZV8xX0ZlYl8yOF8xMV8yMyI6ZmFsc2UsIkFUX1hZWl9pc19UcnVlXzFfRmViXzI4XzExXzMxIjpmYWxzZSwiQVRfWFlaX2lzX1RydWVfMV9GZWJfMjhfMTFfNDQiOmZhbHNlLCJBVF9YWVpfaXNfVHJ1ZV8xX0ZlYl8yOF8xMV81NSI6ZmFsc2UsIkFUX1hZWl9pc19UcnVlXzFfRmViXzI4XzEyXzAyIjpmYWxzZSwiQVRfWFlaX2lzX1RydWVfMV9GZWJfMjhfMTNfMDAiOmZhbHNlLCJBVF9YWVpfaXNfVHJ1ZV8xX0ZlYl8yOF8xM18zMiI6ZmFsc2UsIkFUX1hZWl9pc19UcnVlXzFfRmViXzI4XzE0XzA0IjpmYWxzZSwiQVRfWFlaX2lzX1RydWVfMV9NYXJfMDJfMDlfMjkiOmZhbHNlLCJBVF9YWVpfaXNfRmFsc2VfMV9NYXJfMDJfMTJfMDQiOmZhbHNlLCJBVF9YWVpfaXNfRmFsc2VfMV9NYXJfMDJfMTJfMDUiOmZhbHNlLCJBVF9YWVpfaXNfRmFsc2VfMV9NYXJfMDJfMTJfMjIiOmZhbHNlLCJBVF9YWVpfaXNfVHJ1ZV8xX01hcl8wMl8xMl80MyI6ZmFsc2UsIlRFU1RfQVRfWFlaX2lzX1RydWVfMV9NYXJfMDJfMTNfMTgiOmZhbHNlLCJURVNUMV9BVF9YWVpfaXNfVHJ1ZV8yX01hcl8wMl8xNF81MyI6ZmFsc2UsIkFUX1hZWl9pc19UcnVlXzFfTWFyXzAyXzE1XzI2IjpmYWxzZSwiQVRfWFlaX2lzX0ZhbHNlXzFfTWFyXzAyXzE1XzI5IjpmYWxzZSwiQVRfWFlaX2lzX0ZhbHNlXzFfTWFyXzAyXzE1XzM2IjpmYWxzZSwiQVRfWFlaX2lzX1RydWVfMV9NYXJfMDJfMTZfMjQiOmZhbHNlLCJBVF9YWVpfaXNfVHJ1ZV8xX01hcl8wMl8xNl8yNyI6ZmFsc2UsInRlc3QiOmZhbHNlLCJBVF9YWVpfaXNfVHJ1ZV8xX01hcl8wM18xMF81NF8zcG9pbnRzX0FsbERhdGEiOmZhbHNlLCJBVF9YWVpfaXNfRmFsc2VfMV9NYXJfMDNfMTFfMDBfM3BvaW50c19BbGxEYXRhIjpmYWxzZSwiQVRfWFlaX2lzX0ZhbHNlXzFfTWFyXzAzXzExXzA2M3BvaW50c19Ob0ltYWdlIjpmYWxzZSwiQVRfWFlaX2lzX1RydWVfMV9NYXJfMDNfMTFfMzIzcG9pbnRzX0FsbERhdGEiOmZhbHNlLCJBVF9YWVpfaXNfVHJ1ZV8xX01hcl8wM18xMl8zMDNwb2ludHNfQWxsRGF0YSI6ZmFsc2UsIkFUX1hZWl9pc19GYWxzZV8xX01hcl8wM18xNF8zNTNwb2ludHNfQWxsRGF0YSI6ZmFsc2UsIkFUX1hZWl9pc19UcnVlXzFfTWFyXzAzXzE2XzEzX0FsbHBvaW50c19BbGxEYXRhIjpmYWxzZSwiSU09MV9YWVo9VHJ1ZV9Vc2VJTWFnZT1UcnVlX01hcl8wNF8xNF8yMyI6ZmFsc2UsIklNPTFfWFlaPVRydWVfVXNlSU1hZ2U9VHJ1ZV9NYXJfMDRfMTRfMjYiOmZhbHNlLCJJTT0xX1hZWj1UcnVlX1VzZUlNYWdlPVRydWVfTWFyXzA0XzE0XzI4IjpmYWxzZSwiSU09MV9YWVo9VHJ1ZV9Vc2VJTWFnZT1UcnVlX3J1bj0xTWFyXzA0XzE0XzI5IjpmYWxzZSwiSU09MWJhdGNoU2l6ZT01MF9YWVo9VHJ1ZV9Vc2VJTWFnZT1UcnVlX3J1bj0xTWFyXzA0XzE0XzM2IjpmYWxzZSwiSU09MWJhdGNoU2l6ZT01MF9YWVo9VHJ1ZV9Vc2VJTWFnZT1UcnVlX3J1bj0yTWFyXzA0XzE0XzM4IjpmYWxzZSwiSU09MWJhdGNoU2l6ZT01MF9YWVo9VHJ1ZV9Vc2VJTWFnZT1UcnVlX3J1bj0zTWFyXzA0XzE0XzQwIjpmYWxzZSwiSU09MWJhdGNoU2l6ZT01MF9YWVo9VHJ1ZV9Vc2VJTWFnZT1UcnVlX3J1bj00TWFyXzA0XzE0XzQxIjpmYWxzZSwiSU09MWJhdGNoU2l6ZT01MF9YWVo9VHJ1ZV9Vc2VJTWFnZT1UcnVlX3J1bj01TWFyXzA0XzE0XzQzIjpmYWxzZSwiSU09MWJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlSU1hZ2U9RmFsc2VfcnVuPTFNYXJfMDRfMTRfNDciOmZhbHNlLCJJTT0xYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VJTWFnZT1GYWxzZV9ydW49Mk1hcl8wNF8xNF80OSI6ZmFsc2UsIklNPTFiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZUlNYWdlPUZhbHNlX3J1bj0zTWFyXzA0XzE0XzUyIjpmYWxzZSwiSU09MWJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlSU1hZ2U9RmFsc2VfcnVuPTRNYXJfMDRfMTRfNTQiOmZhbHNlLCJJTT0xYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VJTWFnZT1GYWxzZV9ydW49NU1hcl8wNF8xNF81NSI6ZmFsc2UsIklNPTFiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZUlNYWdlPVRydWVfcnVuPTFNYXJfMDRfMTRfNTgiOmZhbHNlLCJJTT0xYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VJTWFnZT1UcnVlX3J1bj0yTWFyXzA0XzE1XzAyIjpmYWxzZSwiSU09MWJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlSU1hZ2U9VHJ1ZV9ydW49M01hcl8wNF8xNV8wNSI6ZmFsc2UsIklNPTFiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZUlNYWdlPVRydWVfcnVuPTRNYXJfMDRfMTVfMDkiOmZhbHNlLCJJTT0xYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VJTWFnZT1UcnVlX3J1bj01TWFyXzA0XzE1XzExIjpmYWxzZSwiSU09MWJhdGNoU2l6ZT01MF9YWVo9VHJ1ZV9Vc2VJTWFnZT1GYWxzZV9ydW49MU1hcl8wNF8xNV8xMyI6ZmFsc2UsIklNPTFiYXRjaFNpemU9NTBfWFlaPVRydWVfVXNlSU1hZ2U9RmFsc2VfcnVuPTJNYXJfMDRfMTVfMTYiOmZhbHNlLCJJTT0xYmF0Y2hTaXplPTUwX1hZWj1UcnVlX1VzZUlNYWdlPUZhbHNlX3J1bj0zTWFyXzA0XzE1XzE4IjpmYWxzZSwiSU09MWJhdGNoU2l6ZT01MF9YWVo9VHJ1ZV9Vc2VJTWFnZT1GYWxzZV9ydW49NE1hcl8wNF8xNV8yMSI6ZmFsc2UsIklNPTFiYXRjaFNpemU9NTBfWFlaPVRydWVfVXNlSU1hZ2U9RmFsc2VfcnVuPTVNYXJfMDRfMTVfMjMiOmZhbHNlLCJJTT0xYmF0Y2hTaXplPTUwX1hZWj1UcnVlX1VzZUlNYWdlPVRydWVfcnVuPTFNYXJfMDRfMTVfMjgiOmZhbHNlLCJJTT0xYmF0Y2hTaXplPTUwX1hZWj1UcnVlX1VzZUlNYWdlPVRydWVfcnVuPTJNYXJfMDRfMTVfMzAiOmZhbHNlLCJJTT0xYmF0Y2hTaXplPTUwX1hZWj1UcnVlX1VzZUlNYWdlPVRydWVfcnVuPTRNYXJfMDRfMTVfMzUiOmZhbHNlLCJJTT0xYmF0Y2hTaXplPTUwX1hZWj1UcnVlX1VzZUlNYWdlPVRydWVfcnVuPTNNYXJfMDRfMTVfNDYiOmZhbHNlLCJJTT0xYmF0Y2hTaXplPTUwX1hZWj1UcnVlX1VzZUlNYWdlPVRydWVfcnVuPTVNYXJfMDRfMTVfNTEiOmZhbHNlLCJNaW51c18xX0lNPTFiYXRjaFNpemU9NTBfWFlaPVRydWVfVXNlX0lNYWdlPVRydWVfcnVuPTEiOmZhbHNlLCJNaW51c18xX0lNPTFiYXRjaFNpemU9NTBfWFlaPVRydWVfVXNlX0lNYWdlPVRydWVfcnVuPTIiOmZhbHNlLCJNaW51c18xX0lNPTFiYXRjaFNpemU9NTBfWFlaPVRydWVfVXNlX0lNYWdlPVRydWVfcnVuPTMiOmZhbHNlLCJNaW51c18xX0lNPTFiYXRjaFNpemU9NTBfWFlaPVRydWVfVXNlX0lNYWdlPVRydWVfcnVuPTQiOmZhbHNlLCJNaW51c18xX0lNPTFiYXRjaFNpemU9NTBfWFlaPVRydWVfVXNlX0lNYWdlPVRydWVfcnVuPTUiOmZhbHNlLCJNaW51c18xX0lNPTJiYXRjaFNpemU9NTBfWFlaPVRydWVfVXNlX0lNYWdlPVRydWVfcnVuPTEiOmZhbHNlLCJJTT00YmF0Y2hTaXplPTUwX1hZWj1UcnVlX1VzZV9JTWFnZT1UcnVlX3J1bj0xIjpmYWxzZSwiSU09NGJhdGNoU2l6ZT01MF9YWVo9VHJ1ZV9Vc2VfSU1hZ2U9VHJ1ZV9ydW49MiI6ZmFsc2UsIklNPTRiYXRjaFNpemU9NTBfWFlaPVRydWVfVXNlX0lNYWdlPVRydWVfcnVuPTMiOmZhbHNlLCJJTT00YmF0Y2hTaXplPTUwX1hZWj1UcnVlX1VzZV9JTWFnZT1UcnVlX3J1bj00IjpmYWxzZSwiSU09NGJhdGNoU2l6ZT01MF9YWVo9VHJ1ZV9Vc2VfSU1hZ2U9VHJ1ZV9ydW49NSI6ZmFsc2UsIklNPTRiYXRjaFNpemU9NTBfWFlaPVRydWVfVXNlX0lNYWdlPUZhbHNlX3J1bj0xIjpmYWxzZSwiSU09NGJhdGNoU2l6ZT01MF9YWVo9VHJ1ZV9Vc2VfSU1hZ2U9RmFsc2VfcnVuPTIiOmZhbHNlLCJJTT00YmF0Y2hTaXplPTUwX1hZWj1UcnVlX1VzZV9JTWFnZT1GYWxzZV9ydW49MyI6ZmFsc2UsIklNPTRiYXRjaFNpemU9NTBfWFlaPVRydWVfVXNlX0lNYWdlPUZhbHNlX3J1bj00IjpmYWxzZSwiSU09NGJhdGNoU2l6ZT01MF9YWVo9VHJ1ZV9Vc2VfSU1hZ2U9RmFsc2VfcnVuPTUiOmZhbHNlLCJJTT00YmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9ydW49MSI6ZmFsc2UsIklNPTRiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX3J1bj0yIjpmYWxzZSwiSU09NGJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfcnVuPTMiOmZhbHNlLCJJTT00YmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9ydW49NCI6ZmFsc2UsIklNPTRiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX3J1bj01IjpmYWxzZSwiSU09NGJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPUZhbHNlX3J1bj0xIjpmYWxzZSwiSU09NGJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPUZhbHNlX3J1bj0yIjpmYWxzZSwiSU09NGJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPUZhbHNlX3J1bj0zIjpmYWxzZSwiSU09NGJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPUZhbHNlX3J1bj00IjpmYWxzZSwiSU09NGJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPUZhbHNlX3J1bj01IjpmYWxzZSwiSU09NWJhdGNoU2l6ZT01MF9YWVo9VHJ1ZV9Vc2VfSU1hZ2U9VHJ1ZV9ydW49MSI6ZmFsc2UsIklNPTViYXRjaFNpemU9NTBfWFlaPVRydWVfVXNlX0lNYWdlPVRydWVfcnVuPTIiOmZhbHNlLCJJTT01YmF0Y2hTaXplPTUwX1hZWj1UcnVlX1VzZV9JTWFnZT1UcnVlX3J1bj0zIjpmYWxzZSwiSU09NWJhdGNoU2l6ZT01MF9YWVo9VHJ1ZV9Vc2VfSU1hZ2U9VHJ1ZV9ydW49NCI6ZmFsc2UsIklNPTViYXRjaFNpemU9NTBfWFlaPVRydWVfVXNlX0lNYWdlPVRydWVfcnVuPTUiOmZhbHNlLCJJTT01YmF0Y2hTaXplPTUwX1hZWj1UcnVlX1VzZV9JTWFnZT1GYWxzZV9ydW49MSI6ZmFsc2UsIklNPTViYXRjaFNpemU9NTBfWFlaPVRydWVfVXNlX0lNYWdlPUZhbHNlX3J1bj0yIjpmYWxzZSwiSU09NWJhdGNoU2l6ZT01MF9YWVo9VHJ1ZV9Vc2VfSU1hZ2U9RmFsc2VfcnVuPTMiOmZhbHNlLCJJTT01YmF0Y2hTaXplPTUwX1hZWj1UcnVlX1VzZV9JTWFnZT1GYWxzZV9ydW49NCI6ZmFsc2UsIklNPTViYXRjaFNpemU9NTBfWFlaPVRydWVfVXNlX0lNYWdlPUZhbHNlX3J1bj01IjpmYWxzZSwiSU09NWJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfcnVuPTEiOmZhbHNlLCJJTT01YmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9ydW49MiI6ZmFsc2UsIklNPTViYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX3J1bj0zIjpmYWxzZSwiSU09NWJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfcnVuPTQiOmZhbHNlLCJJTT01YmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9ydW49NSI6ZmFsc2UsIklNPTViYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1GYWxzZV9ydW49MSI6ZmFsc2UsIklNPTViYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1GYWxzZV9ydW49MiI6ZmFsc2UsIklNPTViYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1GYWxzZV9ydW49MyI6ZmFsc2UsIklNPTViYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1GYWxzZV9ydW49NCI6ZmFsc2UsIklNPTViYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1GYWxzZV9ydW49NSI6ZmFsc2UsIklNPTNiYXRjaFNpemU9NTBfWFlaPVRydWVfVXNlX0lNYWdlPVRydWVfcnVuPTEiOmZhbHNlLCJJTT0zYmF0Y2hTaXplPTUwX1hZWj1UcnVlX1VzZV9JTWFnZT1UcnVlX3J1bj0yIjpmYWxzZSwiSU09M2JhdGNoU2l6ZT01MF9YWVo9VHJ1ZV9Vc2VfSU1hZ2U9VHJ1ZV9ydW49MyI6ZmFsc2UsIklNPTNiYXRjaFNpemU9NTBfWFlaPVRydWVfVXNlX0lNYWdlPVRydWVfcnVuPTQiOmZhbHNlLCJJTT0zYmF0Y2hTaXplPTUwX1hZWj1UcnVlX1VzZV9JTWFnZT1UcnVlX3J1bj01IjpmYWxzZSwiSU09M2JhdGNoU2l6ZT01MF9YWVo9VHJ1ZV9Vc2VfSU1hZ2U9RmFsc2VfcnVuPTEiOmZhbHNlLCJJTT0zYmF0Y2hTaXplPTUwX1hZWj1UcnVlX1VzZV9JTWFnZT1GYWxzZV9ydW49MiI6ZmFsc2UsIklNPTNiYXRjaFNpemU9NTBfWFlaPVRydWVfVXNlX0lNYWdlPUZhbHNlX3J1bj0zIjpmYWxzZSwiSU09M2JhdGNoU2l6ZT01MF9YWVo9VHJ1ZV9Vc2VfSU1hZ2U9RmFsc2VfcnVuPTQiOmZhbHNlLCJJTT0zYmF0Y2hTaXplPTUwX1hZWj1UcnVlX1VzZV9JTWFnZT1GYWxzZV9ydW49NSI6ZmFsc2UsIklNPTNiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX3J1bj0xIjpmYWxzZSwiSU09M2JhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfcnVuPTIiOmZhbHNlLCJJTT0zYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9ydW49MyI6ZmFsc2UsIklNPTNiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX3J1bj00IjpmYWxzZSwiSU09M2JhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfcnVuPTUiOmZhbHNlLCJJTT0zYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9RmFsc2VfcnVuPTEiOmZhbHNlLCJJTT0zYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9RmFsc2VfcnVuPTIiOmZhbHNlLCJJTT0zYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9RmFsc2VfcnVuPTMiOmZhbHNlLCJJTT0zYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9RmFsc2VfcnVuPTQiOmZhbHNlLCJJTT0zYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9RmFsc2VfcnVuPTUiOmZhbHNlLCJJTT02YmF0Y2hTaXplPTUwX1hZWj1UcnVlX1VzZV9JTWFnZT1UcnVlX3J1bj0xIjpmYWxzZSwiSU09NmJhdGNoU2l6ZT01MF9YWVo9VHJ1ZV9Vc2VfSU1hZ2U9VHJ1ZV9ydW49MiI6ZmFsc2UsIklNPTZiYXRjaFNpemU9NTBfWFlaPVRydWVfVXNlX0lNYWdlPVRydWVfcnVuPTMiOmZhbHNlLCJJTT02YmF0Y2hTaXplPTUwX1hZWj1UcnVlX1VzZV9JTWFnZT1UcnVlX3J1bj00IjpmYWxzZSwiSU09NmJhdGNoU2l6ZT01MF9YWVo9VHJ1ZV9Vc2VfSU1hZ2U9VHJ1ZV9ydW49NSI6ZmFsc2UsIklNPTZiYXRjaFNpemU9NTBfWFlaPVRydWVfVXNlX0lNYWdlPUZhbHNlX3J1bj0xIjpmYWxzZSwiSU09NmJhdGNoU2l6ZT01MF9YWVo9VHJ1ZV9Vc2VfSU1hZ2U9RmFsc2VfcnVuPTIiOmZhbHNlLCJJTT02YmF0Y2hTaXplPTUwX1hZWj1UcnVlX1VzZV9JTWFnZT1GYWxzZV9ydW49MyI6ZmFsc2UsIklNPTZiYXRjaFNpemU9NTBfWFlaPVRydWVfVXNlX0lNYWdlPUZhbHNlX3J1bj00IjpmYWxzZSwiSU09NmJhdGNoU2l6ZT01MF9YWVo9VHJ1ZV9Vc2VfSU1hZ2U9RmFsc2VfcnVuPTUiOmZhbHNlLCJJTT02YmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9ydW49MSI6ZmFsc2UsIklNPTZiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX3J1bj0yIjpmYWxzZSwiSU09NmJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfcnVuPTMiOmZhbHNlLCJJTT02YmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9ydW49NCI6ZmFsc2UsIklNPTZiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX3J1bj01IjpmYWxzZSwiSU09NmJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPUZhbHNlX3J1bj0xIjpmYWxzZSwiSU09NmJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPUZhbHNlX3J1bj0yIjpmYWxzZSwiSU09NmJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPUZhbHNlX3J1bj0zIjpmYWxzZSwiSU09NmJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPUZhbHNlX3J1bj00IjpmYWxzZSwiSU09NmJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPUZhbHNlX3J1bj01IjpmYWxzZSwiSU09MmJhdGNoU2l6ZT01MF9YWVo9VHJ1ZV9Vc2VfSU1hZ2U9VHJ1ZV9ydW49MSI6ZmFsc2UsIklNPTJiYXRjaFNpemU9NTBfWFlaPVRydWVfVXNlX0lNYWdlPVRydWVfcnVuPTIiOmZhbHNlLCJJTT0yYmF0Y2hTaXplPTUwX1hZWj1UcnVlX1VzZV9JTWFnZT1UcnVlX3J1bj0zIjpmYWxzZSwiSU09MmJhdGNoU2l6ZT01MF9YWVo9VHJ1ZV9Vc2VfSU1hZ2U9VHJ1ZV9ydW49NCI6ZmFsc2UsIklNPTJiYXRjaFNpemU9NTBfWFlaPVRydWVfVXNlX0lNYWdlPVRydWVfcnVuPTUiOmZhbHNlLCJJTT0yYmF0Y2hTaXplPTUwX1hZWj1UcnVlX1VzZV9JTWFnZT1GYWxzZV9ydW49MSI6ZmFsc2UsIklNPTJiYXRjaFNpemU9NTBfWFlaPVRydWVfVXNlX0lNYWdlPUZhbHNlX3J1bj0yIjpmYWxzZSwiSU09MmJhdGNoU2l6ZT01MF9YWVo9VHJ1ZV9Vc2VfSU1hZ2U9RmFsc2VfcnVuPTMiOmZhbHNlLCJJTT0yYmF0Y2hTaXplPTUwX1hZWj1UcnVlX1VzZV9JTWFnZT1GYWxzZV9ydW49NCI6ZmFsc2UsIklNPTJiYXRjaFNpemU9NTBfWFlaPVRydWVfVXNlX0lNYWdlPUZhbHNlX3J1bj01IjpmYWxzZSwiSU09MmJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfcnVuPTEiOmZhbHNlLCJJTT0yYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9ydW49MiI6ZmFsc2UsIklNPTJiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX3J1bj0zIjpmYWxzZSwiSU09MmJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfcnVuPTQiOmZhbHNlLCJJTT0yYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9ydW49NSI6ZmFsc2UsIklNPTJiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1GYWxzZV9ydW49MSI6ZmFsc2UsIklNPTJiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1GYWxzZV9ydW49MiI6ZmFsc2UsIklNPTJiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1GYWxzZV9ydW49MyI6ZmFsc2UsIklNPTJiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1GYWxzZV9ydW49NCI6ZmFsc2UsIklNPTJiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1GYWxzZV9ydW49NSI6ZmFsc2UsIklNPTFiYXRjaFNpemU9NTBfWFlaPVRydWVfVXNlX0lNYWdlPVRydWVfcnVuPTEiOmZhbHNlLCJJTT0xYmF0Y2hTaXplPTUwX1hZWj1UcnVlX1VzZV9JTWFnZT1UcnVlX3J1bj0yIjpmYWxzZSwiSU09MWJhdGNoU2l6ZT01MF9YWVo9VHJ1ZV9Vc2VfSU1hZ2U9VHJ1ZV9ydW49MyI6ZmFsc2UsIklNPTFiYXRjaFNpemU9NTBfWFlaPVRydWVfVXNlX0lNYWdlPVRydWVfcnVuPTQiOmZhbHNlLCJJTT0xYmF0Y2hTaXplPTUwX1hZWj1UcnVlX1VzZV9JTWFnZT1UcnVlX3J1bj01IjpmYWxzZSwiSU09MWJhdGNoU2l6ZT01MF9YWVo9VHJ1ZV9Vc2VfSU1hZ2U9RmFsc2VfcnVuPTEiOmZhbHNlLCJJTT0xYmF0Y2hTaXplPTUwX1hZWj1UcnVlX1VzZV9JTWFnZT1GYWxzZV9ydW49MiI6ZmFsc2UsIklNPTFiYXRjaFNpemU9NTBfWFlaPVRydWVfVXNlX0lNYWdlPUZhbHNlX3J1bj0zIjpmYWxzZSwiSU09MWJhdGNoU2l6ZT01MF9YWVo9VHJ1ZV9Vc2VfSU1hZ2U9RmFsc2VfcnVuPTQiOmZhbHNlLCJJTT0xYmF0Y2hTaXplPTUwX1hZWj1UcnVlX1VzZV9JTWFnZT1GYWxzZV9ydW49NSI6ZmFsc2UsIklNPTFiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX3J1bj0xIjpmYWxzZSwiSU09MWJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfcnVuPTIiOmZhbHNlLCJJTT0xYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9ydW49MyI6ZmFsc2UsIklNPTFiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX3J1bj00IjpmYWxzZSwiSU09MWJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfcnVuPTUiOmZhbHNlLCJJTT0xYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9RmFsc2VfcnVuPTEiOmZhbHNlLCJJTT0xYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9RmFsc2VfcnVuPTIiOmZhbHNlLCJJTT0xYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9RmFsc2VfcnVuPTMiOmZhbHNlLCJJTT0xYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9RmFsc2VfcnVuPTQiOmZhbHNlLCJJTT0xYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9RmFsc2VfcnVuPTUiOmZhbHNlLCJJTT0xYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX3J1bj0xIjpmYWxzZSwiSU09MWJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9ydW49MiI6ZmFsc2UsIklNPTFiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX1VzZV9FbmRwb2ludEZlYXR1cmVzPVRydWVfcnVuPTMiOmZhbHNlLCJJTT0xYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX3J1bj00IjpmYWxzZSwiSU09MWJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9ydW49NSI6ZmFsc2UsIklNPTJiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX1VzZV9FbmRwb2ludEZlYXR1cmVzPVRydWVfcnVuPTEiOmZhbHNlLCJJTT0yYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX3J1bj0yIjpmYWxzZSwiSU09MmJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9ydW49MyI6ZmFsc2UsIklNPTJiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX1VzZV9FbmRwb2ludEZlYXR1cmVzPVRydWVfcnVuPTQiOmZhbHNlLCJJTT0yYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX3J1bj01IjpmYWxzZSwiSU09M2JhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9ydW49MSI6ZmFsc2UsIklNPTRiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX1VzZV9FbmRwb2ludEZlYXR1cmVzPVRydWVfcnVuPTEiOmZhbHNlLCJJTT01YmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX3J1bj0xIjpmYWxzZSwiSU09NmJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9ydW49MSI6ZmFsc2UsIklNPTNiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX1VzZV9FbmRwb2ludEZlYXR1cmVzPVRydWVfcnVuPTIiOmZhbHNlLCJJTT00YmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX3J1bj0yIjpmYWxzZSwiSU09NWJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9ydW49MiI6ZmFsc2UsIklNPTZiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX1VzZV9FbmRwb2ludEZlYXR1cmVzPVRydWVfcnVuPTIiOmZhbHNlLCJJTT0zYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX3J1bj0zIjpmYWxzZSwiSU09NGJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9ydW49MyI6ZmFsc2UsIklNPTViYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX1VzZV9FbmRwb2ludEZlYXR1cmVzPVRydWVfcnVuPTMiOmZhbHNlLCJJTT02YmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX3J1bj0zIjpmYWxzZSwiSU09M2JhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9ydW49NCI6ZmFsc2UsIklNPTRiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX1VzZV9FbmRwb2ludEZlYXR1cmVzPVRydWVfcnVuPTQiOmZhbHNlLCJJTT01YmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX3J1bj00IjpmYWxzZSwiSU09NmJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9ydW49NCI6ZmFsc2UsIklNPTNiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX1VzZV9FbmRwb2ludEZlYXR1cmVzPVRydWVfcnVuPTUiOmZhbHNlLCJJTT00YmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX3J1bj01IjpmYWxzZSwiSU09NWJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9ydW49NSI6ZmFsc2UsIklNPTZiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX1VzZV9FbmRwb2ludEZlYXR1cmVzPVRydWVfcnVuPTUiOmZhbHNlLCJJTT0xYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTI1MF9ydW49MSI6ZmFsc2UsIklNPTJiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX1VzZV9FbmRwb2ludEZlYXR1cmVzPVRydWVfRXBvY2g9MjUwX3J1bj0xIjpmYWxzZSwiSU09M2JhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9FcG9jaD0yNTBfcnVuPTEiOmZhbHNlLCJJTT00YmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTI1MF9ydW49MSI6ZmFsc2UsIklNPTViYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX1VzZV9FbmRwb2ludEZlYXR1cmVzPVRydWVfRXBvY2g9MjUwX3J1bj0xIjpmYWxzZSwiSU09NmJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9FcG9jaD0yNTBfcnVuPTEiOmZhbHNlLCJJTT0xYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTI1MF9ydW49MiI6ZmFsc2UsIklNPTJiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX1VzZV9FbmRwb2ludEZlYXR1cmVzPVRydWVfRXBvY2g9MjUwX3J1bj0yIjpmYWxzZSwiSU09M2JhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9FcG9jaD0yNTBfcnVuPTIiOmZhbHNlLCJJTT00YmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTI1MF9ydW49MiI6ZmFsc2UsIklNPTViYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX1VzZV9FbmRwb2ludEZlYXR1cmVzPVRydWVfRXBvY2g9MjUwX3J1bj0yIjpmYWxzZSwiSU09NmJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9FcG9jaD0yNTBfcnVuPTIiOmZhbHNlLCJJTT0xYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTI1MF9ydW49MyI6ZmFsc2UsIklNPTJiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX1VzZV9FbmRwb2ludEZlYXR1cmVzPVRydWVfRXBvY2g9MjUwX3J1bj0zIjpmYWxzZSwiSU09M2JhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9FcG9jaD0yNTBfcnVuPTMiOmZhbHNlLCJJTT00YmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTI1MF9ydW49MyI6ZmFsc2UsIklNPTViYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX1VzZV9FbmRwb2ludEZlYXR1cmVzPVRydWVfRXBvY2g9MjUwX3J1bj0zIjpmYWxzZSwiSU09NmJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9FcG9jaD0yNTBfcnVuPTMiOmZhbHNlLCJJTT0xYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTI1MF9ydW49NCI6ZmFsc2UsIklNPTJiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX1VzZV9FbmRwb2ludEZlYXR1cmVzPVRydWVfRXBvY2g9MjUwX3J1bj00IjpmYWxzZSwiSU09M2JhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9FcG9jaD0yNTBfcnVuPTQiOmZhbHNlLCJJTT00YmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTI1MF9ydW49NCI6ZmFsc2UsIklNPTViYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX1VzZV9FbmRwb2ludEZlYXR1cmVzPVRydWVfRXBvY2g9MjUwX3J1bj00IjpmYWxzZSwiSU09NmJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9FcG9jaD0yNTBfcnVuPTQiOmZhbHNlLCJJTT0xYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTI1MF9ydW49NSI6ZmFsc2UsIklNPTJiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX1VzZV9FbmRwb2ludEZlYXR1cmVzPVRydWVfRXBvY2g9MjUwX3J1bj01IjpmYWxzZSwiSU09M2JhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9FcG9jaD0yNTBfcnVuPTUiOmZhbHNlLCJJTT00YmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTI1MF9ydW49NSI6ZmFsc2UsIklNPTZiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX1VzZV9FbmRwb2ludEZlYXR1cmVzPVRydWVfRXBvY2g9MjUwX3J1bj01IjpmYWxzZSwiSU09NWJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9FcG9jaD0yNTBfcnVuPTUiOmZhbHNlLCJJTT00YmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTI1MF9ydW49NiI6ZmFsc2UsIlRtcF8zcG9pbnRNZXJnZXJfSU09MWJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9FcG9jaD0xNTBfcnVuPTUiOmZhbHNlLCJUbXBfM3BvaW50TWVyZ2VyX0lNPTFiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX1VzZV9FbmRwb2ludEZlYXR1cmVzPVRydWVfRXBvY2g9MTUwX3J1bj0xIjpmYWxzZSwiVG1wXzNwb2ludE1lcmdlcl9JTT0xYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTE1MF9ydW49MiI6ZmFsc2UsIlRtcF8zcG9pbnRNZXJnZXJfSU09MWJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9FcG9jaD0xNTBfcnVuPTMiOmZhbHNlLCJUbXBfM3BvaW50TWVyZ2VyX0lNPTFiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX1VzZV9FbmRwb2ludEZlYXR1cmVzPVRydWVfRXBvY2g9MTUwX3J1bj00IjpmYWxzZSwiVG1wXzNwb2ludE1lcmdlcl9JTT0yYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTE1MF9ydW49MSI6ZmFsc2UsIlRtcF8zcG9pbnRNZXJnZXJfSU09NmJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9FcG9jaD0xNTBfcnVuPTQiOmZhbHNlLCJUbXBfM3BvaW50TWVyZ2VyX0lNPTViYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX1VzZV9FbmRwb2ludEZlYXR1cmVzPVRydWVfRXBvY2g9MTUwX3J1bj00IjpmYWxzZSwiVG1wXzNwb2ludE1lcmdlcl9JTT00YmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTE1MF9ydW49NCI6ZmFsc2UsIlRtcF8zcG9pbnRNZXJnZXJfSU09M2JhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9FcG9jaD0xNTBfcnVuPTQiOmZhbHNlLCJUbXBfM3BvaW50TWVyZ2VyX0lNPTJiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX1VzZV9FbmRwb2ludEZlYXR1cmVzPVRydWVfRXBvY2g9MTUwX3J1bj00IjpmYWxzZSwiVG1wXzNwb2ludE1lcmdlcl9JTT02YmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTE1MF9ydW49MyI6ZmFsc2UsIkluaXRpYWxpemVkX1RtcF8zcG9pbnRNZXJnZXJfSU09MmJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9FcG9jaD0yNTBfcnVuPTEiOmZhbHNlLCJJbml0aWFsaXplZF9OZXdfVG1wXzNwb2ludE1lcmdlcl9JTT0xYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTI1MF9ydW49MSI6ZmFsc2UsIkluaXRpYWxpemVkX05ld19UbXBfM3BvaW50TWVyZ2VyX0lNPTJiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX1VzZV9FbmRwb2ludEZlYXR1cmVzPVRydWVfRXBvY2g9MjUwX3J1bj0xIjpmYWxzZSwiSW5pdGlhbGl6ZWRfTmV3X1RtcF8zcG9pbnRNZXJnZXJfSU09M2JhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9FcG9jaD0yNTBfcnVuPTEiOmZhbHNlLCJJbml0aWFsaXplZF9OZXdfVG1wXzNwb2ludE1lcmdlcl9JTT02YmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTI1MF9ydW49NSI6ZmFsc2UsIkluaXRpYWxpemVkX05ldzFfVG1wXzNwb2ludE1lcmdlcl9JTT0xYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTI1MF9ydW49MSI6ZmFsc2UsIkluaXRpYWxpemVkX05ldzFfVG1wXzNwb2ludE1lcmdlcl9JTT0xYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTI1MF9ydW49MiI6ZmFsc2UsIkluaXRpYWxpemVkX05ldzFfVG1wXzNwb2ludE1lcmdlcl9JTT0xYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTI1MF9ydW49MyI6ZmFsc2UsIkluaXRpYWxpemVkX05ldzFfVG1wXzNwb2ludE1lcmdlcl9JTT0xYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTI1MF9ydW49NCI6ZmFsc2UsIkluaXRpYWxpemVkX05ldzFfVG1wXzNwb2ludE1lcmdlcl9JTT0xYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTI1MF9ydW49NSI6ZmFsc2UsIkluaXRpYWxpemVkX05ldzFfVG1wXzNwb2ludE1lcmdlcl9JTT0xYmF0Y2hTaXplPTIwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTI1MF9ydW49MSI6ZmFsc2UsIkluaXRpYWxpemVkX05ldzFfVG1wXzNwb2ludE1lcmdlcl9JTT0xYmF0Y2hTaXplPTIwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTI1MF9ydW49MiI6ZmFsc2UsIkluaXRpYWxpemVkX05ldzFfVG1wXzNwb2ludE1lcmdlcl9JTT0xYmF0Y2hTaXplPTIwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTI1MF9ydW49NSI6ZmFsc2UsIkluaXRpYWxpemVkX05ldzFfVG1wXzNwb2ludE1lcmdlcl9JTT0xYmF0Y2hTaXplPTIwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTI1MF9ydW49NCI6ZmFsc2UsIkluaXRpYWxpemVkX05ldzFfVG1wXzNwb2ludE1lcmdlcl9JTT0xYmF0Y2hTaXplPTIwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTI1MF9ydW49MyI6ZmFsc2UsIkluaXRpYWxpemVkX2hlX25vcm1hbF9UbXBfM3BvaW50TWVyZ2VyX0lNPTFiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX1VzZV9FbmRwb2ludEZlYXR1cmVzPVRydWVfRXBvY2g9MjUwX3J1bj01IjpmYWxzZSwiSW5pdGlhbGl6ZWRfaGVfbm9ybWFsX1RtcF8zcG9pbnRNZXJnZXJfSU09MWJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9FcG9jaD0yNTBfcnVuPTQiOmZhbHNlLCJJbml0aWFsaXplZF9oZV9ub3JtYWxfVG1wXzNwb2ludE1lcmdlcl9JTT0xYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTI1MF9ydW49MyI6ZmFsc2UsIkluaXRpYWxpemVkX2hlX25vcm1hbF9UbXBfM3BvaW50TWVyZ2VyX0lNPTFiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX1VzZV9FbmRwb2ludEZlYXR1cmVzPVRydWVfRXBvY2g9MjUwX3J1bj0yIjpmYWxzZSwiSW5pdGlhbGl6ZWRfaGVfbm9ybWFsX1RtcF8zcG9pbnRNZXJnZXJfSU09MWJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9FcG9jaD0yNTBfcnVuPTEiOmZhbHNlLCJJbml0aWFsaXplZF9yYW5kb21fdW5pZm9ybV9UbXBfM3BvaW50TWVyZ2VyX0lNPTFiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX1VzZV9FbmRwb2ludEZlYXR1cmVzPVRydWVfRXBvY2g9MjUwX3J1bj0xIjpmYWxzZSwiSW5pdGlhbGl6ZWRfcmFuZG9tX3VuaWZvcm1fVG1wXzNwb2ludE1lcmdlcl9JTT0xYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTI1MF9ydW49MiI6ZmFsc2UsIkluaXRpYWxpemVkX3JhbmRvbV91bmlmb3JtX1RtcF8zcG9pbnRNZXJnZXJfSU09MWJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9FcG9jaD0yNTBfcnVuPTMiOmZhbHNlLCJJbml0aWFsaXplZF9yYW5kb21fdW5pZm9ybV9UbXBfM3BvaW50TWVyZ2VyX0lNPTFiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX1VzZV9FbmRwb2ludEZlYXR1cmVzPVRydWVfRXBvY2g9MjUwX3J1bj00IjpmYWxzZSwiSW5pdGlhbGl6ZWRfcmFuZG9tX3VuaWZvcm1fVG1wXzNwb2ludE1lcmdlcl9JTT0xYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTI1MF9ydW49NSI6ZmFsc2UsIkluaXRpYWxpemVkX2hlX3VuaWZvcm1fVG1wXzNwb2ludE1lcmdlcl9JTT0xYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTI1MF9ydW49MiI6ZmFsc2UsIkluaXRpYWxpemVkX2hlX3VuaWZvcm1fVG1wXzNwb2ludE1lcmdlcl9JTT0xYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTI1MF9ydW49MSI6ZmFsc2UsIkluaXRpYWxpemVkX2xlY3VuX25vcm1hbF9UbXBfM3BvaW50TWVyZ2VyX0lNPTFiYXRjaFNpemU9NTBfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX1VzZV9FbmRwb2ludEZlYXR1cmVzPVRydWVfRXBvY2g9MjUwX3J1bj01IjpmYWxzZSwiSW5pdGlhbGl6ZWRfbGVjdW5fbm9ybWFsX1RtcF8zcG9pbnRNZXJnZXJfSU09MWJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9FcG9jaD0yNTBfcnVuPTQiOmZhbHNlLCJJbml0aWFsaXplZF9sZWN1bl9ub3JtYWxfVG1wXzNwb2ludE1lcmdlcl9JTT0xYmF0Y2hTaXplPTUwX1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTI1MF9ydW49MyI6ZmFsc2UsImZpeGVkX2xvY2Fsc19Jbml0aWFsaXplZF9oZV91bmlmb3JtX1RtcF8zcG9pbnRNZXJnZXJfSU09MWJhdGNoU2l6ZT0yNV9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9FcG9jaD0xMDAwX3J1bj0xIjpmYWxzZSwiZml4ZWRfbG9jYWxzX0luaXRpYWxpemVkX2hlX3VuaWZvcm1fVG1wXzNwb2ludE1lcmdlcl9JTT0xYmF0Y2hTaXplPTI1X1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTEwMDBfcnVuPTIiOmZhbHNlLCJmaXhlZF9sb2NhbHNfSW5pdGlhbGl6ZWRfaGVfdW5pZm9ybV9UbXBfM3BvaW50TWVyZ2VyX0lNPTFiYXRjaFNpemU9MjVfWFlaPUZhbHNlX1VzZV9JTWFnZT1UcnVlX1VzZV9FbmRwb2ludEZlYXR1cmVzPVRydWVfRXBvY2g9MTAwMF9ydW49MyI6ZmFsc2UsImZpeGVkX2xvY2Fsc19Jbml0aWFsaXplZF9oZV91bmlmb3JtX1RtcF8zcG9pbnRNZXJnZXJfSU09MWJhdGNoU2l6ZT0yNV9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9FcG9jaD0xMDAwX3J1bj00IjpmYWxzZSwiZml4ZWRfbG9jYWxzX0luaXRpYWxpemVkX2hlX3VuaWZvcm1fVG1wXzNwb2ludE1lcmdlcl9JTT0xYmF0Y2hTaXplPTI1X1hZWj1GYWxzZV9Vc2VfSU1hZ2U9VHJ1ZV9Vc2VfRW5kcG9pbnRGZWF0dXJlcz1UcnVlX0Vwb2NoPTEwMDBfcnVuPTUiOmZhbHNlLCJBbGxEYXRhX2ZpeGVkX2xvY2Fsc19Jbml0aWFsaXplZF9oZV91bmlmb3JtX1RtcF8zcG9pbnRNZXJnZXJfSU09MWJhdGNoU2l6ZT01MF9YWVo9RmFsc2VfVXNlX0lNYWdlPVRydWVfVXNlX0VuZHBvaW50RmVhdHVyZXM9VHJ1ZV9FcG9jaD0xNTBfcnVuPTEiOmZhbHNlfQ%3D%3D&_smoothingWeight=0.586

            # checkpoint # This is Called Early Stopping (stop at best validation)
            from keras.callbacks import ModelCheckpoint
            filepath= root_dir + '/data/models/'+PltNAme+"_weights.max_val_acc.hdf5"
            checkpoint_max_val_acc = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

            filepath = root_dir + '/data/models/'+ PltNAme + "_weights.min_val_acc.hdf5"
            checkpoint_min_val_acc = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                                     mode='min')

            filepath = root_dir + '/data/models/'+ PltNAme + "_weights.max_val_loss.hdf5"
            checkpoint_max_val_loss = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')

            filepath = root_dir + '/data/models/'+ PltNAme + "_weights.min_val_loss.hdf5"
            checkpoint_min_val_loss = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                                      mode='min')

            # filepath = 'DataFiles/' + PltNAme + "_weights.best_AUC.hdf5"
            # checkpoint_AUC = ModelCheckpoint(filepath, monitor=keras.metrics.AUC(), verbose=1, save_best_only=True,
            #                                       mode='max')

            # fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0,
            # validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,
            # steps_per_epoch=None, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1,
            # use_multiprocessing=False)
            if UseConv:
                X_IMs = IMsTrain3D.reshape(IMsTrain3D.shape[0], IMsTrain3D.shape[1],
                                           IMsTrain3D.shape[2], IMsTrain3D.shape[3],1)
            else:
                X_IMs = IMTrain

            # Track Training History
            history = model.fit([X_IMs,FeatureTrain],
                                LabelsTrain,
                                epochs=epoch,
                                batch_size=batch_size,
                                validation_split=0.30,
                                verbose=1
                                ,callbacks=[checkpoint_max_val_acc,checkpoint_min_val_acc,checkpoint_max_val_loss,checkpoint_min_val_loss,tensorboard,lrate],
                                )

            # Save Model
            model.save(root_dir + '/data/models/'+ PltNAme+'.h5')

# PltNAme = 're2_com_LR=0.001_lim100sc_AlDat_augm_loc_Init_he_uniform_Tmp_
# IM=1batchSize=10_XYZ=False_Use_IMage=True_Use_EndpointFeatures=True_Epoch=150_run=5'
# model = load_model('DataFiles/'+PltNAme+'.h5')

            # Model Summary
            print(model.summary())
            from keras.utils.vis_utils import plot_model
            pltName = root_dir + '/data/models/'+PltNAme+'.png'
            print(pltName)
            plot_model(model, to_file=pltName, show_shapes=True, show_layer_names=True)

            # # Plot training & validation accuracy values
            plt.figure()
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            fig=plt.show()
            pltLoss = root_dir + '/data/models/'+PltNAme+'_Acc.png'
            plt.savefig(pltLoss)
            print(pltLoss)

            # Plot training & validation loss values
            plt.figure()
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            fig=plt.show()
            pltLoss = root_dir + '/data/models/'+PltNAme+'_loss.png'
            plt.savefig(pltLoss)
            print(pltLoss)
