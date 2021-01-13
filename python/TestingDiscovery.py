#
# Created on 9/17/2019
#
# @author Seyed
#
# Email: mousavikahaki@gmail.com
#

# https://www.datacamp.com/community/tutorials/deep-learning-python


# for Merging Model: https://datascience.stackexchange.com/questions/26103/merging-two-different-models-in-keras

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
from keras.constraints import maxnorm
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, BatchNormalization, Concatenate
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score,balanced_accuracy_score, accuracy_score
from keras.callbacks import TensorBoard
from keras.optimizers import SGD
import datetime
import scipy.io as sio
import AT_Classes as Classes
from imblearn.over_sampling import SMOTE
from collections import Counter



Result = np.zeros([2, 16, 5])
# lst_XYZ = ['True','False']
# lst_useImage = ['True','False']

lst_XYZ = ['False']
lst_useImage = ['True']
useEndpointFeatures = 'True'
kernel_initializer='he_uniform'
rotation_degrees = [0,90,180,270]
flips = ['right']
UseConv = True

root_dir = 'E:/AutomatedTracing/TraceProofreading/TraceProofreading'


# modelFolder = '/models_best_10232020'

# modelnum = 0
# modelFolder = '/models_best_3Run_10232020'

# modelnum = 1
# modelFolder = 'models_best_3Run_12092020'

modelnum = 2
modelFolder = 'models_best_3Run_01112021'


for UseIMage in lst_useImage:
    for ImagetoTest in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]:
        for run in [0,1,2]:


            epoch = 100
            batch_size = 50  # 50
            verbose = 1  # verbose=1 will show you an animated progress bar
            doSMOTE = False  # do replicate data using SMOTE method
            learning_Rate = 0.0008  # default  =0.01

            x = datetime.datetime.today()
            nowTimeDate = x.strftime("%b_%d_%H_%M")
            # PltNAme = 'AT_XYZ_is_'+str(useXYZ_Positions)+'_'+str(ImagetoTest)+'_run=2'+nowTimeDate

            # PltNAme = 'NEW_1_INV_FEATURES_CONV=' + str(UseConv) + '_LR=' + str(learning_Rate) + '_100_sce_' + str(
            #             #     kernel_initializer) + '_IM=' + str(ImagetoTest) + 'bchSiz=' + str(batch_size) + '_Use_IM=' + str(
            #             #     UseIMage) + '_Epoch=' + str(
            #             #     epoch) + '_run=' + str(run + 1)
            #             # print(PltNAme)





            # filepath = 'E:\AutomatedTraceResults\DataForConnectingTraining\Data_For_AE_BranchScenarios\IMonce_limit100scen_NEW_Inv_FEATURES.mat'
            # filepath = 'E:\AutomatedTraceResults\DataForConnectingTraining\Data_For_AE_BranchScenarios\IMonce_100_scen_NEW_Inv_FEATURES_User=RG.mat'

            filepath = 'E:\AutomatedTraceResults\DataForConnectingTraining\Data_For_AE_BranchScenarios\S1and2_IMonce_100_scen_NEW_Inv_FEATURES_User=SK.mat'
            ScenariosData = sio.loadmat(filepath)

            IMnums = ScenariosData['IMnum']
            # IMnum = IMnums[0,IMnums.shape[1]-1]
            # IMnums.shape

            Features = ScenariosData['NewFeatures']
            # Feature = Features[0,Features.shape[1]-1]
            # Feature.shape

            IMs = ScenariosData['IMs']
            # IMtmp = IMs[0,IMs.shape[1]-1]
            # IMtmp.shape

            Scenarios = ScenariosData['Scenarios']
            # Scenarios.shape
            # Scenarios[0,Scenarios.shape[1]-1]

            Labels = ScenariosData['Labels']
            # Labels[0,Labels.shape[1]-1]

            maxNumPoints = 12

            IMsTrain = []
            FeatureTrain = []
            LabelsTrain = []
            ScenariosTrain = []

            IMsTest = []
            FeatureTest = []
            LabelsTest = []
            ScenariosTest = []

            UseUpper = False
            numScenarios = Scenarios.shape
            counter = 0

            for i in range(numScenarios[1]):
                scenario = Scenarios[0, i]

                IM = IMs[0, i]
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

                S = Classes.cl_scenario(maxNumPoints, scenario.shape[0], scenario, 0)
                if UseUpper:
                    scenario_arr = S.getUpperArr()
                else:
                    scenario_arr = S.getWholeArr()

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

            ScenariosTrain = np.asarray(ScenariosTrain, dtype=np.float)
            IMsTrain = np.asarray(IMsTrain, dtype=np.float)
            IMsTrain3D = IMsTrain
            FeatureTrain = np.asarray(FeatureTrain, dtype=np.float)
            FeatureTrain = FeatureTrain[:, 0,:]

            LabelsTrain = np.asarray(LabelsTrain, dtype=np.float)
            LabelsTrain = LabelsTrain[:, 0]
            LabelsTrain = LabelsTrain[:, 0]
            IMsTrain1 = np.reshape(IMsTrain, [IMsTrain.shape[0], np.product(IMsTrain[0, :, :, :].shape)])

            ScenariosTest = np.asarray(ScenariosTest, dtype=np.float)
            IMsTest = np.asarray(IMsTest, dtype=np.float)
            IMsTest3D = IMsTest
            FeatureTest = np.asarray(FeatureTest, dtype=np.float)
            FeatureTest = FeatureTest[:, 0, :]

            # Endpoint_features_Test = Endpoint_features_Test[:, :, 0]
            LabelsTest = np.asarray(LabelsTest, dtype=np.float)
            LabelsTest = LabelsTest[:, 0]
            LabelsTest = LabelsTest[:, 0]
            IMsTest1 = np.reshape(IMsTest, [IMsTest.shape[0], np.product(IMsTest[0, :, :, :].shape)])

            # Sbhuffle Data
            indices = np.arange(len(ScenariosTrain))
            np.random.shuffle(indices)
            ScenariosTrain = ScenariosTrain[indices]
            IMsTrain = IMsTrain[indices]
            FeatureTrain = FeatureTrain[indices]

            LabelsTrain = LabelsTrain[indices]



            XIMs_train = IMsTrain1

            XFeature_train = FeatureTrain

            XScenarios_train = ScenariosTrain
            yIMs_train = LabelsTrain

            XIMs_test = IMsTest1
            XFeature_test = FeatureTest

            XScenarios_test = ScenariosTest
            yIMs_test = LabelsTest
            yFeature_test = LabelsTest
            yScenarios_test = LabelsTest


            print(XIMs_train.shape)
            print(XFeature_train.shape)
            print(XScenarios_train.shape)
            print(yIMs_train.shape)

            # Plot data count
            # z_train = Counter(yIMs_train)
            # sns.countplot(yIMs_train)

            # to ignore image data
            if UseIMage == False:
                XIMs_train = np.zeros(XIMs_train.shape)
                XIMs_test = np.zeros(XIMs_test.shape)
                print('Not Using Image')

            # PltNAme = 'NEW_SmUnet1TEST19INV_FEATURES_CONV=True_LR=0.001_100_sce_he_uniform_IM=1bchSiz=25_Use_IM=True_Epoch=200_run=1'

            # PltNAme = 'NEW_1_INV_FEATURES_CONV=False_LR=0.001_100_sce_he_uniform_IM=3bchSiz=50_Use_IM=True_Epoch=150_run=' + str(run+1)


            model = load_model(root_dir + '/data/' + modelFolder + '/IM=' + str(ImagetoTest) + '_run=' + str(run+1) + '.h5')

            filepath = root_dir + '/data/' + modelFolder + '/IM=' + str(ImagetoTest) + '_run=' + str(run+1) + ".hdf5"

            # filepath = root_dir + '/data/models/' + PltNAme + "_weights.max_val_acc.hdf5"


            print(filepath)
            # checkpoint_val_loss = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

            # filepath = 'DataFiles/' + PltNAme + "_weights.best_AUC.hdf5"
            # checkpoint_AUC = ModelCheckpoint(filepath, monitor=keras.metrics.AUC(), verbose=1, save_best_only=True,
            #                                       mode='max')

            # fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)
            if UseConv:
                X_IMs = IMsTrain3D.reshape(IMsTrain3D.shape[0], IMsTrain3D.shape[1], IMsTrain3D.shape[2],
                                           IMsTrain3D.shape[3],
                                           1)
                XIMs_test = IMsTest3D.reshape(IMsTest3D.shape[0], IMsTest3D.shape[1], IMsTest3D.shape[2],
                                           IMsTest3D.shape[3],
                                           1)
            else:
                XIMs_test = XIMs_test


            #######                Predict Values
            model.load_weights(filepath)
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            y_pred = model.predict([XIMs_test,XFeature_test])
            y_pred = y_pred[:,0]
            y_pred.shape




            #######                Check Correct Scenarios
            ClusterStr = sio.loadmat(root_dir + '/data/mat_Discovery/' +str(ImagetoTest)+'_L6_AS_withALLClusters1.mat')
            ClusterStr1 = ClusterStr['ClustersStr']
            y_pred_Final = np.zeros(shape=(ClusterStr1.shape[1],ClusterStr1[0,ClusterStr1.shape[1]-1]['cost_components'].shape[1]))
            #y_pred_Final = np.zeros(shape=(119,375))
            y_origina_Final = np.zeros(shape=(ClusterStr1.shape[1],ClusterStr1[0,ClusterStr1.shape[1]-1]['cost_components'].shape[1]))
            #y_origina_Final = np.zeros(shape=(119,375))
            counter = 0
            for i in range(ClusterStr1.shape[1]):#54
                C2 = ClusterStr1[0,i]['cost_components'].shape[1]
                if C2 >100:
                    C2 = 100
                for s in range(C2):
                    y_pred_Final[i,s] = y_pred[counter]
                    y_origina_Final[i,s] = LabelsTest[counter]
                    counter = counter + 1

            # sio.savemat('E:/AutomatedTracing/Data/TrainingData/scenarios_images_features/Final_Shuffled_Matrix_Predict_IM_'+str(ImagetoTest)+'_moldel='+str(modelnum+1)+'_run='+str(run+1)+'_EndpointFeatures.mat',{"y_pred":y_pred_Final})
            resultfile = 'E:/AutomatedTracing/Data/TrainingData/scenarios_images_features/Discovery3run_Final_Shuffled_Matrix_Predict_IM_' + str(
                    ImagetoTest) + '_moldel=' + str(modelnum + 1) + '_run=' + str(run + 1) + '_NewFeatures_li100_SmallUnet1_.mat'
            sio.savemat(resultfile
                ,
                {"y_pred": y_pred_Final})
            print(resultfile)

            corrctnum = 0
            incorrectnum = 0
            # for i in range(y_pred_Final.shape[0]):
            #     Pr = max(y_pred_Final[i, :])
            #     Or = max(y_origina_Final[i,:])
            #     if round(Pr) == round(Or):
            #         corrctnum = corrctnum +1
            #     else:
            #         incorrectnum = incorrectnum + 1

            for i in range(y_pred_Final.shape[0]):
                if np.argmax(y_pred_Final[i, :])==np.argmax(y_origina_Final[i, :]):
                    corrctnum = corrctnum +1
                else:
                    incorrectnum = incorrectnum + 1




            print("Incorrect Scenario Connections: ",incorrectnum)
            print("Total Scenarios: ",corrctnum+incorrectnum)
            # print("Incorrect Scenario Connections: ",incorrectnum)

            # TP = corrctnum
            # FN = incorrectnum
            #
            Result[0,ImagetoTest-1,run] = incorrectnum # correct scenario
            Result[1, ImagetoTest-1, run] = corrctnum+incorrectnum # total scenario



import scipy.io as io
# basePath = 'E:/AutomatedTracing/AutomatedTracing/Python/MachineLeatningAutomatedTracing/DataFiles/Tensorboard/All_points/'
basePath = 'E:/AutomatedTracing/TraceProofreading/'
io.savemat(basePath+'Discovery3run_Final_Shuffled_Matrix_Predict_IM_model=' + str(modelnum + 1) + '.mat', mdict={'Result': Result})
print('Done!')




# import scipy.io as io
# basePath = 'E:/AutomatedTracing/AutomatedTracing/Python/MachineLeatningAutomatedTracing/DataFiles/Tensorboard/All_points/'
# io.savemat(basePath+'IM='+str(ImagetoTest)+'_NewFeatures_li100_reg_com_SmallUnet1_Discovery.mat', mdict={'Result': Result})
# print('Result Data: '+ basePath+'IM='+str(ImagetoTest)+'_NewFeatures_li100_reg_com_SmallUnet1_.mat')
#
# print(Result)
#
# Result_data = Result[0,:,:]
#
# print(Result_data.mean(axis=1))
#
# print("Done!")
