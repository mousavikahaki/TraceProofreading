% clear all
% close all
clc
addpath('C:\Users\Seyed\Documents\DatasetTests\AutomatedTracing\AutomatedTracing\NeuronTracerV20');
parameter_file_path='C:\Users\Seyed\Documents\DatasetTests\AutomatedTracing\AutomatedTracing\NeuronTracerV20\Parameter Files\Parameters_L6.txt';
Parameters=Read_Parameter_File(parameter_file_path);
user = ['AS';'RG';'JC'];
imgnum = 1;
modelnum = 1;
% run = 2; 


usernum = 1;
for run = 4:5
manual = load(['E:\AutomatedTracing\Data\Traces\L1_org\',num2str(imgnum),'_L6_',user(usernum,:),'_withALLClusters1.mat']);
ClusterStr_manual = manual.ClustersStr;

manual = load(['E:\AutomatedTracing\Data\Traces\L1\',num2str(imgnum),'_L6_',user(usernum,:),'.mat']);
AMlbl_manual = manual.AM;
r_manual = manual.r;
R_manual = manual.R;
IM=manual.Original;

% figure
% imshow(max(IM,[],3))
% hold on
% PlotAM_1(AMlbl_manual,r_manual)
% title('Manual Trace');        

%[AMlbl_initial,r_initial,R_initial,ClusterStr_automated]=GenerateClustersStrW_new(IM,AMlbl_manual,r_manual,R_manual,Parameters);
[AMlbl_initial,r_initial,R_initial] = Disconnect_Branches(AMlbl_manual,r_manual,R_manual);

% figure
% imshow(max(IM,[],3))
% hold on
% PlotAM_1(AMlbl_initial,r_initial)
% title('Initial Trace');

% load('E:/AutomatedTracing/Data/TrainingData/scenarios_images_features/AllforFinal_Prediction6_L6_AS_NEW_TEST.mat');
% load('E:/AutomatedTracing/Data/TrainingData/scenarios_images_features/AllforFinal_Prediction6_L6_AS_NEW_1-5.mat');

% load(['E:/AutomatedTracing/Data/TrainingData/scenarios_images_features/Shuffled_Matrix_Predict_IM_',num2str(imgnum),'_out.mat']);


% Old Features
% load(['E:/AutomatedTracing/Data/TrainingData/scenarios_images_features/Final_Shuffled_Matrix_Predict_IM_',num2str(imgnum),'_moldel=',num2str(modelnum),'_run=',num2str(run),'.mat']);

% End point Features
% load(['E:/AutomatedTracing/Data/TrainingData/scenarios_images_features/Final_Shuffled_Matrix_Predict_IM_',num2str(imgnum),'_moldel=',num2str(modelnum),'_run=',num2str(run),'_EndpointFeatures.mat']);

% End point Features with Augment
% load(['E:/AutomatedTracing/Data/TrainingData/scenarios_images_features/Final_Shuffled_Matrix_Predict_IM_',num2str(imgnum),'_moldel=',num2str(modelnum),'_run=',num2str(run),'_EndpointFeatures_Epoh=150_AllData_fixed_augmented_limit100.mat']);

% load(['E:/AutomatedTracing/Data/TrainingData/scenarios_images_features/Final_Shuffled_Matrix_Predict_IM_',num2str(imgnum),'_moldel=',num2str(modelnum),'_run=',num2str(run),'_NewFeatures_li100.mat']);

load(['E:/AutomatedTracing/Data/TrainingData/scenarios_images_features/Final_Shuffled_Matrix_Predict_IM_',num2str(imgnum),'_moldel=',num2str(modelnum),'_run=',num2str(run),'_NewFeatures_li100_SmallUnet1.mat']);

% 
% for numCluster = 1: size(ClusterStr_manual,2)
%     if size(ClusterStr_manual(numCluster).scenarios,3) > 100
%         ClusterStr_manual(numCluster).scenarios = ClusterStr_manual(numCluster).scenarios(:,:,1:100);
%         ClusterStr_manual(numCluster).cost_components = ClusterStr_manual(numCluster).cost_components(:,1:100);
%         ClusterStr_manual(numCluster).best_merger = ClusterStr_manual(numCluster).best_merger(:,1:100);
%         ClusterStr_manual(numCluster).alpha = ClusterStr_manual(numCluster).alpha(:,1:100);
%     end
% end



ClusterStr_automated=ClusterStr_manual;

merging_errors=[];
for i = 1:size(ClusterStr_automated,2)
    ClusterStr_automated(i).predicted_merger = y_pred(i,1:length(ClusterStr_automated(i).best_merger));
    [~,automated_ind]=max(ClusterStr_automated(i).predicted_merger);
    manual_ind=find(ClusterStr_manual(i).best_merger==1);
    if automated_ind~=manual_ind
        merging_errors=[merging_errors,i];
    end
end
[AMlbl_automated,r_automated,R_automated]=BranchMerger(AMlbl_initial,r_initial,R_initial,ClusterStr_automated);

figure
imshow(max(IM,[],3))
hold on
PlotAM_1(AMlbl_automated,r_automated)
% title(['Automated Trace: ',num2str(length(merging_errors)),'/',num2str(length(ClusterStr_manual)),' errors']);
for i=1:length(merging_errors)
    r_error=mean(ClusterStr_manual(merging_errors(i)).end_point_r);
%     plot3(r_error(2),r_error(1),r_error(3),'c*','markersize',12)
end
end
