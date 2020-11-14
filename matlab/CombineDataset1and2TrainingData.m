clear all
clc

D1 = load('E:\AutomatedTraceResults\DataForConnectingTraining\Data_For_AE_BranchScenarios\IMonce_limit100scen_NEW_Inv_FEATURES.mat');
D2 = load('E:\AutomatedTraceResults\DataForConnectingTraining\Data_For_AE_BranchScenarios\S2B_IMonce_100_scen_NEW_Inv_FEATURES_User=SK.mat');

counter = 0;
Labels = {};
IMs = {};
IMnum = {};
NewFeatures = {};
Scenarios = {};

for i = 1:size(D1.IMnum,2)   
    counter = counter + 1;
    
    Labels{1,counter} = D1.Labels{i};
    IMs{1,counter} = D1.IMs{i};
    IMnum{1,counter} = D1.IMnum{i};
    NewFeatures{1,counter} = D1.NewFeatures{i};
    Scenarios{1,counter} = D1.Scenarios{i};
end

for i = 1:size(D2.IMnum,2)   
    counter = counter + 1;
    
    Labels{1,counter} = D2.Labels{i};
    IMs{1,counter} = D2.IMs{i};
    IMnum{1,counter} = D2.IMnum{i}+6;
    NewFeatures{1,counter} = D2.NewFeatures{i};
    Scenarios{1,counter} = D2.Scenarios{i};
end
path = 'E:\AutomatedTraceResults\DataForConnectingTraining\Data_For_AE_BranchScenarios\S1and2_IMonce_100_scen_NEW_Inv_FEATURES_User=SK.mat';
save(path,'IMs','Scenarios','Labels','IMnum','NewFeatures');