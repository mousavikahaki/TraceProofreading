function runEvaluation%(ResultPath,numIMs)
% close all
numIMs = [1];
% Mine Generated: E:\AutomatedTracing\AutomatedTracing\MATLAB\AT_Mine\At.m
% ResultPath = 'E:\AutomatedTracing\Data\AT_Results\mine1';

% ShihLuen code generated:
% C:\Users\Seyed\Documents\DatasetTests\AutomatedTracing\AutomatedTracing\CompareTraces_test_defensePPT.m
%  ResultPath = 'E:\AutomatedTracing\Data\AT_Results\mine_withScenarioAI';
%ResultPath = 'E:\AutomatedTraceResults\fImo_15_1_200_CutNet95_alpha05_onlyMST_b20_mesh1_scenarios';
ResultPath = 'E:\AutomatedTracing\Data\AT_Results\AT_Keras\';
% Get results from Shih-Luen
% ResultPath = 'Connectivity_Metrics&IndexStr_ploter&Seed_F_D\net03+Astar03+D04_ImfImo_15_1_200_CutNet98';

addpath('../NeuronTracerV20');

Plot_ATvsMT(ResultPath,numIMs)

% Connectivity_Metrics(ResultPath)

% Plot_ATvsAT(ResultPath1,ResultPath2)

% imNum = 1;
% ShowAT('E:\AutomatedTracing\Data\AT_Results\mine1\1_AS.mat',imNum);






