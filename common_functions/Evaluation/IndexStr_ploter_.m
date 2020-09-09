%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% user vs user
close all
clc
addpath('../NeuronTracerV20')
user = ['AS';'RG';'JC'];
metric_plot1 = [];
metric_plot2 = [];
metric_plot3 = [];
fp_length = [];
fn_length = [];
fp_TP = [];
fn_TP = [];
fp_BP = [];
fn_BP = [];
% mother = ['Connectivity_Metrics&IndexStr_ploter&Seed_F_D\user_user'];
% for i = 1:3
%     for j = 1:3
%         if i ~= j
%             for num = 1:6
%                 son = [mother,'\',num2str(num),'_',user(j,:),'_',user(i,:)];
%                 stru = load(son,'IndexStr','AM_G', 'r_G', 'R_G', 'AM_A', 'r_A', 'R_A', 'h_length','h_bp','h_tp');
%                 metric_plot1 = [metric_plot1;stru.IndexStr.Dag_trace_full];
%                 metric_plot2 = [metric_plot2;stru.IndexStr.Dag_TP_full];
%                 metric_plot3 = [metric_plot3;stru.IndexStr.Dag_BP_full];
%                 fp_length = [fp_length;stru.IndexStr.fp_length];
%                 fn_length = [fn_length;stru.IndexStr.fn_length];
%                 fp_TP = [fp_TP;stru.IndexStr.fp_TP];
%                 fn_TP = [fn_TP;stru.IndexStr.fn_TP];
%                 fp_BP = [fp_BP;stru.IndexStr.fp_BP];
%                 fn_BP = [fn_BP;stru.IndexStr.fn_BP]; 
%             end
%         end
%     end
% end
% 
% Rohan_Plots_1(metric_plot1,metric_plot2,metric_plot3,fp_length,fn_length,fp_TP,fn_TP,fp_BP,fn_BP)

mother1 = ['Connectivity_Metrics&IndexStr_ploter&Seed_F_D\DefensePPT_ImoImo_15_1_200_CutNet99_alpha05_onlyMST_b20_mesh3_postpro'];
mother2 = ['Connectivity_Metrics&IndexStr_ploter&Seed_F_D\net03+Astar03+D04_ImfImo_15_1_200_CutNet98'];

% compare 2 AT
% Plot_AT2(mother1,mother2)

% Compare AT and Manual Trace
Plot_ATvsMT(mother2)

% mother = ['E:\Shih-Luen\Lab\Publications\Thesis\Thesis\emf\Roahn_Metrics\net_MST_ImfImo_15'];
% Plot_AT(mother)
% mother = ['E:\Shih-Luen\Lab\Publications\Thesis\Thesis\emf\Roahn_Metrics\AT_user'];
% Plot_AT(mother)

function [] = Plot_AT(mother)

addpath('../NeuronTracerV20')
user = ['AS';'RG';'JC'];
metric_plot1 = [];
metric_plot2 = [];
metric_plot3 = [];
fp_length = [];
fn_length = [];
fp_TP = [];
fn_TP = [];
fp_BP = [];
fn_BP = [];
for j = 1:3
    for num = 1:6
        son = [mother,'\',num2str(num),'_',user(j,:)];
        stru = load(son,'IndexStr','AM_G', 'r_G', 'R_G', 'AM_A', 'r_A', 'R_A', 'h_length','h_bp','h_tp');
        metric_plot1 = [metric_plot1;stru.IndexStr.Dag_trace_full];
        metric_plot2 = [metric_plot2;stru.IndexStr.Dag_TP_full];
        metric_plot3 = [metric_plot3;stru.IndexStr.Dag_BP_full];
        fp_length = [fp_length;stru.IndexStr.fp_length];
        fn_length = [fn_length;stru.IndexStr.fn_length];
        fp_TP = [fp_TP;stru.IndexStr.fp_TP];
        fn_TP = [fn_TP;stru.IndexStr.fn_TP];
        fp_BP = [fp_BP;stru.IndexStr.fp_BP];
        fn_BP = [fn_BP;stru.IndexStr.fn_BP];       
    end
end

Rohan_Plots_1(metric_plot1,metric_plot2,metric_plot3,fp_length,fn_length,fp_TP,fn_TP,fp_BP,fn_BP)

end

function [] = Plot_AT2(mother1,mother2)

addpath('../NeuronTracerV20')
user = ['AS';'RG';'JC'];
DT1 = [];
DT2 = [];
DTP1 = [];
DTP2 = [];
DBP1 = [];
DBP2 = [];
fp_length1 = [];
fp_length2 = [];
fn_length1 = [];
fn_length2 = [];
fp_TP1 = [];
fp_TP2 = [];
fn_TP1 = [];
fn_TP2 = [];
fp_BP1 = [];
fp_BP2 = [];
fn_BP1 = [];
fn_BP2 = [];
for j = 1:3
    for num = 1:6
        son1 = [mother1,'\',num2str(num),'_',user(j,:)];
        son2 = [mother2,'\',num2str(num),'_',user(j,:)];
        stru1 = load(son1,'IndexStr','AM_G', 'r_G', 'R_G', 'AM_A', 'r_A', 'R_A', 'h_length','h_bp','h_tp');
        stru2 = load(son2,'IndexStr','AM_G', 'r_G', 'R_G', 'AM_A', 'r_A', 'R_A', 'h_length','h_bp','h_tp');
        DT1 = [DT1;stru1.IndexStr.Dag_trace_full];
        DT2 = [DT2;stru2.IndexStr.Dag_trace_full];
        DTP1 = [DTP1;stru1.IndexStr.Dag_TP_full];
        DTP2 = [DTP2;stru2.IndexStr.Dag_TP_full];
        DBP1 = [DBP1;stru1.IndexStr.Dag_BP_full];
        DBP2 = [DBP2;stru2.IndexStr.Dag_BP_full];
        fp_length1 = [fp_length1;stru1.IndexStr.fp_length];
        fp_length2 = [fp_length2;stru2.IndexStr.fp_length];
        fn_length1 = [fn_length1;stru1.IndexStr.fn_length];
        fn_length2 = [fn_length2;stru2.IndexStr.fn_length];
        fp_TP1 = [fp_TP1;stru1.IndexStr.fp_TP];
        fp_TP2 = [fp_TP2;stru2.IndexStr.fp_TP];
        fn_TP1 = [fn_TP1;stru1.IndexStr.fn_TP];
        fn_TP2 = [fn_TP2;stru2.IndexStr.fn_TP];
        fp_BP1 = [fp_BP1;stru1.IndexStr.fp_BP];
        fp_BP2 = [fp_BP2;stru2.IndexStr.fp_BP];
        fn_BP1 = [fn_BP1;stru1.IndexStr.fn_BP]; 
        fn_BP2 = [fn_BP2;stru2.IndexStr.fn_BP]; 
    end
end

Rohan_Plots_2(DT1,DTP1,DBP1,fp_length1,fn_length1,fp_TP1,fn_TP1,fp_BP1,fn_BP1,DT2,DTP2,DBP2,fp_length2,fn_length2,fp_TP2,fn_TP2,fp_BP2,fn_BP2)

end

function [] = Plot_ATvsMT(mother1)

addpath('../NeuronTracerV20')
user = ['AS';'RG';'JC'];
DT1 = [];
DTP1 = [];
DBP1 = [];
fp_length1 = [];
fn_length1 = [];
fp_TP1 = [];
fn_TP1 = [];
fp_BP1 = [];
fn_BP1 = [];
for j = 1:3
    for num = 1:6
        son1 = [mother1,'\',num2str(num),'_',user(j,:)];
        stru1 = load(son1,'IndexStr','AM_G', 'r_G', 'R_G', 'AM_A', 'r_A', 'R_A', 'h_length','h_bp','h_tp');
        DT1 = [DT1;stru1.IndexStr.Dag_trace_full];
        DTP1 = [DTP1;stru1.IndexStr.Dag_TP_full];
        DBP1 = [DBP1;stru1.IndexStr.Dag_BP_full];
        fp_length1 = [fp_length1;stru1.IndexStr.fp_length];
        fn_length1 = [fn_length1;stru1.IndexStr.fn_length];
        fp_TP1 = [fp_TP1;stru1.IndexStr.fp_TP];
        fn_TP1 = [fn_TP1;stru1.IndexStr.fn_TP];
        fp_BP1 = [fp_BP1;stru1.IndexStr.fp_BP];
        fn_BP1 = [fn_BP1;stru1.IndexStr.fn_BP]; 
    end
end

DT = [];
DTP = [];
DBP = [];
fp_length = [];
fn_length = [];
fp_TP = [];
fn_TP = [];
fp_BP = [];
fn_BP = [];
mother = ['Connectivity_Metrics&IndexStr_ploter&Seed_F_D\user_user'];
for i = 1:3
    for j = 1:3
        if i ~= j
            for num = 1:6
                son = [mother,'\',num2str(num),'_',user(j,:),'_',user(i,:)];
                stru = load(son,'IndexStr','AM_G', 'r_G', 'R_G', 'AM_A', 'r_A', 'R_A', 'h_length','h_bp','h_tp');
                DT = [DT;stru.IndexStr.Dag_trace_full];
                DTP = [DTP;stru.IndexStr.Dag_TP_full];
                DBP = [DBP;stru.IndexStr.Dag_BP_full];
                fp_length = [fp_length;stru.IndexStr.fp_length];
                fn_length = [fn_length;stru.IndexStr.fn_length];
                fp_TP = [fp_TP;stru.IndexStr.fp_TP];
                fn_TP = [fn_TP;stru.IndexStr.fn_TP];
                fp_BP = [fp_BP;stru.IndexStr.fp_BP];
                fn_BP = [fn_BP;stru.IndexStr.fn_BP]; 
            end
        end
    end
end

Rohan_Plots_2(DT,DTP,DBP,fp_length,fn_length,fp_TP,fn_TP,fp_BP,fn_BP,DT1,DTP1,DBP1,fp_length1,fn_length1,fp_TP1,fn_TP1,fp_BP1,fn_BP1)

end


% function [AM_A,r_A,R_A] = post_processing(AM_A,r_A,R_A)
% [AM_A,r_A,R_A] = AdjustPPM(AM_A,r_A,R_A,1);
% % [AM_A,r_A,~] = Eliminate_Small_Trees(AM_A,r_A,zeros(size(r_A,1)),100);
% % AM_A=LabelBranchesAM(AM_A);
% % [AM_A,r_A,~] = Eliminate_Terminal_Branches(AM_A,r_A,20,1,1);
% % [AM_A,r_A,~] = Eliminate_Terminal_Branches(AM_A,r_A,20,1,1);
% % [AM_A,r_A,~] = Eliminate_Terminal_Branches(AM_A,r_A,20,1,1);
% % AM_A=LabelTreesAM(AM_A);
% end