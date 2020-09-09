function [] = Plot_ATvsMT(mother1,numIMs)
addpath('NeuronTracerV20')
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
    for num = numIMs
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
mother = ['E:\AutomatedTraceResults\user_user'];
for i = 1:3
    for j = 1:3
        if i ~= j
            for num = numIMs
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