function [] = Plot_ATvsAT(mother1,mother2)

addpath('NeuronTracerV20')
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
