function [] = Rohan_Plots_1(Dag_trace_full,Dag_TP_full,Dag_BP_full,fp_length,fn_length,fp_TP,fn_TP,fp_BP,fn_BP)

plot1(Dag_trace_full,0.25:0.5:4.75,[0,1,2,3,4,5]);
plot1(Dag_TP_full,0.25:0.5:9.75,[0,5,10]);
plot1(Dag_BP_full,0.25:0.5:9.75,[0,5,10]);

% fp_length_hist = hist(fp_length,250:500:3750);
% fn_length_hist = hist(fn_length,250:500:3750);
% fp_TP_hist = hist(fp_TP,5:10:60);
% fn_TP_hist = hist(fn_TP,5:10:60);
% fp_BP_hist = hist(fp_BP,5:10:60);
% fn_BP_hist = hist(fn_BP,5:10:60);
bar2(fp_length,fn_length,250:500:3750);
bar2(fp_TP,fn_TP,5:10:60);
bar2(fp_BP,fn_BP,5:10:60);
% figure,bar([fp_length_hist',-fn_length_hist'])
% figure,bar([fp_TP_hist',-fn_TP_hist'])
% figure,bar([fp_BP_hist',-fn_BP_hist'])

function [] = plot1(y,x,xtic)
y_hist = hist(y,x);
figure,plot(x,2*y_hist/length(y),'.-')
xticks(xtic)

function [] = bar2(yp,yn,x)
yp_hist = hist(yp,x);
yn_hist = hist(yn,x);
figure,bar(x,[yp_hist',-yn_hist'])