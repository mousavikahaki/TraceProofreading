function [] = Rohan_Plots(IndexStr)

trace_hist = hist(IndexStr.Dag_trace_full,0.25:0.5:4.75);
TP_hist = hist(IndexStr.Dag_TP_full,0.25:0.5:9.75);
BP_hist = hist(IndexStr.Dag_BP_full,0.25:0.5:9.75);

figure,plot(2*trace_hist/length(IndexStr.Dag_trace_full),'.-')
figure,plot(2*TP_hist/length(IndexStr.Dag_TP_full),'.-')
figure,plot(2*BP_hist/length(IndexStr.Dag_BP_full),'.-')