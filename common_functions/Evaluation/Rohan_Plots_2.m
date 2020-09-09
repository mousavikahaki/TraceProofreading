function [] = Rohan_Plots_2(Dag_trace_full1,Dag_TP_full1,Dag_BP_full1,fp_length1,fn_length1,fp_TP1,fn_TP1,fp_BP1,fn_BP1,Dag_trace_full2,Dag_TP_full2,Dag_BP_full2,fp_length2,fn_length2,fp_TP2,fn_TP2,fp_BP2,fn_BP2)

fp_length = fp_length1;
fn_length = fn_length1;
fp_TP = fp_TP1;
fn_TP = fn_TP1;
fp_BP = fp_BP1;
fn_BP = fn_BP1;

plot2(Dag_trace_full1,Dag_trace_full2,0.25:0.5:5.25,[0,1,2,3,4,5,6]);
title('Distance between traces, [Voxels]');
fig = gca;
movegui(fig,[100 600]);

plot2(Dag_TP_full1,Dag_TP_full2,0.25:0.5:9.75,[0,5,10]);
title('Distance between terminals, [Voxels]');
fig = gca;
movegui(fig,[670 600]);

plot2(Dag_BP_full1,Dag_BP_full2,0.25:0.5:9.75,[0,5,10]);
title('Distance between branch points, [Voxels]');
fig = gca;
movegui(fig,[1240 600]);


bar2(fp_length1,fn_length1,fp_length2,fn_length2,250:500:3750);
ylim([-1,1]);
title('Mismatch in length, [Voxels]');
fig = gca;
movegui(fig,[100 80]);

bar2(fp_TP1,fn_TP1,fp_TP2,fn_TP2,5:10:60);
ylim([-1,1]);
title('# terminal point errors');
fig = gca;
movegui(fig,[670 80]);

bar2(fp_BP1,fn_BP1,fp_BP2,fn_BP2,5:10:60);
ylim([-1,1]);
title('# branch point errors');
fig = gca;
movegui(fig,[1240 80]);


function [] = plot1(y,x,xtic)
y_hist = hist(y,x);
figure,plot(x,2*y_hist/length(y),'.-')
xticks(xtic)
axis square

function [] = plot2(y1,y2,x,xtic)
y1_hist = hist(y1,x);
y2_hist = hist(y2,x);
figure,plot(x(1:end-1),2*y1_hist(1:end-1)/length(y1),'.-')
hold on
plot(x(1:end-1),2*y2_hist(1:end-1)/length(y2),'.-')
xticks(xtic)
axis square

function [] = bar1(yp,yn,x)
yp_hist = hist(yp,x);
yn_hist = hist(yn,x);
figure,bar(x,[yp_hist'/length(yp),-yn_hist'/length(yn)])
axis square

function [] = bar2(yp1,yn1,yp2,yn2,x)
yp1_hist = hist(yp1,x);
yn1_hist = hist(yn1,x);
yp2_hist = hist(yp2,x);
yn2_hist = hist(yn2,x);
figure,
b = bar(x,[yp1_hist'/length(yp1),yp2_hist'/length(yp2),-yn1_hist'/length(yn1),-yn2_hist'/length(yn2)]);
shift = [-3,-1,1,3]*max(b(1).XData)/75;
yValues = repmat(x,4,1)+shift';
A= [yp1_hist'/length(yp1),yp2_hist'/length(yp2),-yn1_hist'/length(yn1),-yn2_hist'/length(yn2)]';
text(yValues(:)', A(:),num2str(round(A(:),3)),'Rotation',90);

b(1).FaceColor = 'blue';
b(2).FaceColor = 'red';
b(3).FaceColor = 'blue';
b(4).FaceColor = 'red';
axis square