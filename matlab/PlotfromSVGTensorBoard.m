%Plot from SVG (TensorBoard)
clear all 
close all

doplot = false;
R_Val_Acc=loadsvg('C:\Users\Seyed\Downloads\val_acc.svg',0.2,doplot);
R_Val_loss=loadsvg('C:\Users\Seyed\Downloads\val_loss.svg',0.2,doplot);
R_loss=loadsvg('C:\Users\Seyed\Downloads\loss_.svg',0.2,doplot);
R_Acc=loadsvg('C:\Users\Seyed\Downloads\acc.svg',0.2,doplot);



for i = 1:6
    [s,d] = cellfun(@size,R_Val_Acc);
    maxim = min(s);
    Val_Acc(:,i) = R_Val_Acc{i}(1:maxim,2);
    
    [s,d] = cellfun(@size,R_Val_loss);
    maxim = min(s);
    Val_loss(:,i) = R_Val_loss{i}(1:maxim,2);
    
    [s,d] = cellfun(@size,R_loss);
    maxim = min(s);
    loss(:,i) = R_loss{i}(1:maxim,2);
    
    [s,d] = cellfun(@size,R_Acc);
    maxim = min(s);
    Acc(:,i) = R_Acc{i}(1:maxim,2);
    
    
end

%Mean on Different runs
acc_m = mean(Acc,2);
Val_Acc_m = mean(Val_Acc,2);
loss_m = mean(loss,2);
Val_loss_m = mean(Val_loss,2);

%Normalize
minVal = min(acc_m);
maxVal = max(acc_m);
acc_m = (acc_m - minVal) / ( maxVal - minVal );
minVal = min(Val_Acc_m);
maxVal = max(Val_Acc_m);
Val_Acc_m = (Val_Acc_m - minVal) / ( maxVal - minVal );
minVal = min(loss_m);
maxVal = max(loss_m);
loss_m = (loss_m - minVal) / ( maxVal - minVal );
minVal = min(Val_loss_m);
maxVal = max(Val_loss_m);
Val_loss_m = (Val_loss_m - minVal) / ( maxVal - minVal );

%Ploting
figure,plot(acc_m);
hold on
plot(Val_Acc_m);
hold on
plot(loss_m);
hold on
plot(Val_loss_m);
hold on
legend('Validation Loss','Training Loss','Validation Accuracy','Training Accuracy')

clear all
load('E:\AutomatedTracing\AutomatedTracing\Python\MachineLeatningAutomatedTracing\DataFiles\Tensorboard\All_points\IM=6_NewFeatures_li100_reg_com_SmallUnet1.mat')
data = [];
dataorg(:,:) = Result(1,:,:);
data(:,:) = Result(1,[2,4,3,6,1,5],:);

y = mean(data);
errors = std(data);
x = 1:size(data,2);
figure,plot(x, y)
hold on;
h = errorbar(x, y, errors);
set(h,'linestyle','none')
