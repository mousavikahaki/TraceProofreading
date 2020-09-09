filename = 'OP_image1_sample1000times_dim100_Dissertation_v2';
load(['../../data/training_result/',num2str(filename)])

plot_StandardError(CVCV);
ylim([0,0.8])
% title('Coefficient of variation of axon intensity')

plot_StandardError(RaRa);
ylim([0.8,2.2])
% title('Mean axon radius')

plot_StandardError(NTSR_);
ylim([0,0.07])
% title('Background to axon intensity ratio')

plot_StandardError(IJIJ);
ylim([0.6,1.1])
% title('Normalized intensity in axon ross-over regions')

function [] = plot_StandardError(matrix)
[row, ~] = find(isnan(matrix)|isinf(matrix));
urow = unique(row);
matrix(urow,:)=[];
figure,
errorbar(1:size(matrix,2),mean(matrix),std(matrix,[],1)/sqrt(size(matrix,1)),'.')
axis square, xlim([0,size(matrix,2)+1])
end

function [Irc] = plot_Scatter0(Pd,Ir)
[row, ~] = find(isnan(Pd)|isnan(Ir));
urow = unique(row);
Pd(urow,:)=[];
Ir(urow,:)=[];

figure
for i = 1:size(Pd,1)
    for j = 1:size(Pd,2)
        scatter(Pd(i,j),Ir(i,j),[],[0,1-j/size(Pd,2),j/size(Pd,2)])
        hold on
    end
end
axis square
end

function [Irc] = plot_Scatter(Pd,Ir)
[row, ~] = find(isnan(Pd)|isnan(Ir));
urow = unique(row);
Pd(urow,:)=[];
Ir(urow,:)=[];

d = unique(Pd);
Irc = cell(length(d),size(Ir,2));
for i = 1:length(d)
    for j = 1:size(Ir,2)
        temp = Ir(:,j);
        Irc{i,j} = temp(Pd(:,j)==d(i));
    end
end

figure,
for j = 1:size(Ir,2)
    for i = 1:length(d)
        errorbar(d(i),mean(Irc{i,j}),std(Irc{i,j}),'.','color',[0,1-j/size(Ir,2),j/size(Ir,2)])
        hold on
    end
end
axis square
end

function [Irc] = plot_Scatter2(Pd,Ir)
[row, ~] = find(isnan(Pd)|isnan(Ir));
urow = unique(row);
Pd(urow,:)=[];
Ir(urow,:)=[];

d = unique(Pd);
Irc = cell(length(d),size(Ir,2));
for i = 1:length(d)
    for j = 1:size(Ir,2)
        temp = Ir(:,j);
        Irc{i,j} = temp(Pd(:,j)==d(i));
    end
end

for i = 1:length(d)
    figure,
    mean_std = zeros(2,size(Ir,2));
    for j = 1:size(Ir,2)
        mean_std(1,j) = mean(Irc{i,j});
        mean_std(2,j) = std(Irc{i,j})/sqrt(length(Irc{i,j}));
    end
    errorbar(1:size(Ir,2),mean_std(1,:),mean_std(2,:),'.')
    axis square
    legend(num2str(d(i)))
    ylim([0,2])
    xlim([0,size(Ir,2)+1])
end
end

function [mmean,msem] = meansem(matrix)
[row, ~] = find(isnan(matrix));
urow = unique(row);
matrix(urow,:)=[];
mmean = mean(matrix);
msem = std(matrix,[],1)/sqrt(size(matrix,1));
end