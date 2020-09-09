function Connectivity_Metrics(ResultPath)
d_corr = 3;
N_sample = 1000;
repeat = 1;
ppm = 1;
dd = 15;
im1 = 'f';
im2 = 'o';
addpath('..\NeuronTracerV20')
user = ['AS';'RG';'JC'];
% ResultPath = ['Connectivity_Metrics&IndexStr_ploter&Seed_F_D\net03+Astar03+D04_ImfImo_15_1_200_CutNet98'];
conf_ASJC = zeros(2);
conf_ASRG = zeros(2);
conf_ASA = zeros(2);
r_AS_corr = cell(6,1);
r_JC_corr = cell(6,1);
r_RG_corr = cell(6,1);
r_A_corr = cell(6,1);
AM_AS_corr_D = cell(6,1);
AM_AS_corr = cell(6,1);
AM_JC_corr = cell(6,1);
AM_RG_corr = cell(6,1);
AM_A_corr = cell(6,1);

bin = [0,50,100,150,200,250,300,350,400,450,512];
FR_JC = zeros([length(bin)-1,3,6*repeat]);
FR_RG = zeros([length(bin)-1,3,6*repeat]);
FR_A = zeros([length(bin)-1,3,6*repeat]);

count = 0;
for num = 1:6
    
    AS = load([ResultPath,'\',num2str(num),'_AS.mat']);
    JC = load([ResultPath,'\',num2str(num),'_JC.mat']);
    RG = load([ResultPath,'\',num2str(num),'_RG.mat']);
    AM_AS = AS.AM_G;
    AM_JC = JC.AM_G;
    AM_RG = RG.AM_G;
    AM_A = AS.AM_A;
    r_AS = AS.r_G;
    r_JC = JC.r_G;
    r_RG = RG.r_G;
    r_A = AS.r_A;
    R_AS = AS.R_G;
    R_JC = JC.R_G;
    R_RG = RG.R_G;
    R_A = AS.R_A;
    [AM_AS,r_AS,R_AS] = AdjustPPM(AM_AS,r_AS,R_AS,ppm);
    [AM_JC,r_JC,R_JC] = AdjustPPM(AM_JC,r_JC,R_JC,ppm);
    [AM_RG,r_RG,R_RG] = AdjustPPM(AM_RG,r_RG,R_RG,ppm);
    [AM_A,r_A,R_A] = AdjustPPM(AM_A,r_A,R_A,ppm);
    
    r_minD = zeros(length(r_AS),6);
    

%     ASJC = ((r_AS(:,1)-r_JC(:,1)').^2+(r_AS(:,2)-r_JC(:,2)').^2+(r_AS(:,3)-r_JC(:,3)').^2).^0.5;
%     ASRG = ((r_AS(:,1)-r_RG(:,1)').^2+(r_AS(:,2)-r_RG(:,2)').^2+(r_AS(:,3)-r_RG(:,3)').^2).^0.5;
%     ASA = ((r_AS(:,1)-r_A(:,1)').^2+(r_AS(:,2)-r_A(:,2)').^2+(r_AS(:,3)-r_A(:,3)').^2).^0.5;
%     [r_minD(:,1),r_minD(:,2)]=min(ASJC,[],2);
%     [r_minD(:,3),r_minD(:,4)]=min(ASRG,[],2);
%     [r_minD(:,5),r_minD(:,6)]=min(ASA,[],2);
    
    for i = 1:length(r_AS)
        ASJC = ((r_AS(i,1)-r_JC(:,1)').^2+(r_AS(i,2)-r_JC(:,2)').^2+(r_AS(i,3)-r_JC(:,3)').^2).^0.5;
        ASRG = ((r_AS(i,1)-r_RG(:,1)').^2+(r_AS(i,2)-r_RG(:,2)').^2+(r_AS(i,3)-r_RG(:,3)').^2).^0.5;
        ASA = ((r_AS(i,1)-r_A(:,1)').^2+(r_AS(i,2)-r_A(:,2)').^2+(r_AS(i,3)-r_A(:,3)').^2).^0.5;
        [r_minD(i,1),r_minD(i,2)]=min(ASJC);
        [r_minD(i,3),r_minD(i,4)]=min(ASRG);
        [r_minD(i,5),r_minD(i,6)]=min(ASA);
    end
    
    corresponding = find(max(r_minD(:,[1,3,5]),[],2) < d_corr);
    for rep = 1:repeat
%         corresponding = corresponding(randperm(length(corresponding),N_sample));
        
        ind_corr = [corresponding,r_minD(corresponding,[2,4,6])];
 
        AM_AS_corr_D{num} = ((r_AS(ind_corr(:,1),1)-r_AS(ind_corr(:,1),1)').^2+(r_AS(ind_corr(:,1),2)-r_AS(ind_corr(:,1),2)').^2+(r_AS(ind_corr(:,1),3)-r_AS(ind_corr(:,1),3)').^2).^0.5;
        [AM_AS_corr{num},r_AS_corr{num}] = determine_connectivity(AM_AS,r_AS,ind_corr(:,1));
        [AM_JC_corr{num},r_JC_corr{num}] = determine_connectivity(AM_JC,r_JC,ind_corr(:,2));
        [AM_RG_corr{num},r_RG_corr{num}] = determine_connectivity(AM_RG,r_RG,ind_corr(:,3));
        [AM_A_corr{num},r_A_corr{num}] = determine_connectivity(AM_A,r_A,ind_corr(:,4));
        
        AM_AS_corr_D{num} = ((r_AS(ind_corr(:,1),1)-r_AS(ind_corr(:,1),1)').^2+(r_AS(ind_corr(:,1),2)-r_AS(ind_corr(:,1),2)').^2+(r_AS(ind_corr(:,1),3)-r_AS(ind_corr(:,1),3)').^2).^0.5;
        [AM_AS_corr{num},r_AS_corr{num}] = determine_connectivity(AM_AS,r_AS,ind_corr(:,1));
        [AM_JC_corr{num},r_JC_corr{num}] = determine_connectivity(AM_JC,r_JC,ind_corr(:,2));
        [AM_RG_corr{num},r_RG_corr{num}] = determine_connectivity(AM_RG,r_RG,ind_corr(:,3));
        [AM_A_corr{num},r_A_corr{num}] = determine_connectivity(AM_A,r_A,ind_corr(:,4));
             
        count = count + 1;
        FR_JC(:,:,count) = FPFN_statistics(AM_AS_corr{num},AM_JC_corr{num},AM_AS_corr_D{num},bin);
        FR_RG(:,:,count) = FPFN_statistics(AM_AS_corr{num},AM_RG_corr{num},AM_AS_corr_D{num},bin);
        FR_A(:,:,count) = FPFN_statistics(AM_AS_corr{num},AM_A_corr{num},AM_AS_corr_D{num},bin);
    end
end

% figure,
% plot_StandardError(squeeze(FR_JC(:,1,:))','b')
% hold on
% plot_StandardError(squeeze(FR_JC(:,2,:))','r')
% title 'FP-FN-JC'
% legend('FDR','FNR','Location','northwest')
% ylim([0,1])
% 
% figure,
% plot_StandardError(squeeze(FR_RG(:,1,:))','b')
% hold on
% plot_StandardError(squeeze(FR_RG(:,2,:))','r')
% title 'FP-FN-RG'
% legend('FDR','FNR','Location','northwest')
% ylim([0,1])
% 
% figure,
% plot_StandardError(squeeze(FR_A(:,1,:))','b')
% hold on
% plot_StandardError(squeeze(FR_A(:,2,:))','r')
% title 'FP-FN-onlyAuto'
% legend('FDR','FNR','Location','northwest')
% ylim([0,1])
% 
% %%% f1
% figure,
% plot_StandardError(squeeze(FR_JC(:,3,:))','b')
% hold on
% plot_StandardError(squeeze(FR_RG(:,3,:))','r')
% hold on
% plot_StandardError(squeeze(FR_A(:,3,:))','g')
% title 'F1-Score'
% legend('JC','RG','A','Location','southwest')
% ylim([0,1])

% dissertation plot
FR_M = cat(3,FR_JC,FR_RG);
figure,
plot_StandardError(squeeze(FR_M(:,1,:))','g')
hold on
plot_StandardError(squeeze(FR_M(:,2,:))','k')
hold on
plot_StandardError(squeeze(FR_A(:,1,:))','b')
hold on
plot_StandardError(squeeze(FR_A(:,2,:))','r')
title 'FP-FN'
legend('M_FP','M_FN','A_FP','A_FN','Location','northwest')
ylim([0,1])
axis square
end
% function [AM_corr,r_corr] = determine_connectivity(AM,r,ind)
% AM_corr = zeros(length(ind));
% for i = 1:length(ind)
% %     [~,~,L1]=find(AM(:,ind(i)),1,'first');
% L1 = max(AM(:,ind(i)));
%     for j = i+1:length(ind)[i,j]
% %         [~,~,L2]=find(AM(:,ind(j)),1,'first');
% L2 = max(AM(:,ind(j)));
%         AM_corr(i,j) = L1==L2; %sum(intersect(AM(:,ind(i)),AM(ind(j),:)))>0;
%     end
% end
% AM_corr = AM_corr + AM_corr';
% r_corr = r(ind,:);
% end

function [AM_corr,r_corr] = determine_connectivity(AM,r,ind)
L = max(AM);
L1 = repmat(L(ind),length(ind),1);
L2 = repmat(L(ind)',1,length(ind));
AM_corr = L1==L2;
r_corr = r(ind,:);
end

function [FR] = FPFN_statistics(AM_g,AM_t,AM_D,bin)
FR = zeros(length(bin)-1,2);
for i = 1:length(bin)-1
    ind = find(AM_D>bin(i)&AM_D<bin(i+1));
    AM_gg = AM_g(ind);
    AM_tt = AM_t(ind);
    FR(i,1) = sum((AM_gg==0).*(AM_tt==1))/(sum((AM_gg==0).*(AM_tt==1))+sum((AM_gg==1).*(AM_tt==1)));
    FR(i,2) = sum((AM_gg==1).*(AM_tt==0))/(sum((AM_gg==1).*(AM_tt==0))+sum((AM_gg==1).*(AM_tt==1)));
    FR(i,3) = 2*sum((AM_gg==1).*(AM_tt==1))/(sum((AM_gg==0).*(AM_tt==1))+sum((AM_gg==1).*(AM_tt==0))+2*sum((AM_gg==1).*(AM_tt==1)));
end
end

function [] = plot_StandardError(matrix,co)
% matrix = matrix(:,[1,5,2,3,4,6,7,8]); % to change the order of models
[row, ~] = find(isnan(matrix)|isinf(matrix));
urow = unique(row);
matrix(urow,:)=[];
errorbar(1:size(matrix,2),mean(matrix),std(matrix,[],1)/sqrt(size(matrix,1)),'.-','color',co)
axis square, xlim([0,size(matrix,2)+1])
end