debugging = 0;
trace_cell = [];
CVCV = [];
LapLap = [];
Lap_cell = cell(1,1);
I_cell = cell(1,1);
AveAve = [];
SVSV = [];
II = [];
RaRa = [];
NTSR_ = [];
IJIJ = [];
IJIJ_d = [];
dn = 'OP'; % name of the dataset
finename = [dn,'_image1_sample1000times_dim100_Dissertation_v2'];

switch dn
    case 'L1'
        n = 6; % number of images in the data
    case 'OP'
        n = 1; % for now only OP1 is available for testing
end

tic
for test_on = 1:n
    c=1;
    %     cost_f = @(x) 1./min(1,x+10^-2); % cost function for A*
    cost_f = @(x) c + 1 - x; % cost function for A*
    pad = 1;
    pad2 = 0;
    N_sets = 10;
    Nsegs = 5;
    ppm = 1;
    Junction_dist = 10;
    Trace_Sample_dist = 5;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    switch dn
        
        case 'L1'
            source_file_list = {
                ['../../data/Training_result/CC_L1_singleIM_5_/output',num2str(test_on),'_',num2str(20000000)],
                ['../../data/Training_result/CTC_L1_singleIM_4/output',num2str(test_on),'_',num2str(7400000)],
                ['../../data/Training_result/3DUNet_L1_singleIM_aaa4_c/output',num2str(test_on),'_',num2str(1881000)],
                ['../../data/Training_result/3DUNet_L1_singleIM_aaa4_c/output',num2str(test_on),'_',num2str(50000)],
                ['../../data/Training_result/3DUNet_L1_singleIM_aaa4_c/output',num2str(test_on),'_',num2str(10000)],
                %         ['../../data/Training_result/RP_Max1Min1_ImfImo_15_1_20_uint8/',num2str(test_on),'_AS.mat']
                ['../../data/Training_result/',num2str(test_on),'_L6_AS_opt_log_212'],
                ['../../data/Training_result/',num2str(test_on),'_L6_AS_opt_SF_median_333'],
                ['../../data/Training_result/',num2str(test_on),'_L6_AS_opt_Mean_Shift_default'],
                %         ['../../data/Training_result/L1_5_OtsuSke']
                };
            
            label_source = read_multipage_tif(['../../data/L1/label_',num2str(test_on),'.tif']);
            
            trace_source=load(['../../data/L1/',num2str(test_on),'_L6_AS_opt.mat']);
            
        case 'OP'
            source_file_list = {
                ['../../data/Training_result/CC_OP_singleIM_102/output',num2str(test_on),'_',num2str(3600000)],
                ['../../data/Training_result/CTC_OP_singleIM_102/output',num2str(test_on),'_',num2str(2730000)],
                ['../../data/Training_result/3DUNet_OP_singleIM_5/output',num2str(test_on),'_',num2str(2311000)],
                %             ['../../data/Training_result/RP_OP_Max1Min1_ImfImo_15_1/',num2str(test_on),'_AS.mat'],
                ['../../data/Training_result/',num2str(test_on),'_OP_AS_opt2_log_212'],
                ['../../data/Training_result/',num2str(test_on),'_OP_AS_opt2_SF_median_333'],
                ['../../data/Training_result/',num2str(test_on),'_OP_AS_opt2_Mean_Shift_default'],
                %         ['../../data/Training_result/OP1_OtsuSke']
                };           
            
            label_source = read_multipage_tif(['../../data/OP/label_',num2str(test_on),'.tif']);
            
            trace_source=load(['../../data/OP/',num2str(test_on),'_OP_AS_opt2.mat']);
            
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    label = double(double(label_source) > 254);
    label = padarray(label,[pad,pad,pad],-1);
    AM = trace_source.AM;
    r = trace_source.r;
    R = trace_source.R;
    
    image=cell(1,1);   
    
    for i = 1:length(source_file_list)
        image_source=load(source_file_list{i});
        IM_ = double(image_source.IM)/255;
        image{i+1} = padarray(IM_,[pad,pad,pad],-1);
    end
    
    Original_source = load(source_file_list{1});
    Original = double(Original_source.Original)/255;
    Original = padarray(Original,[pad,pad,pad],-1);
    image{1} = Original;
    
    r = r + pad;
       
    addpath('NeuronTracerV20')
    AMlbl = LabelBranchesAM(AM);
    labels=unique(AMlbl(AMlbl>0));
    count=1;
    r_start=zeros(N_sets,3);
    r_end=zeros(N_sets,3);
    I=cell(1,length(image));
    Lap=cell(1,length(image));
    CV=cell(1,length(image));
    Ave=cell(1,length(image));
    searched_volumn=cell(1,length(image));
    Ra=cell(1,length(image));
    
    while count<N_sets

        [e1,e2]=find(AMlbl==labels(randi(length(labels))));
        ee=unique([e1;e2]);
        AM_temp=AM(ee,ee);
        r_temp=r(ee,:);
        [AM_temp,r_temp,~] = AdjustPPM(AM_temp,r_temp,R,ppm);
        if size(r_temp,1)>Nsegs
            Start=randi(size(r_temp,1)-Nsegs);
            End=Start+Nsegs;
            l_end=sum((r_temp(Start,:)-r_temp(End,:)).^2)^0.5;
            if l_end>0.8*Nsegs/ppm 
                
                r_start(count,:)=r_temp(Start,:);
                r_end(count,:)=r_temp(End,:);
                [~,trace,~,~] = get_I_trace(label,round(r_temp(Start,:)),round(r_temp(End,:)),cost_f,c);
                if mean(image{1}(trace)) < 100/255

                    
                    close all
                    count=count+1;
                    
                    
                    
                    if debugging == 1
                        figure
                    end
                    for i = 1:length(image)
                        trace_cell{count,i} = trace;
                        I_trace = image{i}(trace);
                        I_cell{count,i} = I_trace;
                        Lap_trace = 6*image{i}(trace)-(image{i}(trace+1)+image{i}(trace-1)+image{i}(trace+size(image{i},1))+image{i}(trace-size(image{i},1))+image{i}(trace+size(image{i},1)*size(image{i},2))+image{i}(trace-size(image{i},1)*size(image{i},2)));
                        Lap_cell{count,i} = Lap_trace;
                        Lap{i}=[Lap{i};std(Lap_trace)/mean(Lap_trace)];
                        I{i} = [I{i};I_trace];
                        CV{i}=[CV{i};std(I_trace)/mean(I_trace)];
                        Ave{i}=[Ave{i};mean(I_trace)];
                        [x1,y1,z1] = ind2sub(size(image{i}),trace);
                        R_ = Optimize_R(image{i},sparse(diag(ones(1,length(x1)-1),1)),[x1,y1,z1],R,1,100,1,1,1);
                        Ra{i}=[Ra{i};mean(R_)];
                        if debugging == 1
                            subplot(2,4,i),imshow(max(Image_small,[],3));hold on
                            plot3(y1,x1,z1,'r.')
                            plot3(round(r_start(2)-MIN(2)+1+pad),round(r_start(1)-MIN(1)+1+pad),round(r_start(3)-MIN(3)+1+pad),'g*')
                            plot3(round(r_end(2)-MIN(2)+1+pad),round(r_end(1)-MIN(1)+1+pad),round(r_end(3)-MIN(3)+1+pad),'r*')
                        end
                    end
                    if debugging == 1
                        temp=cell2mat(CV); temp(end,:)
                        aveI_trace
                    end
                end
            end
        end
    end
    
    CVt = [];
    for i = 1:length(image)
        CVt = [CVt,CV{i}];
    end
    CVCV = [CVCV;CVt];
    
    Lapt = [];
    for i = 1:length(image)
        Lapt = [Lapt,Lap{i}];
    end
    LapLap = [LapLap;Lapt];
    
    Avet = [];
    for i = 1:length(image)
        Avet = [Avet,Ave{i}];
    end
    AveAve = [AveAve;Avet];
    
    SVt = [];
    for i = 1:length(image)
        SVt = [SVt,searched_volumn{i}];
    end
    SVSV = [SVSV;SVt];
    
    It = [];
    for i = 1:length(image)
        It = [It,I{i}];
    end
    II = [II;It];
    
    Rat = [];
    for i = 1:length(image)
        Rat = [Rat,Ra{i}];
    end
    RaRa = [RaRa;Rat];
    
    O = Original(2:end-1,2:end-1,2:end-1);
    s_x = 64;
    s_y = 64;
    s_z = 10;
    NTSR = cell(1,length(image));
    for i = 1:s_x:512-(s_x-1)
        for j = 1:s_y:512-(s_y-1)
            for k = 1:s_z:size(O,3)-(s_z-1)
                fore = label_source(i:i+s_x-1,j:j+s_y-1,k:k+s_z-1)>254;
                back = label_source(i:i+s_x-1,j:j+s_y-1,k:k+s_z-1)<1;
                if sum(fore(:)) > 10 && sum(back(:))
                    for ind = 1:length(image)
                        foregrounds = image{ind}(1+i:i+s_x,1+j:j+s_y,1+k:k+s_z).*fore;
                        backgrounds = image{ind}(1+i:i+s_x,1+j:j+s_y,1+k:k+s_z).*back;
                        mfI = sum(foregrounds(:))/sum(fore(:));
                        mbI = sum(backgrounds(:))/sum(back(:));
                        n_to_s = mbI/mfI;
                        NTSR{ind} = [NTSR{ind};n_to_s];
                    end
                end
            end
        end
    end
    
    NTSRt = [];
    for i = 1:length(image)
        NTSRt = [NTSRt,NTSR{i}];
    end
    NTSR_ = [NTSR_;NTSRt];
    
    [J1,J2,Start1,End1,Start2,End2] = Find_Junction_Seeds(AMlbl,r,Junction_dist,Trace_Sample_dist);
    I_J=cell(1,length(image));
    I_J_d=cell(1,length(image));
    trace0 = cell(1,size(J1,1));
    trace1 = cell(1,size(J1,1));
    trace2 = cell(1,size(J1,1));
    for j = 1:size(J1,1)
        [~,trace0_temp,~,~] = get_I_trace(image{1},round(J1(j,:)),round(J2(j,:)),cost_f,c);
        trace0{j} = trace0_temp;
        [~,trace1_temp,~,~] = get_I_trace(label,round(Start1(j,:)),round(End1(j,:)),cost_f,c);
        trace1{j} = trace1_temp;
        [~,trace2_temp,~,~] = get_I_trace(label,round(Start2(j,:)),round(End2(j,:)),cost_f,c);
        trace2{j} = trace2_temp;
    end
    for i = 1:length(image)
        
        figure,PlotAM(AMlbl,r)
        hold on
        for j = 1:size(J1,1)
            disp([i,j])
            
            I_trace_J = image{i}(trace0{j});
            trace_J = trace0{j};
            show_trace(image{i},pad2,round(J1(j,:)),round(J2(j,:)),trace_J,'r')
            
            I_trace_J1 = image{i}(trace1{j});
            trace_J1 = trace1{j};
            show_trace(image{i},pad2,round(Start1(j,:)),round(End1(j,:)),trace_J1,'g')
            
            I_trace_J2 = image{i}(trace2{j});
            trace_J2 = trace2{j};
            show_trace(image{i},pad2,round(Start2(j,:)),round(End2(j,:)),trace_J2,'b')
            
            I_J{i}=[I_J{i};2*mean(I_trace_J)/(mean(I_trace_J1)+mean(I_trace_J2))];
            length(I_trace_J),mean(I_trace_J)
            length(I_trace_J1),mean(I_trace_J1)
            length(I_trace_J2),mean(I_trace_J2)
            
            dxyz = round(J1(j,:))-round(J2(j,:));
            I_J_d{i}=[I_J_d{i};sqrt(sum(dxyz.^2))];
        end
    end
    
    
    IJIJt = [];
    for i = 1:length(image)
        IJIJt = [IJIJt,I_J{i}];
    end
    IJIJ = [IJIJ;IJIJt];
    
    IJIJ_dt = [];
    for i = 1:length(image)
        IJIJ_dt = [IJIJ_dt,I_J_d{i}];
    end
    IJIJ_d = [IJIJ_d;IJIJ_dt];
    
end
toc

save(['../../data/Training_result/',finename],'trace_cell','CVCV','I_cell','LapLap','Lap_cell','SVSV','AveAve','II','RaRa','NTSR_','IJIJ','IJIJ_d')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [I_trace,trace,cost_so_far,came_from] = get_I_trace(A,sp,gp,cost_f,c)

[trace, I_trace, cost_so_far,came_from] = I_Path(cost_f,c,A,sp(1),sp(2),sp(3),gp(1),gp(2),gp(3));

end

function [trace, I_trace, cost_so_far,came_from] = I_Path(cost_f,c,M_temp,x_s,y_s,z_s,x_g,y_g,z_g)

start = sub2ind(size(M_temp),x_s,y_s,z_s);
goal = sub2ind(size(M_temp),x_g,y_g,z_g);

[cost_so_far,came_from]= PathFinding(x_s,y_s,z_s,x_g,y_g,z_g,cost_f,c,M_temp);


here = goal;
trace = here;
I_trace = [];
while here ~= start
    here = came_from(here);
    trace = [trace;here];
    I_trace = [I_trace;M_temp(here)];
end
end

function [cost_so_far,came_from,front]= PathFinding(x_s,y_s,z_s,x_g,y_g,z_g,cost_f,c,M)
start = sub2ind(size(M),x_s,y_s,z_s);
goal = sub2ind(size(M),x_g,y_g,z_g);
cost_so_far=-ones(size(M));
came_from = -ones(size(M));
priority = inf(size(M));
sizeIm = size(M);
front = start;
cost_so_far(start)= 0;
N6_ind=[-1;+1;-sizeIm(1);+sizeIm(1);-sizeIm(1)*sizeIm(2);+sizeIm(1)*sizeIm(2)];

while ~isempty(front) && ~any(front == goal)
    [~,I] = min(priority(front));
    next=[front(I)+N6_ind(1),front(I)+N6_ind(2),front(I)+N6_ind(3), ...
        front(I)+N6_ind(4),front(I)+N6_ind(5),front(I)+N6_ind(6)];
    next = next(M(next)~=-1); %!!!!!!!!!!!
    new_cost = cost_so_far(front(I))*ones(size(next)) + 1/2*(cost_f(M(front(I)))*ones(size(next))+cost_f(M(next)));
    ind = (came_from(next) == -1) | (new_cost < cost_so_far(next));
    next = next(ind);
    new_cost = new_cost(ind);
    cost_so_far(next) = new_cost;
    priority(next) = new_cost + c*heuristic(M, x_g,y_g,z_g, next);
    front = [front,next];
    came_from(next) = front(I);
    front(I) = [];
end
end

function [FinalImage] = read_multipage_tif(FileTif)
InfoImage=imfinfo(FileTif);
mImage=InfoImage(1).Width;
nImage=InfoImage(1).Height;
NumberImages=length(InfoImage);

FinalImage=zeros(nImage,mImage,NumberImages,'uint8');
for i=1:NumberImages
    FinalImage(:,:,i)=imread(FileTif,i);
end
end

function [heu_dist] = heuristic(M, x_g,y_g,z_g, next)
[x_b,y_b,z_b] = ind2sub_AS(size(M),next);
%     heu_dist = abs(x_g-x_b)+abs(y_g-y_b)+abs(z_g-z_b); % manhattan_dist
heu_dist = sqrt((x_g-x_b).^2+(y_g-y_b).^2+(z_g-z_b).^2); %     euclidean_dist
end

function [] = show_trace(M,pad,start,goal,trace_inds,usecolor) % plot the image as well as start and goal
plot3(start(2)-pad,start(1)-pad,start(3)-pad,'s','markers',16,'LineWidth',3,'color',usecolor)
hold on
plot3(goal(2)-pad,goal(1)-pad,goal(3)-pad,'x','markers',12,'LineWidth',3,'color',usecolor)
hold on
[x_trace,y_trace,z_trace] = ind2sub(size(M),trace_inds);
plot3(y_trace-pad,x_trace-pad,z_trace-pad,'.','markers',5,'color',usecolor)
hold on
end