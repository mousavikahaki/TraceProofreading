function [AMlbl2,SVr2] =AutomatedTracing(Original)
% Original=255-Original(:,:,1:400);
% Original=255-Original;

Original1 =double(Original)-0.9.*smooth3(Original,'gaussian',5);

% find seeds
Mesh=8;
thr=58;
sigma=[];
SVr=Find_Seeds(Original1,Mesh,thr,sigma);
figure
imshow(max(Original,[],3),[0,max(Original(:))])
hold on
plot3(SVr(:,2),SVr(:,1),SVr(:,3),'r*')

% cost_f=@(x) 1./x; 
% c = 1; % 1 for AStart, 0 for DJk
% dd = 15;
% [AM_cost, AM_dist, AM_C_path] = A_star_cost_DefensePPT(double(Original),SVr,cost_f,c,dd)

% find the weight matrix
Dist=((SVr(:,1)-SVr(:,1)').^2+(SVr(:,2)-SVr(:,2)').^2+(SVr(:,3)-SVr(:,3)').^2).^0.5;
ind=find(Dist<5*Mesh & Dist>0);
N=size(Dist,1);
[ii,jj]=ind2sub(N,ind);
% t=(0:0.01:1)';
% val=zeros(1,length(ind));
% for i=1:length(ind)
%     pix=t*(SVr(jj(i),:)-SVr(ii(i),:))+SVr(ii(i),:);
%     pix=unique(round(pix),'rows');
%     pix_ind=sub2ind(size(Original),pix(:,1),pix(:,2),pix(:,3));
%     val(i)=mean(Original(pix_ind));
% end
% W=sparse(ii,jj,val,N,N);
% Original1=255-Original1;
val=zeros(1,length(ind));
pad=[1,1,1];
for i=1:length(ind),i
    Max=max(SVr(ii(i),:),SVr(jj(i),:)); %+3; Max=min(Max,size(Original));
    Min=min(SVr(ii(i),:),SVr(jj(i),:)); %-3; Min=max(Min,[1,1,1]);
    Start=SVr(ii(i),:)-Min+1+pad;
    End=SVr(jj(i),:)-Min+1+pad;
    IM=Original1(Min(1):Max(1),Min(2):Max(2),Min(3):Max(3))+20;
    IM=double(IM);
    IM=padarray(IM,pad,-1,'both');
    figure, imshow(max(IM,[],3),[0,max(Original(:))])
    [trace, I_trace] = A_Star(IM,Start(1),Start(2),Start(3),End(1),End(2),End(3),1);
    
%     figure
%     imshow(max(IM,[],3),[0 255])
%     hold on
%     plot3(Start(:,2),Start(:,1),Start(:,3),'g*')
%     plot3(End(:,2),End(:,1),End(:,3),'r*')
%     [xx,yy,zz]=ind2sub(size(IM),trace);
%     plot3(yy,xx,zz,'c.')
    
    
    val(i)= mean(I_trace)./length(I_trace); %mean(I_trace)./std(I_trace)./length(I_trace); % 
end
W=sparse(ii,jj,val,N,N);
D=nan(1,size(W,1));
%W=p-0.5;

% generate the trace
[AMlbl,Wmax] = MaxSpanningForest(W,D);
[AMlbl1,SVr1,~] = Eliminate_Small_Trees(AMlbl,SVr,zeros(size(SVr,1)),100);
[AMlbl2,SVr2,~] = Eliminate_Terminal_Branches(AMlbl1,SVr1,10,1,1);
figure
imshow(max(Original,[],3))
hold on
PlotAM(AMlbl2,SVr2)
view(2)

function SVr=Find_Seeds(Original,Mesh,thr,sigma)

if ~isempty(sigma)
    Original=smooth3(Original,'gaussian',sigma);
end

sizeIm=size(Original);
if length(sizeIm)==2
    sizeIm=[sizeIm,1];
end

N=ceil(sizeIm(1)/Mesh)*ceil(sizeIm(2)/Mesh)*ceil(sizeIm(3)/Mesh);
SVx=zeros(N,1);
SVy=zeros(N,1);
SVz=zeros(N,1);

count=0;
[Max,ind]=max(Original(:));
while Max>=thr
    count=count+1;
    [SVx(count),SVy(count),SVz(count)]=ind2sub(sizeIm,ind);
    Original(max(SVx(count)-Mesh,1):min(SVx(count)+Mesh,sizeIm(1)),max(SVy(count)-Mesh,1):min(SVy(count)+Mesh,sizeIm(2)),max(SVz(count)-Mesh,1):min(SVz(count)+Mesh,sizeIm(3)))=0;
    [Max,ind]=max(Original(:)); 
    disp(double([count,Max]))
end

SVr=[SVx(1:count),SVy(1:count),SVz(1:count)];
