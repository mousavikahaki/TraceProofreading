%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this function generates N uniformly distributed seed points
% Mesh is the minimum distance between the seeds
% thr is the threshold
% sigma is a bluring parameter

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
