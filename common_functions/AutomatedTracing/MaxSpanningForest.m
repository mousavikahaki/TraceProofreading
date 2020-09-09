% This function finds a degree constrained maximum spanning forest
% The forest need not span all vartices in the graph
% The sum of edge weights is maximized (Wmax). The weights can be negative.
% Node degrees are either fixed (e.g. 1,2,3) or not (NaN)
% This is a modification of Kruskal's algorithm
% W is a sparse symmetric matrix of edge weights
% D is a vector of node degrees
% AM is the resulting adjacency matrix

function [AMlbl,Wmax] = MaxSpanningForest(W,D)

%{
addpath('C:\Armen\DIADEM\Neuron Tracer V20')
NN=100000;
W=sparse(randi(NN,3*NN,1),randi(NN,3*NN,1),rand(3*NN,1),NN,NN);
W=W-diag(diag(W)); W=W+W'; 
D=nan(1,NN);
%}

N=length(D);
Tree_ind=(1:N);
W=triu(W);

ind=find(W>0);
[~,ind_sort]=sort(W(ind),'descend');
ind_sort=ind(ind_sort);
[ii,jj]=ind2sub([N,N],ind_sort);

count=0;
ii_keep=zeros(size(ind));
jj_keep=zeros(size(ind));
for i=1:length(ind)
    if Tree_ind(ii(i))~=Tree_ind(jj(i)) % if there are no loops
        count=count+1;
        ii_keep(count)=ii(i);
        jj_keep(count)=jj(i);
        Tree_ind(Tree_ind==Tree_ind(jj(i)))=Tree_ind(ii(i));
    end
end
ii_keep=ii_keep(1:count);
jj_keep=jj_keep(1:count);
AM=sparse(ii_keep,jj_keep,ones(count,1),N,N);
AMlbl=LabelTreesAM(AM);
Wmax=sum(W(AM>0));

%{
% check the code by considering all possible networks
bin_list=dec2bin(0:2^length(ind)-1,length(ind));
bin_list=(double(bin_list)-48)>0;
for i=1:size(bin_list,1)
    temp=bin_list(i,:);
    ind_temp=ind(temp);
    Wtemp=sum(W(ind_temp));
    if Wtemp>Wmax
        [ii_temp,jj_temp]=ind2sub([N,N],ind_temp);
        AM_temp=sparse(ii_temp,jj_temp,ones(length(ind_temp),1),N,N);
        if ~isempty(Find_Loops(AM_temp))~=Loops(AM_temp)
            disp('Error in the Loops function')
        end
        if Loops(AM_temp)==0
            disp(Wtemp)
        end
    end
end
%}

%{
r=rand(N,3,111);
figure
plot3(r(:,2),r(:,1),r(:,3),'*')
hold on
PlotAM(AMlbl,r)

figure
plot3(r(:,2),r(:,1),r(:,3),'*')
hold on
PlotAM(AM_temp,r)
%}
