function f = Generate_New_Features(r,n,AM)

N=size(r,1);
AM=AM+AM';
A=AM;
for i=2:N-1
    A=A+AM^i;
end
A=A>0;
A=A-diag(diag(A));

r0=mean(r,1);
n0=mean(n,1); 
n0(3)=0; % no z component
n0=n0./(sum(n0.^2)).^0.5;

% PCA on r_xy to find another n0
r=r-ones(N,1)*r0;
[eig_vect,~]=eig(r(:,1:2)'*r(:,1:2));
n1=[eig_vect(:,2)',0];

% create translationally and rotationally (in xy only) invariant
% arguments for the features (rho, cos(phi), z, cos(theta), nz)
m=7; %number of invariant feature arguments
a=zeros(N,m);
a(:,1)=(r(:,1).^2+r(:,2).^2).^0.5;
a(:,2)=(r*n0')./a(:,1);
a(:,3)=abs((r*n1')./a(:,1));
a(:,4)=r(:,3);
a(:,5)=(n*n0')./(n(:,1).^2+n(:,2).^2).^0.5;
a(:,6)=abs((n*n1')./(n(:,1).^2+n(:,2).^2).^0.5);
a(:,7)=n(:,3);

% we use A, I, B, and b 
I=diag(ones(N,1));
B=ones(N,N); B=B-diag(diag(B));
b=ones(N,1);

% 3 scalar features of 0th order in a
f(1)=b'*A*b./N^2;
f(2)=b'*I*b; % cluster size. All other features are normalized
f(3)=b'*B*b./N^2;

% 3*m scalar features of 1st order in a
f(3+(1:m))=b'*A*a./N^2;
f(3+m+(1:m))=b'*I*a./N; %f(14) is always 0 
f(3+2*m+(1:m))=b'*B*a./N^2; %f(21) is always 0 

% 3*m*(m+1)/2 scalar features of 2d order in a
ind=triu(ones(m,m))>0;
q=m*(m+1)/2;
temp=a'*A*a./N^2;
f(3+3*m+(1:q))=temp(ind);
temp=a'*I*a./N;
f(3+3*m+q+(1:q))=temp(ind); %f(54) is always 0 
temp=a'*B*a./N^2;
f(3+3*m+2*q+(1:q))=temp(ind);