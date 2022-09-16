function [IX,B] = CSFS(X,Y,h1,h2)
%	Usage:
%	[Y] = FisherScore(X, W)

[nSmp,nFea]=size(X);
nClass=length(unique(Y));
ind1=find(Y==-1);% ID of non-accident
ind2=find(Y==1);% ID of accident
L=zeros(1,nFea);
n1=length(ind1);
n2=length(ind2);


for r=1:nFea
    u1=mean(X(ind1,r));
    u2=mean(X(ind2,r));
    u=mean(X(:,r));
    cov1=std(X(ind1,r));
    cov2=std(X(ind2,r));
    L(1,r)=(h1*n1*(u1-u)^2+h2*n2*(u2-u)^2)/(cov1^2*n1*h1+cov2^2*n2*h2);
end


[B,IX]=sort(L,'descend');

