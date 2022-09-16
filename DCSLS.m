function [IX,B] = DCSLS(X,Y, W,Wr,cost_setting,lamda)
%	Usage:
%	[Y] = LaplacianScore(X, W)
%
%	X: Rows of vectors of data points
%	W: The affinity matrix.
%	Y: Vector of (1-LaplacianScore) for each feature.
%      The features with larger y are more important.
%

[nSmp,nFea]=size(X);
nClass=length(unique(Y));
u=mean(X,1);
D = full(sum(W,2));
D = sparse(1:nSmp,1:nSmp,D,nSmp,nSmp);

L=zeros(1,nFea);
C=zeros(nSmp,nSmp);
f=[];
for i=1:nSmp
    for j=1:nSmp
        if Y(i)~=Y(j)
            if Y(i)==1&&Y(j)==-1
                C(i,j)=cost_setting.C_OI;
            else if Y(j)==-1&&Y(i)==1
                    C(i,j)=cost_setting.C_IO;
                end
            end
        end
    end
    if Y(i)==1%»Î«÷’ﬂ
        f(i)=cost_setting.C_OI;
    else
        f(i)=cost_setting.C_IO;
    end
end

f=f./mean(f);
Sr=C.*Wr;
DD=diag(f).*D;
onevec=ones(nSmp,1);
WW=diag(f)*W;
DW = full(sum(WW,2));
DW = sparse(1:nSmp,1:nSmp,DW,nSmp,nSmp);

for r=1:nFea
    ur=mean(X(:,r));
    L(r)=X(:,r)'*((1-lamda)*2*(DW-WW)+lamda*(2*Sr-diag(sum(Sr,1))-diag(sum(Sr,2))))*X(:,r);
    fr=X(:,r)-(X(:,r)'*DD*onevec)/(onevec'*DD*onevec)*onevec;
    var=fr'*DD*fr/(onevec'*DD*onevec);
    if var< 1e-12
        var= 10000;
    end
    L(r)=L(r)/var;
end


[B,IX]=sort(L);