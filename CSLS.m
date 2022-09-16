function [IX,B] = CSLS(X,Y, W,cost_setting,lamda)
%	Usage:
%	[Y] = LaplacianScore(X, W)


[nSmp,nFea]=size(X);
nClass=length(unique(Y));
u=mean(X,1);
D = full(sum(W,2));
% D = sparse(1:nSmp,1:nSmp,D,nSmp,nSmp);
L=zeros(1,nFea);
C=zeros(nSmp,nSmp);
f=[];
for i=1:nSmp
    for j=1:nSmp
        if Y(i)~=Y(j)
            if Y(i)==1&&Y(j)==-1
                C(i,j)=cost_setting.C_OI;
            else if Y(i)==-1&&Y(j)==1
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
S=C.*W;
DD=f.*D';
for r=1:nFea
    ur=mean(X(:,r));
    L(r)=X(:,r)'*(2*S-diag(sum(S,1))-diag(sum(S,2))-lamda*diag(DD))*X(:,r)+lamda*(2*X(:,r)'*DD'*ur-sum(DD)*ur^2);
end

[B,IX]=sort(L);