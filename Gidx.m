clear all;
clc;

load Accident_Singapore.mat;

ind1=find(gnd==0);% ID of non-accident
ind2=find(gnd==1);% ID of accident

n1=length(ind1);
n2=length(ind2);

% 50% training and 50% testing
p1=ceil(n1/20);
p2=ceil(n2/20);

trainIdx=[];
testIdx=[];


a=randperm(n1);
trpart=ind1(a(1:p1));
tepart=ind1(a(p1+1:end));
trainIdx=[trainIdx,trpart];
testIdx=[testIdx,tepart];

trpart=[];tepart=[];
b=randperm(n2);
trpart=ind2(b(1:p2));
tepart=ind2(b(p2+1:end));
trainIdx=[trainIdx;trpart];
testIdx=[testIdx;tepart];



trainIdx=trainIdx';
testIdx=testIdx';
save('20.mat','trainIdx','testIdx');
