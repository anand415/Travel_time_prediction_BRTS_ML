close all;
clear;
load test.mat
mdls={'tree','rf','bg','xgb','lgb','cbr'};
actR=reshape(act,5,6)';
prdR=reshape(prd,5,6)';
trrindR=reshape(trrind,5,6)';
tsstindR=reshape(tsstind,5,6)';

acttr=actR{1,:};

for ii=1:6
    tmpmat=[];
   for jj=1:5
       tmpmat=[tmpmat,actR{ii,jj}];
   end
act_all(ii,:)=double(tmpmat);
end


for ii=1:6
    tmpmat=[];
   for jj=1:5
       tmpmat=[tmpmat,prdR{ii,jj}];
   end
pred_all(ii,:)=tmpmat;
end


metrics_test=[];
% metrics(1,:)=['R2' 'MSE' 'MAE'];
for i=1:1:size(act_all,1)
temp=corrcoef(act_all(i,:),pred_all(i,:));
R2(i,1)=temp(1,2);
temp=mean((act_all(i,:)-pred_all(i,:)).^2);
MSE(i,1)=temp;
temp=mean(abs(act_all(i,:)-pred_all(i,:)));
MAE(i,1)=temp;
temp=sum((act_all(i,:)-pred_all(i,:)));
bias(i,1)=temp;
variance(i,1)=(MSE(i,1)-(bias(i,1).^2));

% metric(i+1,:)=[R2(i,1),MSE(i,1),MAE(i,1)];
end
metrics_test=[R2 MSE MAE bias variance];