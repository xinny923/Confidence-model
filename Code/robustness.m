clear all; 

load("x_robust.csv");
load("x.mat");

% load("x_robust_self.csv");
% load("selfx.mat");

dif = [];
for i=1:8
    dif(:,i)=x_robust(:,i)-x(i);
end
dif = abs(dif);
mean_dif = mean(dif,1);
per_dif = mean_dif./x;