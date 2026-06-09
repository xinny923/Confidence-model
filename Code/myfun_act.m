function F = myfun_act(x)
% x = [s1,s2,b]

% -----Change these accordingly-----
numPuz = 30;

load('act_data.mat');
load('skill_data.mat');
load('fitC1.mat');
fitC = C;
load('fitC2.mat');
fitC = [fitC; C];
load('fitselfC1.mat');
fitselfC = C;
load('fitselfC2.mat'); 
fitselfC = [fitselfC; C];
% ----------------------------------
% Normalize
mi = min(skill_data);
ma = max(skill_data);
skill_data = (skill_data-min(mi))/(max(ma)-min(mi));
% ----------------------------------

fitact = [];
for i=2:numPuz+1
    fitact = [fitact 100./(1+exp(x(1).*(fitC(:,i)-fitselfC(:,i)) + x(2).*skill_data' + x(3)))];
%     fitact = [fitact x(1).*(fitC(:,i)-fitselfC(:,i)) + x(2).*skill_data' + x(3)];
end
% % Normalize
% mi = min(fitact);
% ma = max(fitact);
% fitact = (fitact-min(mi))./(max(ma)-min(mi));

F = act_data(:,2:end)'-fitact';
F = reshape(F,[],1);

end

