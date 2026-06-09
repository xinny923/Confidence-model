function [mse,rsq]=modelPlot_act(x,cond)

% -----Change these accordingly-----
numPuz = 30;

load(strcat('act_data',num2str(cond),'.mat'));
load(strcat('skill_data.mat'));
load('fitC1.mat');
fitC = C;
load('fitC2.mat');
fitC = [fitC; C];
load('fitselfC1.mat');
fitselfC = C;
load('fitselfC2.mat'); 
fitselfC = [fitselfC; C];
% ----------------------------------

% Calculate mean of data
meanAct = mean(act_data,1);

% Calculate standard error of the mean
semAct = std(act_data,1)./sqrt(size(act_data,1));

% ------------------------Action plot--------------------------------
% Plot data
p1 = errorbar(0:30,meanAct,semAct);
hold on;
% p1 = errorbar(0:30,meanSelfC,semSelfC);

%----------------------------------------------------------------------

% When the AI gives answer not according to its accuracy 
notacc={[4,9,10,15],[24,29]};
for j = notacc{cond}
    p2 = plot(ones(1,11).*j-1,0:0.1:1,'--m','LineWidth',1.5);
end
for j = notacc{-cond+3}
    p3 = plot(ones(1,11).*j-1,0:0.1:1,'--g','LineWidth',1.5);
end

% When AI accuracy changes
p4 = plot(ones(1,11).*20,0:0.1:1,'r');

% -----------------Plot fitted probability from model---------------------------
fitact = [];
for i=2:numPuz+1
    fitact = [fitact 100./(1+exp(x(1).*(fitC(:,i)-fitselfC(:,i)) + x(2).*skill_data' + x(3)))];
end
% % Normalize
% mi = min(fitact);
% ma = max(fitact);
% fitact = (fitact-min(mi))/(max(ma)-min(mi));

fitMean = mean(fitact,1);
p5 = plot(fitMean);
%------------------------------------------------------------------------

% Plot format
xlabel('t or Puzzle Number');
axis([0 30 0 1]);
ylabel('Reliance on AI');
title(strcat('Cond',num2str(cond),' Reliance Plot'));
legend([p1,p2(1),p3(1),p4,p5], 'Reliance on AI','Bad AI suggestion',...
    'Good AI suggestion','Accuracy change','Model Prediction');



%--------------------Model Accuracy Calulation-----------------------------
ssres = sum((fitMean-meanAct(2:end)).^2);
sstot = sum((meanAct(2:end)-mean(meanAct(2:end))).^2);
mse = ssres/30;
rsq = 1-ssres/sstot;


end