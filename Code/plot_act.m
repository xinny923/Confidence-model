function plot_act(cond)

load(strcat('act_data',num2str(cond),'.mat'));

% ------------------------Action plot--------------------------------
% Calculate mean of data
meanAct = mean(act_data,1);

% Calculate standard error of the mean
semAct = std(act_data,1)./sqrt(size(act_data,1));

% Plot data
p1 = errorbar(0:30,meanAct,semAct);
hold on;

% When the AI gives answer not according to its accuracy 
%  i = [4,10,14,18,23];
notacc={[4,9,10,15],[24,29]};
for j = notacc{cond}
    p2 = plot(ones(1,11).*j-1,0:0.1:1,'--m','LineWidth',1.5);
end
for j = notacc{-cond+3}
    p3 = plot(ones(1,11).*j-1,0:0.1:1,'--g','LineWidth',1.5);
end

% When AI accuracy changes
p4 = plot(ones(1,11).*20,0:0.1:1,'r');
%----------------------------------------------------------------------

% Plot format
title(strcat('Cond',num2str(cond),' Action Plot'));
legend([p1,p2(1),p3(1),p4], 'Action','Bad AI suggestion','Good AI suggestion','Accuracy change');
xlabel('t or Puzzle Number');
ylabel('Probability of following AI suggestion');
axis([0 30 0 1]);


end