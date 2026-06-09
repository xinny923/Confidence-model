function performance()

load('perf_data.mat');
load('C_data.mat');
load('selfC_data.mat');

meanPerf = mean(perf_data,1);
meanC = mean(C_data,1);
meanselfC = mean(selfC_data,1);

% perfC = [meanPerf(2:end)' meanC(2:end)'];
% perfself = [meanPerf(2:end)' meanselfC(2:end)'];
% 
% [r_perfC,p_perfC] = corrcoef(perfC);
% [r_perfself,p_perfself] = corrcoef(perfself);

resp = perf_data(:,2:end)';
resp = resp(:);

selfC = selfC_data(:,2:end)';
selfC = selfC(:);

pred1_2 = [];
for i=1:100
    pred1_2 = [pred1_2; i.*ones(30,1)];
end
pred1_2 = categorical(pred1_2);

aoctool(resp,selfC,pred1_2)

for i=1:2
    load(strcat('perf_data',num2str(i),'.mat'));
    
    for j=1:50
        % Normalize perf data per participant
        perf_data_norm(j,:) = (perf_data(j,:)-min(perf_data(j,:)))/(max(perf_data(j,:))-min(perf_data(j,:)));
%         ind = find(any(allC_data_norm,2)==0); % Find rows with one value (can not normalize)
%         allC_data_norm(ind,:)=allC_data(ind,:); % Replace those rows with original data
    end
    
    meanPerf = mean(perf_data_norm,1);
    semPerf = std(perf_data_norm,1)./sqrt(size(perf_data_norm,1));
    
    figure;
    hold on;
    meanPerf = meanPerf(2:end);
    semPerf = semPerf(2:end);
    p1 = errorbar(0:29,meanPerf,semPerf);
    % When AI accuracy changes
    p2 = plot(ones(1,11).*20,0:0.1:1,'r');
    
    % Self Confidence
    load(strcat('selfC_data',num2str(i),'.mat'));
    meanC = mean(selfC_data,1);
    semC = std(selfC_data,1)./sqrt(size(selfC_data,1));
    p3 = errorbar(0:30,meanC,semC);
    
    % Plot format
    title(strcat('Performance & Self Confidence (Cond ',num2str(i),')'));
    legend([p1,p2,p3],'Performance','Accuracy change','Self Confidence');
    xlabel('t or Puzzle Number');
    ylabel('Performance');
    axis([0 30 0 1]);  
    
end

end