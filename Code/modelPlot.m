function [mse,rsq] = modelPlot(x)

ssres8 = 0;
sstot8 = 0;
ssres = 0;
sstot = 0;
res = 0;
fitted = 0;

% condname1 = {'High-performing', 'Low-performing'};
% condname2 = {'Low-performing', 'High-performing'};
for cond=1:2
    % -----Change these accordingly-----
    numPuz = 30;
    
    load(strcat('C_data',num2str(cond),'.mat'));
    load(strcat('selfC_data',num2str(cond),'.mat'));
    load(strcat('C0_data',num2str(cond),'.mat'));
    load(strcat('B0_data',num2str(cond),'.mat'));
    load(strcat('e_data',num2str(cond),'.mat'));
    load(strcat('datafile',num2str(cond),'.mat'));
    % ----------------------------------
    
    % Calculate mean of data
    meanC = mean(C_data,1);
    
    % Calculate standard error of the mean
    semC = std(C_data,1)./sqrt(size(C_data,1));

    % ------------------------Confidence plot--------------------------------
    % Plot data
    figure;
    hold on;
    p1 = errorbar(0:30,meanC,semC,'k.','LineWidth',1.5,'MarkerSize',25);
    
    % ------------------------ 8 DOF Fit--------------------------------
    x1 = [];
    for i=1:30
        x1 = [x1 ones(1,50).*i];
    end
    C_data_new = C_data(:,2:end);
    p = polyfit(x1,C_data_new(:)',8);
    est = polyval(p,1:30);
    ssres8 = ssres8 + sum((est-meanC(2:end)).^2);
    sstot8 = sstot8 + sum((meanC(2:end)-mean(meanC(2:end))).^2);
    p6 = plot(est);
   
    %----------------------------------------------------------------------
    
    % When the AI gives answer not according to its accuracy
%     notacc={[4,9,10,15],[24,29]};
%     for j = notacc{cond}
%         p2 = plot(ones(1,11).*j-1,0:0.1:1,'--m','LineWidth',1);
%     end
%     for j = notacc{-cond+3}
%         p3 = plot(ones(1,11).*j-1,0:0.1:1,'--g','LineWidth',1);
%     end
    
    % When AI accuracy changes
    p4 = plot(ones(1,11).*20,0:0.1:1,'Color',[1.0,0.4,0],'LineWidth',1);
    
    % -----------------Plot fitted C from model---------------------------
    b = B0_data';
    c = C_data(:,1);
    C = c;
    for i=1:numPuz
        if i==1
            a = C_data(:,i);
        else
            a = x(8).*C(:,i-1)+(1-x(8)).*a;
        end
        e = x(4).*e_data(:,i,1) + x(5).*e_data(:,i,2) + x(6).*e_data(:,i,3) + x(7).*e_data(:,i,4);
        c = c + x(1).*(e-c) + x(2).*(a-c) + x(3).*(b-c);
        C = [C c];
    end
    
    fitMean = mean(C,1);
    p5 = plot(0:30,fitMean,'Color',[0,0.4,0.7],'LineWidth',2);
    
    v1 = [0 0;0 0.1;20 0.1;20 0];
    v2 = [20 0; 20 0.1;30 0.1; 30 0];
    f1 = [1 2 3 4];
    if cond==1
        patch('Faces',f1,'Vertices',v2,'FaceColor','red','FaceAlpha',.2,'LineStyle','None');
        patch('Faces',f1,'Vertices',v1,'FaceColor','blue','FaceAlpha',.2,'LineStyle','None');
    else
        patch('Faces',f1,'Vertices',v1,'FaceColor','red','FaceAlpha',.2,'LineStyle','None');
        patch('Faces',f1,'Vertices',v2,'FaceColor','blue','FaceAlpha',.2,'LineStyle','None');
    end
    
%     fitCI1 = sum(C,1)./47.5;
%     fitCI2 = sum(C,1)./52.5;
%     p7 = fill([0:30,fliplr(0:30)],[fitCI1,fliplr(fitCI2)],1,'facecolor', 'b', 'edgecolor', 'none', 'facealpha', 0.2);
%     
%     save(strcat('fitC',num2str(cond),'.mat'),'C');
    %------------------------------------------------------------------------
    
    % Plot format
    xlabel('Puzzle number, n','FontSize',15,'FontWeight','bold');
    axis([0 30 0 1]);
    ylabel('Confidence in AI','FontSize',15,'FontWeight','bold');
%     title(strcat('Condition',{' '},num2str(cond),' :',{' '},condname1{cond},' to',{' '},condname2{cond},' AI'),'FontSize',30);
%     legend([p1,p5,p4,p2(1),p3(1)], 'Data','Model fit',...
%         'Performance change','Unexpected poor AI suggestions','Unexpected good AI suggestions','FontSize',15);
    legend('Data','AI performance change','Model fit');
    set(gca,'FontSize',15)
    set(gca,'FontName','Helvetica')
    box on;
    grid on;
    hold off;
    
    %--------------------Model Accuracy Calulation-----------------------------
    ssres = ssres + sum((fitMean(2:end)-meanC(2:end)).^2);
    sstot = sstot + sum((meanC(2:end)-mean(meanC(2:end))).^2);
%     if cond==1
%         mse1 = ssres/30;
%         rsq1 = 1-ssres/sstot;
%     else
%         mse2 = sum((fitMean(2:end)-meanC(2:end)).^2)/30;
%         rsq2 = 1-sum((fitMean(2:end)-meanC(2:end)).^2)/sum((meanC(2:end)-mean(meanC(2:end))).^2);
%     end
    res = [res fitMean(2:end)-meanC(2:end)];
    fitted = [fitted fitMean(2:end)];

end

mse8 = ssres8/60;
rsq8 = 1-ssres8/sstot8;
rsq8_adj = 1-(99/92)*(ssres8/sstot8);
mse = ssres/60;
rsq = 1-ssres/sstot;
rsq_adj = 1-(99/92)*(ssres/sstot);

%-------------------------- Residual plot -----------------------------
figure;
plot(fitted,res,'.');
hold on;
plot(linspace(0.35,0.7),zeros(100,1));
axis([0.35 0.7 -0.15 0.15])
xlabel('Fitted Data');
ylabel('Residual');
title('Residual Plot - Confidence for AI');

end

