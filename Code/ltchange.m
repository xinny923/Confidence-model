function ltchange()

x = [];
y = [];
for i = 1:2
    load(strcat('C_data',num2str(i),'.mat'));
%     ind = randi([1,50],1,40); %pick random # of data points (to see how many participants are needed)
%     C_data = C_data(ind,:);
    all_C_data{i} = C_data;
    meanC = mean(C_data,1);
    sem = std(C_data,1)./sqrt(size(C_data,1));
    
    % -----Linear regression to find slope for each participant--------
    for j = 1:size(C_data,1) 
        x1 = [ones(21,1) (0:20)'];
        y1 = C_data(j,1:21)';
        coeff1(j,:) = x1\y1;
        x2 = [ones(11,1) (20:30)'];
        y2 = C_data(j,21:end)';
        coeff2(j,:) = x2\y2;
    end
    
    % Save all coefficient values in fit1 and fit2
    fit1(i,:,:) = coeff1;
    fit2(i,:,:) = coeff2;
    
    max(coeff1)
    min(coeff1)
    max(coeff2)
    min(coeff2)
    
    % Mean of coefficient values
    meanCoeff1 = mean(coeff1,1); %from fit1
    meanCoeff2 = mean(coeff2,1); %from fit2
    
    % -----Linear regression to find slope for averaged data------------
    % Find R-squared value of the regression 
    x1_mean = (0:20)';
    y1_mean = mean(C_data(:,1:21),1);
    mdl1 = fitlm(x1_mean,y1_mean);
    r1 = mdl1.Rsquared.Ordinary;
    x2_mean = (20:30)';
    y2_mean = mean(C_data(:,21:end),1);
    mdl2 = fitlm(x2_mean,y2_mean);
    r2 = mdl2.Rsquared.Ordinary;
    
    p1 = coefTest(mdl1);
    p2 = coefTest(mdl2);
    
    % aoctool
    x = [x [x1_mean; x2_mean]]; % puzzle numbers for two slopes data points
    y = [y [y1_mean y2_mean]']; % confidence values for two slopes data points
%     mdlnum = [ones(21,1); ones(11,1).*2];
%     [h,atab,ctab,stats] = aoctool(x,y,mdlnum,0.05,'','','','off','separate lines');
    
    % --------------------------Plot------------------------------------
    figure;
    hold on;
    p1 = errorbar(0:30,meanC,sem,'k.','LineWidth',1.5,'MarkerSize',25);
    x1 = linspace(0,20,100);
    x2 = linspace(20,30,100);
    if i==1
        p2 = plot(x1,meanCoeff1(1)+x1*meanCoeff1(2),'Color',[0,0.4,0.7],'LineWidth',3.5);
        p3 = plot(x2,meanCoeff2(1)+x2*meanCoeff2(2),'Color',[0,0.4,0.7],'LineWidth',3.5);
    else
        p2 = plot(x1,meanCoeff1(1)+x1*meanCoeff1(2),'Color',[0,0.4,0.7],'LineWidth',3.5);
        p3 = plot(x2,meanCoeff2(1)+x2*meanCoeff2(2),'Color',[0,0.4,0.7],'LineWidth',3.5);
    end
%     p2.Color(4) = 0.5;
%     p3.Color(4) = 0.5;
    
    % When AI accuracy changes
    p4 = plot(ones(1,11).*20,0:0.1:1,'Color',[1.0,0.4,0],'LineWidth',1);
    
    v1 = [0 0;0 0.1;20 0.1;20 0];
    v2 = [20 0; 20 0.1;30 0.1; 30 0];
    f1 = [1 2 3 4];
    if i==1
        patch('Faces',f1,'Vertices',v2,'FaceColor','red','FaceAlpha',.2,'LineStyle','None');
        patch('Faces',f1,'Vertices',v1,'FaceColor','blue','FaceAlpha',.2,'LineStyle','None');
    else
        patch('Faces',f1,'Vertices',v1,'FaceColor','red','FaceAlpha',.2,'LineStyle','None');
        patch('Faces',f1,'Vertices',v2,'FaceColor','blue','FaceAlpha',.2,'LineStyle','None');
    end
    
    % When the AI gives answer not according to its accuracy 
%     notacc={[4,9,10,15],[24,29]};
%     for j = notacc{i}
%         p5 = plot(ones(1,11).*j-1,0:0.1:1,'--m');
%     end
%     for j = notacc{-i+3}
%         p6 = plot(ones(1,11).*j-1,0:0.1:1,'--g');
%     end
%     if i == 1
%         a1 = annotation('textarrow',[0.42,0.4],[0.47,meanC(11)+.14],'String',{strcat('Slope= ',num2str(round(meanCoeff1(2),6))),strcat('R^2= ', num2str(round(r1,6)))});
%         a2 = annotation('textarrow',[0.75,0.77],[0.38,meanC(26)+0.05],'String',{strcat('Slope= ',num2str(round(meanCoeff2(2),4))),strcat('R^2= ', num2str(round(r2,3)))});
%     else   
% %         a1 = annotation('textarrow',[0.47,0.4],[0.42,meanC(11)],'String',{strcat('Slope= ',num2str(meanCoeff1(2))),strcat('R^2= ', num2str(r1))});
% %         a2 = annotation('textarrow',[0.75,0.4],[0.38,meanC(26)+.14],'String',{strcat('Slope= ',num2str(meanCoeff2(2))),strcat('R^2= ', num2str(r2))});
%     end
%     a1.Color = [0,0.4,0.7];
%     a1.FontSize = 14;
%     a1.FontName = 'Helvetica';
%     a2.Color = [0,0.4,0.7];
%     a2.FontSize = 14;
%     a2.FontName = 'Helvetica';
        

    % Plot format
%     condname1 = {'High-performing', 'Low-performing'};
%     condname2 = {'Low-performing', 'High-performing'};
%     title(strcat('Condition',{' '},num2str(i),' :',{' '},condname1{i},' to',{' '},condname2{i},' AI'),'FontSize',30);
%     legend([p1,p2,p3,p4,p5(1),p6(1)],'Data',strcat('Slope1=',num2str(meanCoeff1(2)),' (R^2=',...
%         num2str(r1),')'),strcat('Slope2=',num2str(meanCoeff2(2)),' (R^2=',...
%         num2str(r2),')'),'Performance change','Unexpected poor AI suggestions','Unexpected good AI suggestions','FontSize',15);
    legend([p1,p4],'Data','AI performance change','FontSize',14);
    xlabel('Puzzle number, n','FontSize',15,'FontWeight','bold');
    ylabel('Confidence in AI','FontSize',15,'FontWeight','bold');
    axis([0 30 0 1]);
    set(gca,'FontSize',15)
    set(gca,'FontName','Helvetica')
    box on;
    grid on;
    
end

%-------------------------Statistical Tests------------------------------
% s11, s12, s21, s22  
% slCond1 = [fit1(1,:,2)' fit2(1,:,2)'];
% meanSlCond1 = mean(slCond1,1);
% % figure;
% % normplot(slCond1(:,1));
% % figure;
% % normplot(slCond1(:,2));
% slCond2 = [fit1(2,:,2)' fit2(2,:,2)'];
% meanSlCond2 = mean(slCond2,1);
% % figure;
% % normplot(slCond2(:,1));
% % figure;
% % normplot(slCond2(:,2));
% [s11_p,s11_h]=signtest(slCond1(:,1));
% [s12_p,s12_h]=signtest(slCond1(:,2));
% [s21_p,s21_h]=signtest(slCond2(:,1));
% [s22_p,s22_h]=signtest(slCond2(:,2));

slope = [ones(21,1); zeros(11,1)]; 
% s11 vs. s12 
X = [slope x(:,1)]; % Predictors
Y = y(:,1);
mdl = fitlm(X,Y,'interactions','CategoricalVars',1,'Varnames',{'Slope','Puzzle','Confidence'});
% s21 vs. s22 
X = [slope x(:,2)]; % Predictors
Y = y(:,2);
mdl = fitlm(X,Y,'interactions','CategoricalVars',1,'Varnames',{'Slope','Puzzle','Confidence'});

% s11 vs. s21 and s11 vs. -s21
condition = [ones(21,1); zeros(21,1)]; 
X = [condition [x(1:21,1); x(1:21,2)]];
Y = [y(1:21,1); y(1:21,2)];
mdl = fitlm(X,Y,'interactions','CategoricalVars',1,'Varnames',{'Condition','Puzzle','Confidence'});
Y = [y(1:21,1); -y(1:21,2)];
mdl = fitlm(X,Y,'interactions','CategoricalVars',1,'Varnames',{'Condition','Puzzle','Confidence'});
% s12 vs. s22 and -s12 vs. s22
condition = [ones(11,1); zeros(11,1)];
X = [condition [x(22:end,1); x(22:end,2)]];
Y = [y(22:end,1); y(22:end,2)];
mdl = fitlm(X,Y,'interactions','CategoricalVars',1,'Varnames',{'Condition','Puzzle','Confidence'});
Y = [-y(22:end,1); y(22:end,2)];
mdl = fitlm(X,Y,'interactions','CategoricalVars',1,'Varnames',{'Condition','Puzzle','Confidence'});

% % Start,Mid,Final confidence
% [svsm1_p,svsm1_h]=signtest(all_C_data{1}(:,1),all_C_data{1}(:,21));
% [mvsf1_p,mvsf1_h]=signtest(all_C_data{1}(:,21),all_C_data{1}(:,31));
% [svsm2_p,svsm2_h]=signtest(all_C_data{2}(:,1),all_C_data{2}(:,21));
% [mvsf2_p,mvsf2_h]=signtest(all_C_data{2}(:,21),all_C_data{2}(:,31));
% [sm1_p,sm1_h]=signtest((all_C_data{1}(:,21)-all_C_data{1}(:,1))./20);
% [sm2_p,sm2_h]=signtest((all_C_data{2}(:,21)-all_C_data{2}(:,1))./20);
% [mf1_p,mf1_h]=signtest((all_C_data{1}(:,31)-all_C_data{1}(:,21))./10);
% [mf2_p,mf2_h]=signtest((all_C_data{2}(:,31)-all_C_data{2}(:,21))./10);
% 
% [sm_p,sm_h] = ranksum((all_C_data{1}(:,21)-all_C_data{1}(:,1))./20, ...
%     (all_C_data{2}(:,21)-all_C_data{2}(:,1))./20);
% [mf_p,mf_h] = ranksum((all_C_data{1}(:,31)-all_C_data{1}(:,21))./10, ...
%     (all_C_data{2}(:,31)-all_C_data{2}(:,21))./10);
% 
% [smsize_p,smsize_h] = ranksum((all_C_data{1}(:,21)-all_C_data{1}(:,1))./20, ...
%     (all_C_data{2}(:,1)-all_C_data{2}(:,21))./20);
% [mfsize_p,mfsize_h] = ranksum((all_C_data{1}(:,21)-all_C_data{1}(:,31))./10, ...
%     (all_C_data{2}(:,31)-all_C_data{2}(:,21))./10);



end

