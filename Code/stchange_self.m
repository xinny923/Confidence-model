function stchange_self()

for i = 1:2
    load(strcat('selfC_data',num2str(i),'.mat'));
    C_data_all(i,:,:) = selfC_data;
    meanC = mean(selfC_data,1);
    sem = std(selfC_data,1)./sqrt(size(selfC_data,1));
    
    % ----------------------Plot-----------------------------------------
    figure 
    hold on;
    p1 = errorbar(0:30,meanC,sem,'ko');
    
    % When AI accuracy changes
    p2 = plot(ones(1,11).*20,0:0.1:1,'r');
    
    % When the AI gives answer not according to its accuracy 
    notacc={[4,9,10,15],[24,29]};
    for j = notacc{i}
        p3 = plot(ones(1,11).*j-1,0:0.1:1,'--m');
    end
    for j = notacc{-i+3}
        p4 = plot(ones(1,11).*j-1,0:0.1:1,'--g');
    end
    
    % Plot format
    title(strcat('Cond ',num2str(i),' Self-Confidence'));
    legend([p1 p2 p3(1) p4(1)],'Data','Accuracy change',...
        'Unexpected bad AI','Unexpected good AI');
    xlabel('t or Puzzle Number');
    ylabel('Self-Confidence');
    axis([0 30 0 1]);
    
end

% ----------------------Reorganize data---------------------------------
% Change in confidence after a good AI suggestion OR a bad AI suggestion

% General
badchange = [];
goodchange = [];
notacc2=[4,9,10,15,21,22,23,25,26,27,28,30];
for i=1:2
    for j=1:30
        if ismember(j,notacc2)==2-i
            badchange = [badchange C_data_all(i,:,j+1)-C_data_all(i,:,j)];
        else
            goodchange = [goodchange C_data_all(i,:,j+1)-C_data_all(i,:,j)];
        end
    end
end
badchange1 = badchange(1:12*50);
badchange2 = badchange(12*50+1:end);
goodchange1 = goodchange(1:18*50);
goodchange2 = goodchange(18*50+1:end);

% Expected & Unexpected
% badEx = [badchange(4*50+1:12*50) badchange(12*50+1:28*50)];
% badUnex = [badchange(1:4*50) badchange(28*50+1:end)];
% goodEx = [goodchange(1:16*50) goodchange(22*50+1:end)];
% goodUnex = [goodchange(18*50+1:22*50) goodchange(16*50+1:18*50)];
badEx1 = badchange1(4*50+1:end); % Cond 2 after 
badUnex1 = badchange1(1:4*50); % Cond 1 before
badEx2 = badchange2(1:16*50); % Cond 2 before
badUnex2 = badchange2(16*50+1:end); % Cond 2 after
goodEx1 = goodchange1(1:16*50); % Cond 1 before
goodUnex1 = goodchange1(16*50+1:end); % Cond 1 after
goodEx2 = goodchange2(4*50+1:end); % Cond 2 after
goodUnex2 = goodchange2(1:4*50); % Cond 2 before

% Immediately after AI accuracy change 
% afterch1 = badEx1(1:50);
% afterch2 = goodEx2(1:50);

% --------------------Statistical Tests---------------------------------
% All B's and G's, both conditions, both expected/unexpectd, and both before/after 
% Cond 1
[goodex1_p,goodex1_h] = signtest(goodEx1); % Before
[badunex1_p,badunex1_h] = signtest(badUnex1);
[badex1_p,badex1_h] = signtest(badEx1); % After
[goodunex1_p,goodunex1_h] = signtest(goodUnex1);
% Cond2 
[badex2_p,badex2_h] = signtest(badEx2); % Before
[goodunex2_p,goodunex2_h] = signtest(goodUnex2);
[goodex2_p,goodex2_h] = signtest(goodEx2); % After
[badunex2_p,badunex2_h] = signtest(badUnex2);

% % Cond 1 vs. Cond 2 (independent variables)
% [goodex_p,goodex_h] = ranksum(goodEx1, goodEx2);
% [goodunex_p,goodunex_h] = ranksum(goodUnex1, goodUnex2);
% [badex_p,badex_h] = ranksum(badEx1, badEx2);
% [badunex_p,badunex_h] = ranksum(badUnex1, badUnex2);

% Expected vs. Unexpected 
[goodcond1_p,goodcond1_h] = ranksum(goodEx1, goodUnex1);
[badcond1_p,badcond1_h] = ranksum(badEx1, badUnex1);
[goodcond2_p,goodcond2_h] = ranksum(goodEx2, goodUnex2);
[badcond2_p,badcond2_h] = ranksum(badEx2, badUnex2);

% Good vs. Bad
[ex1_p,ex1_h] = ranksum(goodEx1, -badEx1);
[ex2_p,ex2_h] = ranksum(goodEx2, -badEx2);
[unex1_p,unex1_h] = ranksum(goodUnex1, -badUnex1);
[unex2_p,unex2_h] = ranksum(goodUnex2, -badUnex2);


% % General (-badchange vs. goodchange)
% % figure;
% % normplot(-badchange);
% % histogram(-badchange);
% % figure;
% % normplot(goodchange);
% % histogram(goodchange);
% % [h,p] = lillietest(-badchange); % not normal
% % [h,p] = lillietest(goodchange); % not normal
% [badp,badh]=signtest(badchange);
% [goodp,goodh]=signtest(goodchange);
% [change_p,change_h]=signtest(change(1,:),change(2,:));
% 
% % Compare the change in confidence after a good AI suggestion before or
% % after the accuracy change ([goodchange(1:16*50) goodchange(18*50+1:22*50)] 
% % vs. [goodchange(16*50+1:18*50) goodchange(22*50+1:end)])
% gchange = {[goodchange(1:16*50) goodchange(18*50+1:22*50)], ...
%     [goodchange(16*50+1:18*50) goodchange(22*50+1:end)]};
% meanGchange = [mean(gchange{1}) mean(gchange{2})];
% [gchange_p,gchange_h]=ranksum(gchange{1},gchange{2});
% 
% % Compare the change in confidence after a bad AI suggestion before or
% % after the accuracy change ([badchange(1:4*50) badchange(12*50+1:28*50)] 
% % vs. [badchange(4*50+1:12*50) badchange(28*50+1:end)])
% bchange = {[badchange(1:4*50) badchange(12*50+1:28*50)], ...
%     [badchange(4*50+1:12*50) badchange(28*50+1:end)]};
% meanBchange = [mean(bchange{1}) mean(bchange{2})];
% [bchange_p,bchange_h]=ranksum(bchange{1},bchange{2});
% 
% 
% % Compare the magnitude of the change in confidence (-gbchange vs.
% % bgchange)
% % figure;
% % normplot(-gbchange);
% % histogram(-gbchange);
% % figure;
% % normplot(bgchange);
% % histogram(bgchange);
% % [h,p] = lillietest(-gbchange); % not normal
% % [h,p] = lillietest(bgchange); % not normal
% fewchange = [-gbchange; bgchange];
% meanFewchange = mean(fewchange,2);
% [gbp,gbh]=signtest(gbchange);
% [bgp,bgh]=signtest(bgchange);
% [fewchange_p,fewchange_h]=signtest(fewchange(1,:),fewchange(2,:));
% 
% % Compare the change in confidence before and after the accuracy change 
% %(good to bad)
% gbChange = {gbchange(1:200),gbchange(201:300)};
% meanGbChange = [mean(gbChange{1}), mean(gbChange{2})];
% [gbChange_p,gbChange_h]=ranksum(gbChange{1},gbChange{2});
% 
% % % Compare the change in confidence before and after the accuracy change (bad to good)
% % bgChange = {bgchange(1:100),bgchange(101:300)};
% % meanBgChange = [mean(bgChange{1}), mean(bgChange{2})];
% % [bgChange_p,bgChange_h]=ranksum(bgChange{1},bgChange{2});
% 
% 
% % Compare change in confidence after an expected vs. unexpected AI
% % suggestion
% bchangeEx = {[badchange(4*50+1:12*50) badchange(12*50+1:28*50)],...
%     [badchange(1:4*50) badchange(28*50+1:end)]};
% meanBchangeEx = [mean(bchangeEx{1}) mean(bchangeEx{2})];
% gchangeEx = {[goodchange(1:16*50) goodchange(22*50+1:end)],...
%     [goodchange(18*50+1:22*50) goodchange(16*50+1:18*50)]};
% meanGchangeEx = [mean(gchangeEx{1}) mean(gchangeEx{2})];
% [gchangeEx_p,gchangeEx_h]=ranksum(gchangeEx{1},gchangeEx{2});
% [bchangeEx_p,bchangeEx_h]=ranksum(bchangeEx{1},bchangeEx{2});

end