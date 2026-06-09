clear all;

% % Overall 
% figure;
% x = categorical({'Confidence for AI', 'Self-Confidence'});
% coeff = [-0.0539,0.3294,0.1495; -0.8643,-1.1197,-1.0014];
% bar(x,coeff);
% title('Regression Coefficient - Probability of Accepting AI Suggestions');
% legend('Condition 1', 'Condition2', 'All');
% ylabel('Regression Coefficient');
% figure;
% x = categorical({'Condition 1', 'Condition 2','All'});
% x = reordercats(x,{'Condition 1', 'Condition 2','All'});
% coeff = [-0.8643,-1.1197,-1.0014];
% b = bar(x,coeff,0.7);
% for i=1:length(coeff)
%     text(i,coeff(i)-0.1,num2str(coeff(i)),'HorizontalAlignment','center',...
%         'VerticalAlignment','bottom');
% end
% title('Self-Confidence vs. Probability of Accepting AI Suggestions');
% ylim([-1.5 1]);
% ylabel('Regression Coefficient');
% b.FaceColor = 'flat';
% b.CData(2,:) = [0.8 0.3 0];
% b.CData(3,:) = [0.9 0.8 0];
% hold off;


% By skill level or score 
figure;
x = categorical({'Low (Poor)', 'Mid (Fair)','High (Good)'});
x = reordercats(x,{'Low (Poor)', 'Mid (Fair)','High (Good)'});
% coeff = [-1.3637,-0.8131,1.7819]; % score - selfC
% p_val = [0.0895,0.0038,0.0087];
coeff = [-0.0737,0.0820,-0.6911]; % score - C
p_val = [0.9198,0.7467,0.1962];

b = bar(x,coeff,0.7);

for i=1:length(coeff)
    if coeff(i)<0 
        text(i,coeff(i)-0.1,num2str(coeff(i)),'HorizontalAlignment','center',...
            'VerticalAlignment','bottom','FontSize',14);
        text(i,coeff(i)-0.2,strcat('(p=',num2str(p_val(i)),')'),'HorizontalAlignment','center',...
            'VerticalAlignment','bottom','FontSize',14);
    else
        text(i,coeff(i)+0.1,num2str(coeff(i)),'HorizontalAlignment','center',...
            'VerticalAlignment','top','FontSize',14);
        text(i,coeff(i)+0.1,strcat('(p=',num2str(p_val(i)),')'),'HorizontalAlignment','center',...
            'VerticalAlignment','bottom','FontSize',14);
    end
end
% title('Correlation Coefficient vs. Team Performance score','FontSize',30);
% ylim([-2 2.5]);
ylim([-1 1]);
xlabel('Team performance score','FontSize',15,'FontWeight','bold');
ylabel('Regression Coefficient','FontSize',15,'FontWeight','bold');
b.FaceColor = 'flat';
b.CData(1,:)=[1.0,0.4,0];
b.CData(2,:)=[1,0.8,0];
b.CData(3,:)=[0,0.4,0.7];
set(gca,'FontSize',15)
set(gca,'FontName','Helvetica')
hold off;
grid on;
box on;

% coeff = {[-0.1832,-0.7512,0.2332; -0.6042,2.4131,-1.5098],...
%     [-0.1304,-0.3841,0.1246; -0.7792,0.5300,-0.6554],...
%     [0.5800,0.1327,0.3236; -1.8366,-1.8296,-1.8117]}; % skill
% coeff = {[-1.3407,-1.8418,-0.0737; 0.1491,3.1381,-1.3637],...
%     [-0.0432,-0.2180,0.0820; -0.7798,0.8456,-0.8131],...
%     [-1.1413,0.4669,-0.6911; 2.4540,-1.4697,1.7819]}; % score
% figure;
% x = categorical({'Confidence for AI', 'Self-Confidence'});
% for i=1:3
%     subplot(3,1,i)
%     h = bar(x,coeff{i});
%     titleopt = {'Low', 'Mid', 'High'};
%     title(strcat(titleopt(i),' Skill - Regression Coefficients - Probability of Accepting AI Suggestions'));
%     legend('Condition 1', 'Condition2', 'All');
%     ylabel('Regression Coefficient');
%     hold on;
% end
% coeff = [-0.6042,2.4131,-1.5098; -0.7792,0.5300,-0.6554; -1.8366,-1.8296,-1.8117]; % skill
% % coeff = [0.1491,3.1381,-1.3637; -0.7798,0.8456,-0.8131; 2.4540,-1.4697,1.7819]; % score
% figure;
% x = categorical({'Low','Mid','High'});
% x = reordercats(x,{'Low','Mid','High'});
% bar(x,coeff,0.9);
% a=0.75;
% for i=1:size(coeff,1)
%     for j=1:size(coeff,2)
%         if coeff(i,j)<0
%             text(a,coeff(i,j)-0.2,num2str(coeff(i,j)),'HorizontalAlignment','center',...
%                 'VerticalAlignment','bottom');
%         else
%             text(a,coeff(i,j)+0.2,num2str(coeff(i,j)),'HorizontalAlignment','center',...
%                 'VerticalAlignment','top');
%         end
%         a = a+0.25;
%     end
%     a = a+0.25;
% end
% title('Self-Confidence vs. Probability of Accepting AI Suggestions');
% ylim([-2 3.5]);
% ylabel('Regression Coefficient');
% legend('Condition 1', 'Condition2', 'All');
% 

% Predictor variable 1 - Subjects
subjects = [];
for i=1:100
    subjects = [subjects; i.*ones(30,1)];
end
S = categorical(subjects);
% S_reorder = reordercats(S,{'50';'1';'2';'3';'4';'5';'6';'7';'8';'9';'10';...
%     '11';'12';'13';'14';'15';'16';'17';'18';'19';'20';'21';'22';'23';'24';...
%     '25';'26';'27';'28';'29';'30';'31';'32';'33';'34';'35';'36';'37';'38';...
%     '39';'40';'41';'42';'43';'44';'45';'46';'47';'48';'49';'51';'52';'53';...
%     '54';'55';'56';'57';'58';'59';'60';'61';'62';'63';'64';'65';'66';'67';...
%     '68';'69';'70';'71';'72';'73';'74';'75';'76';'77';'78';'79';'80';'81';...
%     '82';'83';'84';'85';'86';'87';'88';'89';'90';'91';'92';'93';'94';'95';...
%     '96';'97';'98';'99';'100'});
dummyS = dummyvar(S);

% % Predictor variable 2 - Confidence for AI
% cond = [ones(30*50,1); 2.*zeros(30*50,1)];
% dummyCond = dummyvar(categorical(cond));

% Predictor variable 3 - Confidence for AI
load('C_data.mat');
C_data = C_data(:,1:30)';
meanC = mean(C_data,2);
C = C_data(:);

% Predictor variable 4 - Self-Confidence
load('selfC_data.mat');
selfC_data = selfC_data(:,1:30)';
meanSC = mean(selfC_data,2);
SC = selfC_data(:);

% Outcome variable 1 - Decision/Action
load('act_data.mat');
act_data = act_data(:,2:end)';
meanA = mean(act_data,2);
A = act_data(:);

% Outcome variable 2 - Performance
load('perf_data.mat');
perf_data = perf_data(:,1:30)';
% perf_data = perf_data(:,2:end)';
meanP = mean(perf_data,2);
P = perf_data(:);

% ----------------------- Regression overall -----------------------------
X = [dummyS(:,2:end) C SC];
% 
% Multinomial logistic regression
[mnrB,mnrdev,mnrstats] = mnrfit(X,categorical(~A));
[mnrB,mnrdev,mnrstats] = mnrfit(X,categorical(P));
% 
% % Generalized linear model regression
% [B,dev,stats] = glmfit(X,categorical(A),'binomial');
% [B,dev,stats] = glmfit(X,categorical(P),'binomial');
% 
% % Mdl = fitclinear(X,categorical(A),'Learner','logistic');

% ----------------------- Regression per condition -----------------------------
% X = [dummyS(1:30*50,2:50) C(1:30*50) SC(1:30*50)]; % Condition 1
% % X = [dummyS(30*50+2:end,52:100) C(30*50+2:end) SC(30*50+2:end)]; % Condition 2
% 
% % Multinomial logistic regression
% [mnrB,mnrdev,mnrstats] = mnrfit(X,categorical(~A(1:30*50)));
% [mnrB,mnrdev,mnrstats] = mnrfit(X,categorical(P(1:30*50)));
% % [mnrB,mnrdev,mnrstats] = mnrfit(X,categorical(~A(30*50+2:end)));
% % [mnrB,mnrdev,mnrstats] = mnrfit(X,categorical(P(30*50+2:end)));
% 
% % Generalized linear model regression
% [B,dev,stats] = glmfit(X,categorical(A(1:30*50)),'binomial');
% [B,dev,stats] = glmfit(X,categorical(P(1:30*50)),'binomial');
% % [B,dev,stats] = glmfit(X,categorical(A(30*50+2:end)),'binomial');
% % [B,dev,stats] = glmfit(X,categorical(P(30*50+2:end)),'binomial');
% 
% % Mdl = fitclinear(X,categorical(A(1:30*50)),'Learner','logistic');
% % Mdl = fitclinear(X,categorical(A(30*50+2:end)),'Learner','logistic');

% ----------------------Load skill groups------------------------------
% load('skillgroup1.mat') % Condition 1
load('scoregroup1.mat')
loind = group{1};
miind = group{2};
hiind = group{3};
% load('skillgroup2.mat') % Condition 2 
load('scoregroup2.mat')
loind = [loind 50+group{1}];
miind = [miind 50+group{2}];
hiind = [hiind 50+group{3}];
ind = {loind, miind, hiind};

% ------------------- Regression per skill group -----------------------------------
for k = 1:3
    k = ind{k};
    S = [];
    for i=k
        S = [S; i.*ones(30,1)];
    end
    dummyS = dummyvar(categorical(S));

    C = C_data(:,k);
    C = C(:);

    SC = selfC_data(:,k);
    SC = SC(:);

    A = act_data(:,k);
    A = A(:);

    P = perf_data(:,k);
    P = P(:);
    
    X = [dummyS(:,2:end) C SC]; % Change accordingly 
    
    if A(1)==1
        [mnrB,mnrdev,mnrstats] = mnrfit(X,categorical(A));
    else
        [mnrB,mnrdev,mnrstats] = mnrfit(X,categorical(~A));
    end
   [B,dev,stats] = glmfit(X,categorical(A),'binomial');
    
    if P(1)==5
        [mnrB,mnrdev,mnrstats] = mnrfit(X,categorical(P));
    else
        [mnrB,mnrdev,mnrstats] = mnrfit(X,categorical(-P));
    end
    [B,dev,stats] = glmfit(X,categorical(P),'binomial');
    
end
