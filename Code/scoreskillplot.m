load('skill_data.mat'); 
load('score_data.mat');
load('selfC_data.mat');


load('scoregroup1.mat');
ind = group;
load('scoregroup2.mat');

% figure;
% for j=1:3
%     plot(score_data(ind{j}),skill_data(ind{j}),'.','MarkerSize',30);
%     hold on;
%     plot(score_data(group{j}),skill_data(group{j}),'x','MarkerSize',20);
% end

for i=1:3
    ind{i} = [ind{i} 50+group{i}];
end

colors={[1.0,0.4,0],[1,0.8,0],[0,0.4,0.7]};
figure;
% plot(-150:150,-150:150,'k');
for j=1:3
    plot(skill_data(ind{j}),score_data(ind{j}),'.','MarkerSize',20,'Color',colors{j});
    hold on;
end
v1 = [-150 -150; 150 -150; 150 150];
v2 = [-150 -150; -150 150; 150 150];
f1 = [1 2 3];
patch('Faces',f1,'Vertices',v1,'FaceColor','red','FaceAlpha',.03,'LineStyle','None');
patch('Faces',f1,'Vertices',v2,'FaceColor','blue','FaceAlpha',.01,'LineStyle','None');
% title('Individual score vs. Team performance score','FontSize',30);
xlabel({'Individual performance score','(Human only)'},'FontSize',15,'FontWeight','bold');
ylabel({'Team performance score','(Human and AI)'},'FontSize',15,'FontWeight','bold');
legend('Poor decision makers','Fair decision makers','Good decision makers','FontSize',14);
set(gca,'FontSize',15)
set(gca,'FontName','Helvetica')
hold off;
grid on;
box on;

figure;
selfC_data = mean(selfC_data,2);
for j=1:3
%     x = repmat(score_data(ind{j}),31,1);
%     x = x(:);
%     y = selfC_data(ind{j},:)';
%     y = y(:);
    x = score_data(ind{j});
    y = selfC_data(ind{j});
    plot(x,y,'.','MarkerSize',20,'Color',colors{j});   
    hold on;
end
axis([-150 150 0 1]);
% title('Self-confidence vs. Team performance score','FontSize',30);
xlabel('Team performance score','FontSize',15,'FontWeight','bold');
ylabel('Self-confidence','FontSize',15,'FontWeight','bold');
legend('Poor decision maker','Fair decision maker','Good decision maker','FontSize',14);
set(gca,'FontSize',15)
set(gca,'FontName','Helvetica')
hold off;
grid on;
box on;


j = 1;

selfC_1_before = selfC_data(ind{j}(1:10),2:21); % good decision makers in Condition 1, before performance change
selfC_1_before = mean(selfC_1_before(:));
selfC_1_after = selfC_data(ind{j}(1:10),22:end); % good decision makers in Condition 1, after performance change
selfC_1_after = mean(selfC_1_after(:));

selfC_2_before = selfC_data(ind{j}(11:end),2:21); % good decision makers in Condition 2, before performance change
selfC_2_before = mean(selfC_2_before(:));
selfC_2_after = selfC_data(ind{j}(11:end),22:end); % good decision makers in Condition 2, after performance change
selfC_2_after = mean(selfC_2_after(:));

% figure;
% histogram(selfC_data(ind{3}(1:10),2:21),[-0.125 0.125:0.25:0.875 1.125],'FaceAlpha',0.5);
% hold on;
% histogram(selfC_data(ind{3}(1:10),22:end),[-0.125 0.125:0.25:0.875 1.125],'FaceAlpha',0.5);
% xticks([0 0.25 0.5 0.75 1])
% hold off; 
% 
% figure;
% histogram(selfC_data(ind{3}(11:end),2:21),[-0.125 0.125:0.25:0.875 1.125],'FaceAlpha',0.5);
% hold on;
% histogram(selfC_data(ind{3}(11:end),22:end),[-0.125 0.125:0.25:0.875 1.125],'FaceAlpha',0.5);
% xticks([0 0.25 0.5 0.75 1])
% hold off; 
% selfC_data = mean(selfC_data,2);
% x = repmat(1:31,10,1);
% x = x(:);
% y = selfC_data(ind{3}(1:10),:);
% y = y(:);
% plot(x,y,'.','MarkerSize',20);   
% hold on;
% x = repmat(1:31,11,1);
% x = x(:);
% y = selfC_data(ind{3}(11:end),:);
% y = y(:);
% plot(x,y,'x','MarkerSize',20);   
% xlabel('Puzzle');
% ylabel('Self-confidence');
% legend('Condition 1','Condition 2');
% hold off;