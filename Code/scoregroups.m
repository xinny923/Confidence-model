function scoregroups(cond)
% load('skill_data.mat'); % Sum of feedback 1
load('score_data.mat'); % Sum of feedback 2

% -----------------------Overall histogram---------------------------------
data = score_data;
% Find low, medium, high performing participants
datamean = mean(data);
datastd = std(data);
z = 0.675; % 25% with given mean and std
lox = datamean-z*datastd;
hix = datamean+z*datastd;
% Index of low, medium, high performing participants
loind = find(data<=lox);
hiind = find(data>=hix);
miind = find((data<hix) & (data>lox));
ind = {loind, miind, hiind};
group = ind;

% --------------------------Histogram by condition----------------------------------
i = 1+(cond-1)*50;

skdata = skill_data(i:i+49);
scdata = score_data(i:i+49);

count = 0; 
alldata = {skdata scdata};
for j = 1:2
    count = count +1;
    
    data = alldata{j};
    % Find low, medium, high performing participants
    datamean = mean(data);
    datastd = std(data);
    z = 0.675; % 25% with given mean and std
    lox = datamean-z*datastd;
    hix = datamean+z*datastd;
    % Index of low, medium, high performing participants
    loind = find(data<=lox);
    hiind = find(data>=hix);
    miind = find((data<hix) & (data>lox));
    ind = {loind, miind, hiind};
    group = ind;
    
%     if count == 1
%         save(strcat('skillgroup',num2str(cond),'.mat'),'group');
%     else
%         save(strcat('scoregroup',num2str(cond),'.mat'),'group');
%     end

    % [h,p] = lillietest(skdata); % Check normality
% ------------------------Plot histogram----------------------------------
    figure;
    plot(lox.*ones(100,1),linspace(0,15),'Color',[1.0,0.4,0],'LineWidth',1);
    hold on;
    plot(hix.*ones(100,1),linspace(0,15),'Color',[1.0,0.4,0],'LineWidth',1);
    v1 = [-150 0;-150 15;lox 15;lox 0];
    v2 = [lox 15; lox 0;hix 0; hix 15];
    v3 = [hix 0; hix 15; 150 15; 150 0];
    f1 = [1 2 3 4];
    patch('Faces',f1,'Vertices',v1,'FaceColor',[1.0,0.4,0],'FaceAlpha',.07,'LineStyle','None');
    patch('Faces',f1,'Vertices',v2,'FaceColor',[1,0.8,0],'FaceAlpha',.07,'LineStyle','None');
    patch('Faces',f1,'Vertices',v3,'FaceColor',[0,0.4,0.7],'FaceAlpha',.07,'LineStyle','None');

    skdist = histogram(data,15,'FaceColor',[0,0.4,0.7]); % skill histogram
%     text((lox-150)/2,12,num2str(length(loind)),'FontSize',30)
%     text((lox+hix)/2,12,num2str(length(miind)),'FontSize',30)
%     text((150+hix)/2,12,num2str(length(hiind)),'FontSize',30)

    hold off;
    if j==1
        title(strcat('Condition ',string(cond),' Skill Histogram'),'FontSize',30);
        xlabel('Skill');
    else
%         title(strcat('Condition ',string(cond),' Team performance score histogram'),'FontSize',30);
        xlabel('Team performance score','FontSize',15,'FontWeight','bold');
    end
    axis([-150 150 0 15]);
    ylabel('Frequency','FontSize',15,'FontWeight','bold');
%     legend('Low 25%','High 25%','FontSize',15);
    set(gca,'FontSize',15)
    set(gca,'FontName','Helvetica')
    box on;
    grid on;

end

[R,P] = corrcoef(skdata,scdata);

end