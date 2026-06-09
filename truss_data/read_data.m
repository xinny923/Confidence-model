function read_data(N)

% Read all data
% N = number of participants

num_good_moves=[];
for puz=1:33
    moves_score = load(sprintf('~/Desktop/Research/21JMD_Truss_1AI_Confidence/AWS/.results/seq%1.0f/moves_scores_corr.mat',puz));
    threshold = moves_score.moves_corr_scores(1)/2;
    num_good_moves(puz) = sum(moves_score.moves_corr_scores>threshold);
end

feed1_data=cell(1,2);
score1_data=cell(1,2);
feed2_data=cell(1,2);
score2_data=cell(1,2);
C_data=cell(1,2);
sC_data=cell(1,2);
C0_data=cell(1,2);
sC0_data=cell(1,2);
% B0_data=cell(1,2);
% sB0_data=cell(1,2);
e_data=cell(1,2);
e_data{1} = zeros(N/2,30,4);
e_data{2} = zeros(N/2,30,4);
act_data=cell(1,2);

evencount = 0;
oddcount = 0;

for p=1:N
    cond = -1*(rem(p,2)-2); % Condition number
    if cond==1 
        oddcount = oddcount+1;
    else
        evencount = evencount+1;
    end
    % Confidence data
    datafile = table2array(readtable(strcat('~/Desktop/Research/21JMD_Truss_1AI_Confidence/data/.P',num2str(p)...
        ,'/data',num2str(p),'.csv'),'Delimiter',',','ReadVariableNames',false));
    C_data{cond} = [C_data{cond}; datafile(3:end,1)'];
    sC_data{cond} = [sC_data{cond}; datafile(3:end,2)'];
    C0_data{cond} = [C0_data{cond}; datafile(2:end,1)'];
    sC0_data{cond} = [sC0_data{cond}; datafile(2:end,2)'];
%     B0_data{cond} = [B0_data{cond}; datafile(1,1)];
%     sB0_data{cond} = [sB0_data{cond}; datafile(1,2)];
    % Feedback1 data
    feed1 = ones(1,30)*-5;
    for puz=4:33
        moves_desc = load(sprintf('~/Desktop/Research/21JMD_Truss_1AI_Confidence/AWS/.results/seq%1.0f/moves_desc.mat',puz));
        ref = load(sprintf('~/Desktop/Research/21JMD_Truss_1AI_Confidence/data/.P%1.0f/seq%1.0f.move1.mat',p,puz));
        for topnum = 1:num_good_moves(puz)
            if puz==4 || puz==5 ||puz==7||puz==8||puz==9||puz==12||puz==15||puz==33||puz==37
                topimg_filename = strrep(moves_desc.moves_desc_order{topnum},':','_');
            else
                topimg_filename = moves_desc.moves_desc_order{topnum};
            end
            topimg = load(strcat('~/Desktop/Research/21JMD_Truss_1AI_Confidence/AWS/.results/seq',string(puz),'/Candidates/',topimg_filename));
            if isequal(length(ref.member_info),length(topimg.member_info)) & isequal(length(ref.node_info),length(topimg.node_info))
                for b=1:length(ref.member_info)
                    roundn = @(x,n) 10.^n .* round(x/10.^n);
                    memeqx = isequal(roundn(topimg.member_info(b).x,-4),roundn(ref.member_info(b).x,-4));
                    memeqy = isequal(roundn(topimg.member_info(b).y,-4),roundn(ref.member_info(b).y,-4));
                    if memeqx+memeqy == 0
                        memeqx = isequal(flip(roundn(topimg.member_info(b).x,-4)),roundn(ref.member_info(b).x,-4));
                        memeqy = isequal(flip(roundn(topimg.member_info(b).y,-4)),roundn(ref.member_info(b).y,-4));
                    end
                    memeqLW = isequal(topimg.member_info(b).LW,ref.member_info(b).LW);
                    memeq = memeqx+memeqy+memeqLW;
                    if memeq ~= 3
                        break;
                    end
                end
                if memeq ~=3
                    continue;
                end
                for d=1:length(ref.node_info)
                    nodeeqx = isequal(roundn(topimg.node_info(d).x,-4),roundn(ref.node_info(d).x,-4));
                    nodeeqy = isequal(roundn(topimg.node_info(d).y,-4),roundn(ref.node_info(d).y,-4));
                    if nodeeqx+nodeeqy ==0
                        nodeeqx = isequal(roundn(topimg.node_info(d).x,-4),roundn(ref.node_info(d).y,-4));
                        nodeeqy = isequal(roundn(topimg.node_info(d).y,-4),roundn(ref.node_info(d).x,-4));
                    end
                    nodeeq = nodeeqx+nodeeqy;
                    if nodeeq ~= 2
                        break;
                    end
                end
                if nodeeq ~=2
                    continue;
                end
                feed1(puz-3) = 5;
                break;
            end
        end
    end
    feed1_data{cond} = [feed1_data{cond}; feed1];
    % Individual score data
    score1_data{cond} = [score1_data{cond}; sum(feed1)];
    % Feedback2 data
    feed2 = ones(1,30)*-5;
    for puz=4:33
        moves_desc = load(sprintf('~/Desktop/Research/21JMD_Truss_1AI_Confidence/AWS/.results/seq%1.0f/moves_desc.mat',puz));
        ref = load(sprintf('~/Desktop/Research/21JMD_Truss_1AI_Confidence/data/.P%1.0f/seq%1.0f.move2.mat',p,puz));
        for topnum = 1:num_good_moves(puz)
            if puz==4 || puz==5 ||puz==7||puz==8||puz==9||puz==12||puz==15||puz==33||puz==37
                topimg_filename = strrep(moves_desc.moves_desc_order{topnum},':','_');
            else
                topimg_filename = moves_desc.moves_desc_order{topnum};
            end
            topimg = load(strcat('~/Desktop/Research/21JMD_Truss_1AI_Confidence/AWS/.results/seq',string(puz),'/Candidates/',topimg_filename));
            if isequal(length(ref.member_info),length(topimg.member_info)) & isequal(length(ref.node_info),length(topimg.node_info))
                for b=1:length(ref.member_info)
                    roundn = @(x,n) 10.^n .* round(x/10.^n);
                    memeqx = isequal(roundn(topimg.member_info(b).x,-4),roundn(ref.member_info(b).x,-4));
                    memeqy = isequal(roundn(topimg.member_info(b).y,-4),roundn(ref.member_info(b).y,-4));
                    if memeqx+memeqy == 0
                        memeqx = isequal(flip(roundn(topimg.member_info(b).x,-4)),roundn(ref.member_info(b).x,-4));
                        memeqy = isequal(flip(roundn(topimg.member_info(b).y,-4)),roundn(ref.member_info(b).y,-4));
                    end
                    memeqLW = isequal(topimg.member_info(b).LW,ref.member_info(b).LW);
                    memeq = memeqx+memeqy+memeqLW;
                    if memeq ~= 3
                        break;
                    end
                end
                if memeq ~=3
                    continue;
                end
                for d=1:length(ref.node_info)
                    nodeeqx = isequal(roundn(topimg.node_info(d).x,-4),roundn(ref.node_info(d).x,-4));
                    nodeeqy = isequal(roundn(topimg.node_info(d).y,-4),roundn(ref.node_info(d).y,-4));
                    if nodeeqx+nodeeqy ==0
                        nodeeqx = isequal(roundn(topimg.node_info(d).x,-4),roundn(ref.node_info(d).y,-4));
                        nodeeqy = isequal(roundn(topimg.node_info(d).y,-4),roundn(ref.node_info(d).x,-4));
                    end
                    nodeeq = nodeeqx+nodeeqy;
                    if nodeeq ~= 2
                        break;
                    end
                end
                if nodeeq ~=2
                    continue;
                end
                feed2(puz-3) = 5;
                break;
            end
        end
    end
    feed2_data{cond} = [feed2_data{cond}; feed2];
    % Final score data
    score2_data{cond} = [score2_data{cond}; sum(feed2)];
    % People to receive extra $5
    if sum(feed2)>10
        p;
    end
    % Experience data
    e = zeros(30,4);
    action = zeros(1,30);
    for j=4:33
        ref = load(sprintf('~/Desktop/Research/21JMD_Truss_1AI_Confidence/data/.P%1.0f/seq%1.0f.move2.mat',p,j));
        sugg = load(strcat('~/Desktop/Research/21JMD_Truss_1AI_Confidence/data/sugg_files/',string(cond),'/seq',string(j),'.sugg.mat'));
        chooseai=0;
        if isequal(length(ref.member_info),length(sugg.member_info)) & isequal(length(ref.node_info),length(sugg.node_info))
            for b=1:length(ref.member_info) % Check if the members are all the same
                roundn = @(x,n) 10.^n .* round(x/10.^n);
                memeqx = isequal(roundn(sugg.member_info(b).x,-4),roundn(ref.member_info(b).x,-4));
                memeqy = isequal(roundn(sugg.member_info(b).y,-4),roundn(ref.member_info(b).y,-4));
                if memeqx+memeqy == 0
                    memeqx = isequal(flip(roundn(sugg.member_info(b).x,-4)),roundn(ref.member_info(b).x,-4));
                    memeqy = isequal(flip(roundn(sugg.member_info(b).y,-4)),roundn(ref.member_info(b).y,-4));
                end
                memeqLW = isequal(sugg.member_info(b).LW,ref.member_info(b).LW);
                memeq = memeqx+memeqy+memeqLW;
                if memeq ~= 3
                    break;
                end
            end
            if memeq ==3
                for d=1:length(ref.node_info) % Check if the nodes are all the same
                    nodeeqx = isequal(roundn(sugg.node_info(d).x,-4),roundn(ref.node_info(d).x,-4));
                    nodeeqy = isequal(roundn(sugg.node_info(d).y,-4),roundn(ref.node_info(d).y,-4));
                    if nodeeqx+nodeeqy ==0
                        nodeeqx = isequal(roundn(sugg.node_info(d).x,-4),roundn(ref.node_info(d).y,-4));
                        nodeeqy = isequal(roundn(sugg.node_info(d).y,-4),roundn(ref.node_info(d).x,-4));
                    end
                    nodeeq = nodeeqx+nodeeqy;
                    if nodeeq ~= 2
                        break;
                    end
                end
                if nodeeq==2
                    chooseai = 1; % if the code gets here, then chose AI (e1,e3)
                    action(j-3) = chooseai;
                end
            end
        end
        if chooseai==1 % Chose AI (e1,e3)
            e(j-3,feed2(j-3)/5*(-1)+2) = 1;
        else % Chose human (e2,e4)
            e(j-3,feed2(j-3)/5*(-1)+3) = 1;
        end
    end
    if cond==1
        e_data{cond}(oddcount,:,:) = e;
    else
        e_data{cond}(evencount,:,:) = e;
    end
    act_data{cond} = [act_data{cond}; action];
    
end

% save(strcat('feed1_data.mat'),'feed1_data');
% save(strcat('score1_data.mat'),'score1_data');
% save(strcat('feed2_data.mat'),'feed2_data');
% save(strcat('score2_data.mat'),'score2_data');
% % save(strcat('C_data.mat'),'C_data');
% % save(strcat('sC_data.mat'),'sC_data');
% % save(strcat('e_data.mat'),'e_data');
% save(strcat('act_data.mat'),'act_data');

end