function [e_data,C_data,C0_data,B0_data] = inputVar(datafile)
% Extract input variables (C(t),e_i(t),A(t),B(t))
% Row: participant
% Column: time step
numPuz = 30;

datasize = size(datafile,2);
e_data = zeros(datasize,numPuz,4);
for i=1:datasize
%     % All C data for each participant
%     allC_data(i,:) = [str2double(datafile{i}(2:5,10)); str2double(datafile{i}(7:36,10))];
%     % Normalize C data per participant
%     allC_data_norm(i,:) = (allC_data(i,:)-min(allC_data(i,:)))/(max(allC_data(i,:))-min(allC_data(i,:)));
%     ind = find(any(allC_data_norm,2)==0); % Find rows with one value (can not normalize)
%     allC_data_norm(ind,:)=allC_data(ind,:); % Replace those rows with original data

    % Extract C(t)
%     C_data(i,:) = allC_data_norm(i,4:end);
    C_data(i,:) = [str2double([datafile{i}(5,10)]); str2double([datafile{i}(7:36,10)])];

%     % All selfC data for each participant
%     allSelfC_data(i,:) = [str2double(datafile{i}(2:5,9)); str2double(datafile{i}(7:36,9))];
%     % Normalize selfC data per participant
%     allSelfC_data_norm(i,:) = (allSelfC_data(i,:)-min(allSelfC_data(i,:)))/(max(allSelfC_data(i,:))-min(allSelfC_data(i,:)));
%     ind = find(any(allSelfC_data_norm,2)==0); % Find rows with one value (can not normalize)
%     allSelfC_data_norm(ind,:)=allSelfC_data(ind,:); % Replace those rows with original data
    
    % Extract selfC(t)
%     selfC_data(i,:) = allSelfC_data_norm(i,4:end);
    selfC_data(i,:) = [str2double([datafile{i}(5,9)]); str2double([datafile{i}(7:36,9)])];

    % Extract C0(t)
%     C0_data(i,:) = allC_data_norm(i,3:end);
    C0_data(i,:) = [str2double([datafile{i}(4,10)]) C_data(i,:)];

    % Extract B0(t)
%     B0_data(i) = allC_data_norm(i,1);
    B0_data(i) = str2double([datafile{i}(2,10)]);

    % Extract e_i(t)
    for j=1:numPuz
        feed = str2double(datafile{i}{6+j,8});
        if strcmp(datafile{i}{6+j,4},datafile{i}{6+j,6}) % Chose AI (e1,e3)
            e_data(i,j,feed/5*(-1)+2) = 1;
        else % Chose human (e2,e4)
            e_data(i,j,feed/5*(-1)+3) = 1;
        end
    end
    
    % Extract action (follow or not follow AI)
    if strcmp(datafile{i}{5,4},datafile{i}{5,6})
        act_data(j,1) = 1;
    else
        act_data(j,1) = 0;
    end
    for j=1:numPuz
        if strcmp(datafile{i}{6+j,4},datafile{i}{6+j,6})
            act_data(i,j+1) = 1;
        else
            act_data(i,j+1) = 0;
        end
    end
    
    % Extract performance (feedback 2)
    perf_data(i,:) = [str2double(datafile{i}(5,8)); str2double(datafile{i}(7:36,8))];

    % Extract final sum of feedback 1 
    skill_data(i) = sum(str2double(datafile{i}(7:36,7)));
    
    % Extract final score 
    score_data(i) = str2double(datafile{i}(37,2));
    

    
end

% save('e_data.mat','e_data');
% save('C_data.mat','C_data');
% save('selfC_data.mat','selfC_data');
% save('C0_data.mat','C0_data');
% save('B0_data.mat','B0_data');
% save('act_data.mat','act_data');
% save('perf_data.mat','perf_data');
% save('skill_data.mat','skill_data');
% save('score_data.mat','score_data');

end