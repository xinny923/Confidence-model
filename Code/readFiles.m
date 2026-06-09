function datafile = readFiles(startdata,N,cond)
% Read all data
% If cond = 0, all data files are read. 

if cond == 0
    for condi=1:2
        for i=1:N
            datafile{N*(condi-1)+i} = table2array(readtable(strcat(...
                '/Users/leahmchong/Desktop/Trust/ChessStudy/data/.data',num2str(startdata)...
                ,'_',num2str(condi),'.csv'),'Delimiter',',','ReadVariableNames',false));
            startdata = startdata+1;
        end
        startdata = 1;
    end
else
    for i=1:N
        datafile{i} = table2array(readtable(strcat(...
            '/Users/leahmchong/Desktop/Trust/ChessStudy/data/.data',num2str(startdata)...
            ,'_',num2str(cond),'.csv'),'Delimiter',',','ReadVariableNames',false));
        startdata = startdata+1;
    end  
end

save(strcat('datafile',num2str(cond),'.mat'),'datafile');

end