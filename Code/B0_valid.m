load('./datafile0.mat');
count=0;
for i=1:length(datafile)
    if str2double(datafile{i}(2,9))==str2double(datafile{i}(3,9))
        count = count+1;
    end
end