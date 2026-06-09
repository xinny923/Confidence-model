function F = myfun_self_byscore(x)
% x = [ae, aa, ab, w1, w2, w3, w4, g]

% -----Change these accordingly-----
numPuz = 30;

load('selfC_data.mat');
% load('selfC0_data.mat');
% load('B0_data.mat');
load('e_data.mat');
load('datafile0.mat'); 

score_group = 3; % 1, 2, or 3 (low to high respectively)
load('scoregroup1.mat');
ind = group{score_group};
load('scoregroup2.mat');
ind = [ind 50+group{score_group}];

% ----------------------------------
b = selfC_data(ind,1);
c = b;
C = c;
for i=1:numPuz
    if i==1
        a = selfC_data(ind,i);
    else
        a = x(8).*C(:,i-1)+(1-x(8)).*a;
    end
    e = x(4).*e_data(ind,i,1) + x(5).*e_data(ind,i,2) + x(6).*e_data(ind,i,3) + x(7).*e_data(ind,i,4);
    c = c + x(1).*(e-c) + x(2).*(a-c) + x(3).*(b-c);
    C = [C c];
end

F = selfC_data(ind,2:end)'-C(:,2:end)';
F = reshape(F,[],1);

end

