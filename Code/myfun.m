function F = myfun(x)
% x = [ae, aa, ab, w1, w2, w3, w4, g]

% -----Change these accordingly-----
numPuz = 30;

global C_data e_data 

% load('C_data.mat');
% % load('C0_data.mat');
% % load('B0_data.mat');
% load('e_data.mat');
% % load('datafile0.mat'); 

% ----------------------------------
% b = B0_data';
b = C_data(:,1);
c = b;
C = c;
for i=1:numPuz
    if i==1
        a = C_data(:,i);
    else
        a = x(8).*C(:,i-1)+(1-x(8)).*a;
    end
    e = x(4).*e_data(:,i,1) + x(5).*e_data(:,i,2) + x(6).*e_data(:,i,3) + x(7).*e_data(:,i,4);
    c = c + x(1).*(e-c) + x(2).*(a-c) + x(3).*(b-c);
    C = [C c];
end

F = C_data(:,2:end)'-C(:,2:end)';
F = reshape(F,[],1);

% syms E(w1,w2,w3,w4,e1,e2,e3,e4) 
% syms A(g,C0,A0)
% syms B(B0)
% syms pred(C,E_sym,A_sym,B_sym,ae,aa,ab)
% syms A0
% c = sym('c_%d',[1 numPuz]);
% 
% load('symPredC.mat');
% 
% % Substitution and find F
% modelPred = {};
% pred = [];
% F = [];
% for i=1:numPuz
%     modelPred{i} = subs(symPredC{i},[ae,aa,ab,w1,w2,w3,w4,g],x);
%     for j=1:size(C_data,1)
%         % C(t+1) value predicted by the model & c_i substitution for accumulated confidence update
%         pred(i,j) = double(subs(modelPred{i},[A0,C0,B0,C,e1,e2,e3,e4,c],...
%             [C_data(j,1),C0_data(j,i),B0_data(j),C_data(j,i),e_data(j,i,1),...
%             e_data(j,i,2),e_data(j,i,3),e_data(j,i,4),C_data(j,2:numPuz+1)]));        
%         % Difference (actual - pred)
%         F(i,j) = str2double([datafile{j}(6+i,10)]) - pred(i,j);
%     end
% end

end

