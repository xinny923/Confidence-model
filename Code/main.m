clear all;

for i=1:100 %Robustness test
    % load('x.mat'); 
    x0 = [0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]; % Starting guess
    
    % Robustness test
    x0 = [0.267181677384535,0.340450953120926,0.0524238420736158,0.843879167378967,0.211531837817135,2.32469776755680e-14,0.521737124180336,0.389670818141326]; %confidence in AI
    x0 = [0.284419387583965,0.470608119908973,2.25029047761617e-14,0.573623483940813,0.828368714233239,0.238391875829198,0.286322496033589,0.114727712177294]; %self-confidence
    chosen = randperm(100,80);
    
    global selfC_data e_data
    load('selfC_data.mat');
    load('e_data.mat');
    
    % Robustness test
    selfC_data = selfC_data(chosen,:);
    e_data = e_data(chosen,:,:);

    options.TolFun = 1e-10;
    options.TolX = 1e-10;
    options.MaxFunEvals = 1e4;
    options.MaxIter = 1e3;
    %options.Algorithm = 'levenberg-marquardt';
    [x,resnorm] = lsqnonlin(@myfun_self,x0,[0 0 0 0 0 0 0 0],[1 1 1 1 1 1 1 1],options); % Invoke optimizer % Change accordingly
    %[x,resnorm] = lsqnonlin(@myfun,x0,[],[],options);
    %[x,resnorm] = lsqnonlin(@myfun,x0,[],[]);

    % save('x.mat','x'); % Change accordingly
    if i==1
        csvwrite('data_robust_self.csv',chosen);
        csvwrite('x_robust_self.csv',x);
    else
        dlmwrite('data_robust_self.csv',chosen,'delimiter',',','-append');
        dlmwrite('x_robust_self.csv',x,'delimiter',',','-append');
    end
end