function symPredC = modelfun(numPuz)
% Define symbolic functions of the model (different for every t)

load('x.mat');

syms E(w1,w2,w3,w4,e1,e2,e3,e4) 
syms A(g,C0,A0)
syms B(B0)
syms pred(C,E_sym,A_sym,B_sym,ae,aa,ab)
syms A0
c = sym('c_%d',[1 numPuz]);

E(w1,w2,w3,w4,e1,e2,e3,e4) = w1*e1 + w2*e2 + w3*e3 + w4*e4;
A(g,C0,A0) = g*C0 + (1-g)*A0;
B(B0) = B0;
predC(C,E_sym,A_sym,B_sym,ae,aa,ab) = C + ae*(E_sym-C) + aa*(A_sym-C) + ab*(B_sym-C);

symPredC = {};
B0_store = {B0};
A0_store = {A0 A0 A(g,c(1),A0)};
for i=1:numPuz
    symPredC{i} = predC(C,E,A0_store{i},B0_store{i},ae,aa,ab);

    if i>2
        A0_store{i+1} = A(g,c(i-1),A0_store{i});
    end

    B0_store{i+1} = B(B0_store{i});
end


save('symPredC.mat','symPredC');

end