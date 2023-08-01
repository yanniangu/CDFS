function [ W ] = initializationW( Xl, Fl, k)
%INITIALIZATIONW 此处显示有关此函数的摘要
%   此处显示详细说明
n = size(Xl, 2);
gamma = 1;

G = Fl.^gamma;
S = diag(sum(G,2));
D = diag(1./sum(G,1));

a = sum(S(:));
Lt = S - 1/a*S*ones(n,1)*ones(1,n)*S;
Lw = S - G*D*G';

St = Xl*Lt*Xl';
Sw = Xl*Lw*Xl';
Sb = St - Sw;

score = diag(Sb)./(diag(St)+eps);
[~, rank] = sort(score,'descend');
Q = zeros(size(Xl,1),k);
for i = 1:k
    Q(rank(i),i) = 1;
end
P = pinv(Q'*Xl*S*Xl'*Q)*Q'*Xl*G;
W = Q*P;
end

