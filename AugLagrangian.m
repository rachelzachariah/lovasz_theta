function [ min , argmin ] = AugLagrangian(A,mu,maxiter)
% SDP to compute Lovasz number of graph G with adjacency matrix A
% Using the augmented Lagrangian method of Wen-Goldfarb-Yin

vec = @(X) X(:);
m = sum(sum(A))+1; %no of linear constraints
n = size(A,1);
C = -ones(n^2,1);
Alpha = zeros(m,n^2);
row=1;
for i = 1:n
    for j=1:n
        if A(i,j)>0
            Alpha(row,(i-1)*n+j)=1;
            row=row+1;
        end
        if i==j
            Alpha(m,(i-1)*n+j)=1;
        end
    end
end
%fprintf('rank of %f', rank(Alpha))
b=zeros(m,1);
b(m)=1;
Xk= zeros(n^2,1);
Sk= vec(eye(n));

P = (Alpha*Alpha')^-1;

for i=1:maxiter
    yk= -P*(mu*(Alpha*Xk-b)+Alpha*(Sk-C));
    Vk= C - Alpha'*yk-mu*Xk;
    
    [U,S] = eig(reshape(Vk,n,n));
    S(logical(S<0))=0;
    Sk= vec(U*S*U'); 
    
    Xk=vec(1/mu*(Sk-Vk));
    
    if mod(i,1)==0
        fprintf('iteration %d, objective %f\n', i, sum(Xk));
    end
    
end

argmin=Xk;
min=sum(Xk);

end

