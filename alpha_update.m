function [Ahat,alpha_opt] = alpha_update(B,C,Ai,opt)
% EM update of coefficients alpha and gammaw in dictionary learning where
% 'linear': A = alpha1*A1 + alpha2*A2 + .. + alphaP*AP
% 'csmu'  : A = A0 + alpha1*A1 + .. + alphaP*AP

% inputs: 
% Ai: cell array of Ai of dimensions 1xP (linear) , 1x(P+1) (csmu)

% If opt='linear':
    % Returns the coefficients alpha and Ahat that minimizes
    % J(alpha) = trace(A*B*A'-2*A'C)   
    % Solution:
    % alpha_opt = 2*inv(G+G')*v
    % gammaw_opt= ML/(alpha_opt'*G*alpha_opt-2*alpha_opt'*beta), where
    % G_ij = trace(Ai*B*Aj'), v_i = trace(Ai'*C)

[M,N] = size(C);
switch opt
    case 'linear'
        P = length(Ai);
        G = zeros(P);
        v = zeros(P,1);
        for p=1:P
            %v(p) = trace(Ai{p}'*C);
            v(p) = sum(sum(Ai{p}.*C));
            for l=1:P
                %G(p,l) = trace(Ai{p}*B*Ai{l}');
                Gtemp = B*Ai{l}';
                G(p,l) = sum(sum(Ai{p}'.*Gtemp));
            end
        end

        alpha_opt = 2*pinv(G + G')*v;

        Ahat = zeros(M,N);
        for p=1:P
            Ahat = Ahat + alpha_opt(p)*Ai{p};
        end
    case 'csmu'
        P = length(Ai)-1;
        G = zeros(P);
        v = zeros(P,1);
        A0 = Ai{P+1};
        for i=1:P
            v(i) = trace(A0*B*Ai{i}')-trace(Ai{i}'*C);
            for j=1:P
                G(i,j) = trace(Ai{i}*B*Ai{j}');
            end
        end

        alpha_opt = -2*pinv(G + G')*v;

        Ahat = A0;
        for i=1:P
            Ahat = Ahat + alpha_opt(i)*Ai{i};
        end
end
