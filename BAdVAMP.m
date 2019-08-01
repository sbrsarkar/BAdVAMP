function estHist = BAdVAMP(gX,Y,emVampOpt)

%% Load Parameters
N = emVampOpt.N;
M = emVampOpt.M;
L = emVampOpt.L;
dampX = emVampOpt.dampX;
nit = emVampOpt.nit;
verbose = emVampOpt.verbose;
tol = emVampOpt.tol;
learnDict = emVampOpt.learnDict;
Atype = emVampOpt.Atype;
delayAupdate = emVampOpt.delayAupdate;
checkConv = emVampOpt.checkConv;

%% Declare History variables
estHist.A = zeros(M,N,nit+1);
estHist.X1 = zeros(N,L,nit);
estHist.X2 = zeros(N,L,nit);
estHist.eta1 = zeros(L,nit);
estHist.eta2 = zeros(L,nit);
estHist.gammaw = zeros(1,nit+1);
estHist.errA = zeros(1,nit+1);
estHist.errZ = zeros(1,nit);
estHist.errX1 = zeros(1,nit);
estHist.errX2 = zeros(1,nit);

%% Declare variables
earlyStop = false;
computeSVD = true;
X2 = zeros(N,L); %needed for convergence test in just VAMP

fxnQd = @(Q,v) bsxfun(@times,Q,v'); %fast implementation of  Q*diag(v)
onsagRX = @(X,eta,R,gam) fxnQd(fxnQd(X,eta)-fxnQd(R,gam),1./(eta-gam));

%% Initialization
Ahat = emVampOpt.Ahat0;
R1 = emVampOpt.R10;
gamma1 = emVampOpt.gamma10;
gammaw = emVampOpt.gamw0;

%save the initial values
estHist.errA(1) = emVampOpt.error_functionA(Ahat);
estHist.gammaw(1) = gammaw;
estHist.A(:,:,1) = Ahat;

% compute initial left-eigenvalues of Ahat0
if ~learnDict
    [V,S] = svd((Ahat'*Ahat));
end

estHist.erralf = [];
estHist.errb = [];
estHist.errc = [];
estHist.errBC = [];
estHist.bhat = [];
estHist.chat = [];

%% MAIN LOOP
for it=1:nit
    %X1 update with auto-tuning gamma1
    [X1,xvar1,rvar1] = gX.estim(R1,1./gamma1);
    eta1 = 1./mean(xvar1,1)';
    gamma1 = 1./mean(rvar1,1)';
    
    %R2 update
    R2 = onsagRX(X1,eta1,R1,gamma1);
    gamma2 = abs(eta1 - gamma1); % abs is more robust
    
    %compute svd of the new dictionary
    if learnDict && computeSVD
        [V,S] = eig((Ahat'*Ahat));
        computeSVD = false;
        % if verbose, fprintf('sv'); end
    end
    
    X2old = X2;
    Aold = Ahat;
    
    D = 1./bsxfun(@plus,gammaw*diag(S),gamma2');
    
    %X2 update
    X2 = V*(D.*(V'*(gammaw*Ahat'*Y + fxnQd(R2,gamma2))));
    eta2 = 1./mean(D)';
    
    Qsum = V*diag(sum(D,2))*V';
    %Qsum = sum(1./eta2)*eye(N); % faster but doesn't work well for wide A
    
    %A update
    B = Qsum+(X2*X2'); C = Y*X2';
    if learnDict && mod(it,delayAupdate)==0
        if strcmp(Atype,'linear')||strcmp(Atype,'toeplitz')||...
                strcmp(Atype,'toeplitz-alf')
            [Ahat,alpha_opt] = alpha_update(B,C,emVampOpt.Alist,'linear');
            estHist.erralf = [estHist.erralf,...
                emVampOpt.error_function_alf(alpha_opt)];
        elseif strcmp(Atype,'csmu')
            [Ahat,alpha_opt] = alpha_update(B,C,emVampOpt.Alist,'csmu');
            estHist.errb = [estHist.errb emVampOpt.error_functionB(alpha_opt)];
            estHist.errc = [estHist.errc emVampOpt.error_functionC(X2)];
        elseif strcmp(Atype,'calib')
            [Ahat,bhat] = alpha_update(B,C,emVampOpt.Alist,'linear');
            chat = X2;
            bc_est = bhat*chat';
            estHist.errBC = [estHist.errBC emVampOpt.error_functionBC(bc_est)];
            estHist.bhat = [estHist.bhat, bhat];
            estHist.chat = [estHist.chat, chat];
        else
            Ahat = Y*X2'*pinv(Qsum + (X2*X2'));
        end
        computeSVD = true;
    end
    
    %update noise precision gammaw
    gammaw = M*L/(norm(Y-Ahat*X2,'fro')^2 + trace(Ahat*Qsum*Ahat'));
    
    
    %R1 update
    R1 = (1-dampX)*R1 + dampX*onsagRX(X2,eta2,R2,gamma2);
    gamma1 = (1-dampX)*gamma1 + dampX*(eta2 - gamma2);
    
    % error calculation
    Zhat = Ahat*X2;
    A_e = emVampOpt.error_functionA(Ahat);
    Z_e = emVampOpt.error_function(Zhat);
    X1_e= emVampOpt.error_functionX(X1);
    X2_e= emVampOpt.error_functionX(X2);
    
    if mod(it,verbose)==0
        switch Atype
            case 'calib'
                fprintf('\nR=%2d it=%3d  bc_e=%8.4f  Z_e=%8.4f ',...
                    emVampOpt.numR,it,estHist.errBC(end));
                
            otherwise
                fprintf('\nR=%2d it=%3d  A_e=%8.4f  X1_e=%8.4f  X_2=%8.4f  Z_e=%8.4f ',...
                    emVampOpt.numR,it,A_e,X1_e,X2_e,Z_e);
        end
    end
    
    %-----saving history--------------
    estHist.A(:,:,it+1) = Ahat;
    estHist.X1(:,:,it) = X1;
    estHist.X2(:,:,it) = X2;
    estHist.eta1(:,it) = eta1;
    estHist.eta2(:,it) = eta2;
    estHist.gammaw(it+1) = gammaw;
    estHist.errA(it+1) = A_e;
    estHist.errZ(it) = Z_e;
    estHist.errX1(it) = X1_e;
    estHist.errX2(it) = X2_e;
    
    % convergence check
    if checkConv && mod(it,delayAupdate)==0 && ...
            learnDict && norm(Ahat-Aold,'fro')/norm(Ahat,'fro')<tol
        earlyStop = true;
        itEnd = it;
        break;
    end
    
    if checkConv && ~learnDict && it>1 && ...
            norm(X2-X2old,'fro')/norm(X2,'fro')<tol
        earlyStop = true;
        itEnd = it;
        break;
    end
end

if ~earlyStop
    itEnd = it;
end

%trimming the results
estHist.errA = estHist.errA(1:itEnd+1);
estHist.errZ = estHist.errZ(1:itEnd);
estHist.errX1 = estHist.errX1(1:itEnd);
estHist.errX2 = estHist.errX2(1:itEnd);
estHist.A = estHist.A(:,:,1:itEnd+1);
estHist.gammaw = estHist.gammaw(1:itEnd+1);
estHist.eta1 = estHist.eta1(:,1:itEnd);
estHist.eta2 = estHist.eta2(:,1:itEnd);

