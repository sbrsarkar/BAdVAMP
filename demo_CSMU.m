% Compressed Sensing with Matrix Uncertainty (CSMU)
% Estimate c and b1, b2, .., bk from measurements y = A*c + w where
% A = A0 + b1*A1 + b_2*A2 + .. + bk*Ak, c~BG(rho,0,1), w~N(0,wvar)

clear all;
% handle random seed
defaultStream = RandStream.getGlobalStream;
if 1 % new RANDOM trial
    savedState = defaultStream.State;
    save random_state.mat savedState;
else % repeat last trial
    load random_state.mat
end
defaultStream.State = savedState;


%%--------------------PARAMETERS----------------------------
rate = 0.6; % rate=M/N
Nb = 10;  %length of b
Nc = 256; %length of c
K  = 10;  %no. of non-zeros in c
L  = 1;
SNRdB = 40;

bmean0 = 0; bvar0 = 1;
cmean0 = 0.1; cvar0 = 1;

N = Nc;
M = ceil(rate*N);

Anorm = N; % Frobenius norm of A.

% EMVAMP options
emVampOpt.nit = 200;    %nit for EMVamp
emVampOpt.dampX = 0.85; %damping of R1 and gamma1
emVampOpt.delayAupdate = 1; sty = '-';
emVampOpt.EM_R1nit = 0; %EM iterations for gamma1 (1 works the best for square A)
emVampOpt.tuner1pre = false;
emVampOpt.tuner2pre = false;
emVampOpt.learnDict = true; %dictionary learning
emVampOpt.tol = 1e-5;
emVampOpt.checkConv = true;
emVampOpt.verbose = 1;

restart_tol = 1e-3;
nRestarts = 0;

%%--------- GENERATE REALIZATION ----------
% Build true signals
% Draw Bernoulli-Gaussian signal vectors
b = bmean0 + sqrt(bvar0)*randn(Nb,1);
c = cmean0 + sqrt(cvar0)*randn(Nc,L);

%sparsify c
locs = randperm(Nc);
c(locs((K+1):end)) = 0;

%% Build the dictionary
A0 = sqrt(10)*randn(M,N);
Ai = randn(M*N,Nb);

Alist = cell(Nb+1,1);
Alist{Nb+1} = A0;
A = Alist{Nb+1};
for i=1:Nb
    Alist{i} = reshape(Ai(:,i),M,N);
    A = A + b(i)*Alist{i};
end
emVampOpt.Alist = Alist;
disp('Atrue: iid');

%Save the true Z
z = A*c;

%Determine nuw
nuw = norm(reshape(z,[],1))^2/M*10^(-SNRdB/10);
nuw = max(nuw,1e-10);
wvar = nuw;

%Noisy output channel
Y = z + sqrt(nuw)*randn(size(z));


%Specify error functions
error_function = @(qval) 20*log10(norm(qval - z,'fro') / norm(z,'fro'));
error_functionB = @(qval) 20*log10(norm(qval - b)/norm(b));
error_functionC = @(qval) 20*log10(norm(qval - c)/norm(c));


%%---------------------- BAdVAMP ------------------------
emVampOpt.error_functionA = @(q) 20*log10(norm(A - q,'fro')/norm(A,'fro'));
emVampOpt.error_functionB = error_functionB;
emVampOpt.error_functionC = error_functionC;
emVampOpt.error_functionX = error_functionC;
emVampOpt.error_function = error_function;

%% Declare estimators
%Prior on C
gC = AwgnEstimIn(cmean0, cvar0);
gC = SparseScaEstim(gC,K/Nc);
gC = MyVarEstimIn(gC,'nit',emVampOpt.EM_R1nit); %My auto-tuner

b0 = randn(Nb,1);
Ahat0 = Alist{Nb+1};
for l=1:Nb
    Ahat0 = Ahat0 + b0(l)*Alist{l};
end
disp('Ahat0: csmu');

%Init R1: using least squares, true X for gamma10
R10 = pinv(Ahat0)*Y;
gamma10 = 1/bvar0*ones(L,1);

%Init gamw: measurement noise variance
gamw0 = M*L*(1+wvar)/norm(Y,'fro')^2; %1e-10;

emVampOpt.Ahat0 = Ahat0;
emVampOpt.R10 = R10;
emVampOpt.gamma10 = gamma10;
emVampOpt.gamw0 = gamw0;

%Optional
emVampOpt.N = N; emVampOpt.M = M; emVampOpt.L = L; emVampOpt.K = K;
emVampOpt.Atype = 'csmu';

vampHist.errB = [];
vampHist.errC = [];
vampHist.errZ = [];

Ahat0 = emVampOpt.Ahat0;

start = tic;
%Run BAdVAMP multiple times
for t=1:nRestarts+1
    emVampOpt.numR = t-1; %restart number
    
    vampHist2 = BAdVAMP(gC,Y,emVampOpt);
    emVampOpt.Ahat0 = vampHist2.A(:,:,end);
    emVampOpt.R10 = pinv(emVampOpt.Ahat0)*Y;
    
    vampHist.errB = [vampHist.errB vampHist2.errb];
    vampHist.errC = [vampHist.errC vampHist2.errc];
    vampHist.errZ = [vampHist.errZ vampHist2.errZ];
    
    Ahat1 = vampHist2.A(:,:,end);
    if emVampOpt.checkConv && norm(Ahat1-Ahat0,'fro')/norm(Ahat0,'fro')<restart_tol
        break;
    end
    Ahat0 = Ahat1;
end
tEMVAMP = toc(start);

%%-------------- Show Final NMSE and time-----------------
fprintf('\n\nBAdVAMP: ');
fprintf('%d minutes and %2.2f seconds\n', floor(tEMVAMP/60), rem(tEMVAMP,60));
fprintf('b_e = %3.4f   c2_e = %3.4f Z_E = %3.4f',...
    vampHist.errB(end),vampHist.errC(end),vampHist.errZ(end));
fprintf('\ntotal iterations: %d\n', size(vampHist.errB,2));

%%---------------------- PLOT RESULTS -----------------
figure(4);clf;
subplot(121);
    plot(vampHist.errB,'-r');
    ylabel('NMSE(b)'); xlabel('iterations');
    grid on;

subplot(122);
    plot(vampHist.errC,'-r');
    ylabel('NMSE(c)'); xlabel('iterations');
    grid on;
