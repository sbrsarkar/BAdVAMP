% Dictionary Learning (DL)


clear all;
% handle random seed
if verLessThan('matlab','7.14')
    defaultStream = RandStream.getDefaultStream;
else
    defaultStream = RandStream.getGlobalStream;
end
if 1 % new RANDOM trial
    savedState = defaultStream.State;
    save random_state.mat savedState;
else % repeat last trial
    load random_state.mat
end
defaultStream.State = savedState;

%%--------------------PARAMETERS----------------------------
N = 64;
M = N;
L = ceil(5*N*log(N));
K = ceil(0.2*N); %must be <=N (per column)
xmean0 = 0; xvar0 = 1;
SNRdB = 40;

emVampOpt.nit = 70;    %nit for EMVamp
emVampOpt.restartIter = 20;
emVampOpt.dampX = 0.85;
emVampOpt.EM_R1nit = 2; %EM iterations for gamma1
emVampOpt.tuner1pre = true;
emVampOpt.verbose = 60;
emVampOpt.checkConv = true; %check convergence of A and stop early
emVampOpt.tol = 1e-6;
emVampOpt.learnDict = true; %dictionary learning
emVampOpt.delayAupdate = 1; sty = '-';
emVampOpt.Atype = 'iid'; 
restart_tol = 1e-5;

%%--------- GENERATE REALIZATION ----------
%% Build the dictionary

%Draw randomly
A = randn(M,N);

%Normalize the columns
A = bsxfun(@rdivide,A,sqrt(sum(A.^2)));

%Dictionary error function
dictionary_error_function =...
    @(q) 20*log10(norm(A -...
    q*find_permutation(A,q),'fro')/norm(A,'fro'));
emVampOpt.error_functionA = dictionary_error_function;

%% Compute coefficient vectors
%Compute true vectors with exactly K non-zeroes in each
X = xmean0 + sqrt(xvar0)*randn(N,L);
for ll = 1:L
    yada = randperm(N);
    yada2 = zeros(N,1);
    yada2(yada(1:K)) = 1;
    X(:,ll) = X(:,ll) .* yada2;
end
coefficient_error_function = ...
    @(q) 20*log10(norm(X' - ...
    q'*find_permutation(X',q'),'fro')/norm(X','fro'));
emVampOpt.error_functionX = coefficient_error_function;

%% Form the output channel
%Compute noise free output
Z = A*X;

%Define the error function
error_function = @(qval) 20*log10(norm(qval - Z,'fro') / norm(Z,'fro'));
emVampOpt.error_function = error_function;

%Determine nuw
wvar = norm(reshape(Z,[],1))^2/M/L*10^(-SNRdB/10);

%Noisy output channel
Y = Z + sqrt(wvar)*randn(size(Z));

%%---------------------- BAdVAMP ------------------------
%% Declare estimators
gX = BGZeroMeanEstimIn(xvar0,K/N);
gX = MyVarEstimIn(gX,'nit',emVampOpt.EM_R1nit); %My auto-tuner

%Init Ahat: using the same as in BiGAMP
emVampOpt.Ahat0 = randn(M,N)/sqrt(M);

%Init R1:
emVampOpt.R10 = sqrt(50)*randn(N,L);
emVampOpt.gamma10 = 1e-10*ones(L,1);

%Init gamw: measurement noise variance
emVampOpt.gamw0 = M*L*(1+wvar)/norm(Y,'fro')^2; %1e-10;

%Optional
emVampOpt.N = N; emVampOpt.M = M; emVampOpt.L = L; emVampOpt.K = K;

vampHist.errA = [];
vampHist.errZ = [];
vampHist.errX1 = [];
vampHist.errX2 = [];
Ahat0 = emVampOpt.Ahat0;

Zhat = zeros(M,L,emVampOpt.restartIter);
start = tic;

%Run BAdVAMP
if emVampOpt.verbose
    disp(['dampX = ', num2str(emVampOpt.dampX),...
        ', EM_R1nit = ', num2str(emVampOpt.EM_R1nit),...
        ', tuner1pre = ', num2str(emVampOpt.tuner1pre)]);
    fprintf('--------------------------------------------------------------------------------\n');
    line1 = sprintf('%5s%6s    %s         %s        %s      %s      %s      \n','t','Itr','A_e','Z_e','X1_e','X2_e','SNR');
    fprintf(line1);
end


for t=1:emVampOpt.restartIter
    emVampOpt.numR = t; %restart number
    
    vampHist2 = BAdVAMP(gX,Y,emVampOpt);
    
    emVampOpt.Ahat0 = vampHist2.A(:,:,end);
    Z(:,:,t) = vampHist2.A(:,:,end)*vampHist2.X2(:,:,end);
    if emVampOpt.checkConv && t>=2 && norm(Y-Z(:,:,t),'fro')/norm(Y-Z(:,:,t-1))>10
        break;
    end
    
    vampHist.errA = [vampHist.errA vampHist2.errA];
    vampHist.errZ = [vampHist.errZ vampHist2.errZ];
    vampHist.errX1 = [vampHist.errX1 vampHist2.errX1];
    vampHist.errX2 = [vampHist.errX2 vampHist2.errX2];
    
    Ahat1 = vampHist2.A(:,:,end);
    if emVampOpt.checkConv && norm(Ahat1-Ahat0,'fro')/norm(Ahat1,'fro')<restart_tol
        break;
    end
    Ahat0 = Ahat1;
end
%disp(emVampOpt);
tEMVAMP = toc(start);

fprintf('.....')
fprintf('\n\nBAdVAMP: ');
fprintf('A_e=%3.4f  Z_e=%3.4f  X1_e=%3.4f  X2_e=%3.4f ',vampHist.errA(end), vampHist.errZ(end),...
    vampHist.errX1(end),vampHist.errX2(end));
fprintf('\n%d minutes and %2.2f seconds', floor(tEMVAMP/60), rem(tEMVAMP,60));
fprintf('\ntotal iterations: %d\n\n', size(vampHist.errX1,2));


%%---------------------- PLOT RESULTS -----------------
figure(1);
subplot(311);
plot(vampHist.errZ);
xlabel('iterations');ylabel('ZNMSE (dB)');
legend('EMVAMP');

figure(1);
subplot(312);
plot(vampHist.errA);
xlabel('iterations');ylabel('ANMSE (dB)');
legend('EMVAMP');

figure(1);
subplot(313);
plot(vampHist.errX1,'-g');
hold on; plot(vampHist.errX2,'-r');hold off;
xlabel('iterations'); ylabel('XNMSE (dB)');
legend('X1','X2');

