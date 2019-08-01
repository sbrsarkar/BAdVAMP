classdef MyVarEstimIn < EstimIn
    % MyVarEstimIn :  Include variance-tuning in input estimator 

    properties
        est;             % base estimator
        tuneDim = 'col'; % dimension on which the variances are to match
        nit = 4;         % number of EM iterations 
        rvarHist;        % history of rvar variances
        tol = 1e-4;      % tolerance in convergence of rvar
    end
    
    methods
        % Constructor
        function obj = MyVarEstimIn(est,varargin)
            if nargin ~= 0 % Allow nargin == 0 syntax
                obj.est = est;
                for i=1:2:length(varargin) 
                    obj.(varargin{i}) = varargin{i+1};
                end
            end
        end
        
        % Compute prior mean and variance
        function [xhat, xvar, valInit] = estimInit(obj)
            [xhat, xvar, valInit] = obj.est.estimInit;
        end
        
        % Compute posterior mean and variance from Gaussian estimate
        function [xhat, xvar, rvar] = estim(obj, rhat, rvar)
            if size(rvar,2)==1 && size(rvar,1)>1
                rvar = rvar';
            end
            assert(size(rvar,1)==1,'SUB#: rvar should be a row vector');

            [N,~] = size(rhat);
            rvar1 = rvar;

            [xhat, xvar] = obj.est.estim(rhat,repmat(rvar,[N 1]));
            for it = 1:obj.nit
                rvar = mean(abs(rhat-xhat).^2 + xvar);    
                                
                [xhat,xvar] = obj.est.estim(rhat,repmat(rvar,[N 1]));

                if norm(rvar-rvar1)/norm(rvar)<obj.tol
                    break;
                end
                rvar1 = rvar;
            end
        end
        
        % Generate random samples
        function x = genRand(obj, nx)
            x = obj.est.genRand(nx);
        end
        
    end
    
end

