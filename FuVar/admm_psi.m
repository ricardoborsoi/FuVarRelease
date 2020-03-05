function [Psi] = admm_psi(Ym, Psi_init, A, Mh, lambda_m, lambda_1, lambda_2, R, maxiter_admm, epsilon_admm_abs, epsilon_admm_rel)
% =========================================================================
% Solve the optimization problem wrt. Psi
% 
% 
% Ym - M1 * M2 * L_l  : multispectral image
% Psi_init - L * P  : initialization of scaling factors
% A  - P * M1*M2  : abundance maps
% Mh - L * P  : endmember matrix (for hyperspectral image)
% lambda_m - regularization parameter
% lambda_1 - regularization parameter
% lambda_2 - regularization parameter
% R - L_l * L Spectral response function of MS instrument (converts HS to MS bands)
% 
% 
% Psi - L * P  : Matrix with endmember scaling factors (for Mm)
% 
% =========================================================================

% Force nonnegative Psi -------------
% flag_nonnegative = false;
flag_nonnegative = true;


% Yh - N1 * N2 * L
% Ym - M1 * M2 * L_l


% maxiter_admm = 50;
% epsilon_admm_abs = 1e-2;
% epsilon_admm_rel = 1e-2;

% maxiter_admm = 50;
% epsilon_admm_abs = 2e-3;
% epsilon_admm_rel = 2e-3;

% if nargin < 9
%     maxiter_admm = 50;
% end
% epsilon_admm_abs = 1e-4;
% epsilon_admm_rel = 1e-4;

if nargin < 9
    maxiter_admm = 50;
    epsilon_admm_abs = 1e-4;
    epsilon_admm_rel = 1e-4;
elseif nargin < 10
    epsilon_admm_abs = 1e-4;
    epsilon_admm_rel = 1e-4;
end


% maxiter_admm = 50;
% epsilon_admm_abs = 1e-5;
% epsilon_admm_rel = 1e-5;

% maxiter_admm = 200;
% epsilon_admm_abs = 1e-6;
% epsilon_admm_rel = 1e-6;

% maxiter_admm = 200;
% epsilon_admm_abs = 5e-6;
% epsilon_admm_rel = 5e-6;

% maxiter_admm = 200;
% epsilon_admm_abs = 1e-5;
% epsilon_admm_rel = 1e-5;

% maxiter_admm = 200;
% epsilon_admm_abs = 1e-6;
% epsilon_admm_rel = 1e-6;

% maxiter_admm = 200;
% epsilon_admm_abs = 1e-18;
% epsilon_admm_rel = 1e-18;




% number of endmembers
P = size(A,1);
%[N1,N2,L] = size(data_h); % dimensions of the data cube
%[M1,M2,L_l] = size(data_m); % dimensions of the data cube
[M1,M2,L_l] = size(Ym); % dimensions of the data cube
M = M1*M2; % number of HR pixels
L = size(Mh,1);

% Reorder images as matrices
Ym_r = reshape(Ym,M1*M2,L_l)';

verbose = true;


% auxiliary functins
Vec = @(x)(x(:));
inVec = @(x)(reshape(x,L,P));

% %%
% -------------------------------------------------------------------------
% Initialize the derivative convolution matrix
H_l = convmtx([-1 2 -1]',L);

% Fix the boundary conditions
H_l(1,1)     = 0;
H_l(end,end) = 0;
H_l(end-1,L) = 1;
H_l(2,1)     = 1;

nnzmax = 3*L*P;
Hl_mtx = sparse([],[],[],L*P,L*P, nnzmax);
for i=1:P
    Hl_mtx((1:L)+(i-1)*L, (1:L)+(i-1)*L) = sparse(H_l'*H_l);
end
Sysmtx1 = lambda_2 * Hl_mtx + lambda_1 * speye(L*P,L*P);
diagMhMh = sparse(diag(Vec(Mh.*Mh)));
% -------------------------------------------------------------------------
% %%





% barrier parameter of ADMM and related stuff
rho = zeros(maxiter_admm,1);
rho(1) = 1;
% rho(1) = 10;
% rho(1) = 100;
% rho(1) = 100000;

% tau_incr = 200;
% tau_decr = 1.2;
% nu = 1.2;

% tau_incr = 2.00;
% tau_decr = 2.00;
% nu = 100;

% tau_incr = 2;
% tau_decr = 2;
% nu = 10;

tau_incr = 1.2;
tau_decr = 1.2;
nu = 2;



% init variables
B   = Psi_init .* Mh;
U   = zeros(size(Mh));
Psi = Psi_init;




% initialize primal and dual variables
r_primal = zeros(maxiter_admm,1);
r_dual   = zeros(maxiter_admm,1);







% =========================================================================
% =========================================================================
% =========================================================================
% Start the Beast!

for iter = 1:maxiter_admm
    
    % store values from the previous iterations ---------------------------
	Psi_old = Psi;
	B_old   = B;
	U_old   = U;
    
    
    
    % min w.r.t. B --------------------------------------------------------
    % X = dlyap(A,B,C) solves the Sylvester equation AXB - X + C = 0
    A_sylv = - (lambda_m/rho(iter)) * (R'*R);
    B_sylv = A*A';
    C_sylv = -(- (lambda_m/rho(iter)) * (R'*Ym_r*A') - Psi.*Mh + U);
    B = dlyap(A_sylv, B_sylv, C_sylv);


    % min w.r.t. Psi ------------------------------------------------------
    vecPsi = (Sysmtx1 + rho(iter) * diagMhMh ) \ (lambda_1*ones(L*P,1) ...
        + rho(iter) * Vec(Mh.*B + Mh.*U));
    
    Psi = inVec(vecPsi);
    
    if flag_nonnegative
        Psi = max(Psi, 1e-6 * ones(size(Psi)));
    end
    
    
    % dual update U -------------------------------------------------------
    U = U + B - Mh .* Psi;
    
    

    
    % admm stopping criteria ----------------------------------------------
    % compute primal residue, ||Ax+Bz-c||
    r_primal(iter) = norm(B - Mh.*Psi,'fro')^2;
    r_primal(iter) = sqrt(r_primal(iter));
    
    
    % compute dual residue, ||rho*A'B(a-a_old)||
	deltaPsi = Psi-Psi_old;
	temp = norm(Mh .* deltaPsi,'fro')^2;
	r_dual(iter) = rho(iter) * sqrt(temp);
    
    
    
    % compute feasibility tolerances --------------------------------------
	% ||Ax||^2
	normAx = norm(B, 'fro')^2;
	%normAx = sqrt(normAx);
	
	
	% ||Bz||^2
	normBz = norm(Psi.*Mh, 'fro')^2;
	%normBz = sqrt(normBz);
	
	
	% ||rho A'u||^2
	normAtu = rho(iter)^2 * norm(U, 'fro')^2;
    
    
    
    
	
	% update epsilons ----------------------
	epsilon_primal = sqrt(L*P) * epsilon_admm_abs ...
	                 + epsilon_admm_rel * max(normAx, normBz);
	epsilon_dual   = sqrt(L*P) * epsilon_admm_abs ...
	                 + epsilon_admm_rel * normAtu;
	
	
                    
    
    
    
    %fprintf('---- Mighty ADMM at iteration %d ----\n',iter);
    if verbose
        rel_Psi_tmp = abs(norm(Psi,'fro')-norm(Psi_old,'fro'))/norm(Psi_old,'fro');
        rel_Psi = sum(rel_Psi_tmp);
        fprintf('iter %d of %d, rel_Psi = %f, primal = %f, eps_p = %d dual = %f, eps_d = %f, rho = %f \n',iter,maxiter_admm,rel_Psi,r_primal(iter),epsilon_primal,r_dual(iter),epsilon_dual,rho(iter));
    end
    
	% compute objective value on exit -------------------------------------
    if (iter > 1 && ((r_primal(iter) < epsilon_primal && r_dual(iter) < epsilon_dual))) || iter == maxiter_admm
        
        curObj = 0;
        
        break;
    end
    
    
    
    
	% update rho ----------------------------------------------------------
    if iter < maxiter_admm
        if r_primal(iter) > nu * r_dual(iter)
            rho(iter+1) = tau_incr*rho(iter);
        elseif r_dual(iter) > nu * r_primal(iter)
            rho(iter+1) = rho(iter)/tau_decr;
        else
            rho(iter+1) = rho(iter);
        end
    end
    
    % re-scale dual variables ---------------------------------------------
    rhoVariation = rho(iter+1) / rho(iter);
    U = U / rhoVariation;
    
    
end
