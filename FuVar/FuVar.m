function [Zh,Zm,A,Psi]=FuVar(Yh,Ym,A_init,M0,Psi_init,R,decimFactor,H_blur,...
    lambda_m,lambda_a,lambda_1,lambda_2,maxiter_anls)
% =========================================================================
% Code for hyperspectral and multispectral image fusion with spectral 
% variability. We use an ALS scheme to find a local minima of the cost function.
% 
% Yh - N1 * N2 * L  : hyperspectral image
% Ym - M1 * M2 * L_l  : multispectral image
% 
% A_init - P * M1*M2  : initialization for the abundance matrix estimation
% M0 - L * P  : spectral basis matrix
% 
% R - L_l * L Spectral response function of MS instrument (converts HS to MS bands)
% decimFactor - decimation factor from HR to LR
% H_blur - blurring mask of HS sensor PSF
% 
% lambda_i - regularization parameters
% maxiter_anls - number of ALS iterations
% =========================================================================




% initialize optional parameters
if nargin < 13
    maxiter_anls = 10;
end

epsilon_a   = 10^(-3);
epsilon_psi = 10^(-3);
% norm_sr = '1,1';
verbose = true;


% compute sizes and constants
P   = size(A_init,1); % number of endmembers
M1  = size(Ym,1);
M2  = size(Ym,2);
N1  = size(Yh,1);
N2  = size(Yh,2);
L   = size(Yh,3);
L_l = size(Ym,3);
M   = M1*M2;
N   = N1*N2;



%% initialize variables

% relative variations of the variables
ra   = zeros(maxiter_anls,1);
rpsi = zeros(maxiter_anls,1);

% initialize variables
A   = A_init; % initialize the abundance matrix
Zh  = imresize(Yh, decimFactor); % initialize Zh with spatial interpolation
Mh  = M0;
Psi = Psi_init;
Zh  = reshape(Zh,M1*M2,L)';





% % % Initialize Zm
% % Zm = zeros(M1,M2,L);
% % centerPos = zeros(L_l,1);
% % for ii=1:L_l
% %     [row,col] = find(R(ii,:) > 0);
% %     centerPos(ii) = mean(col);
% % end
% % 
% % for i=1:M1
% %     for j=1:M2
% %         if flag_normalizeInitZm
% % %             Zm(i,j,:) = interp(squeeze(Ym(i,j,:)) ./ sum(R')', L/L_l);
% %             Zm(i,j,:) = interp1(centerPos, squeeze(Ym(i,j,:)) ./ sum(R')', 1:L, 'linear', 'extrap');
% %         else
% % %             Zm(i,j,:) = interp(squeeze(Ym(i,j,:)), L/L_l);
% %             Zm(i,j,:) = interp1(centerPos, squeeze(Ym(i,j,:)), 1:L, 'linear', 'extrap');
% %         end
% %     end
% % end
% Zm = reshape(Zm,M1*M2,L)';
% % Order Ym as vector
% Ym_r = reshape(Ym,M1*M2,L_l)';




% TO REVISE
% optimization output
objective = zeros(maxiter_anls,1);
norm_fitting = zeros(maxiter_anls,1);
source_model = zeros(maxiter_anls,1);
TV_a = zeros(maxiter_anls,1);
smooth_psi = zeros(maxiter_anls,1);





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% optimization

for i = 1:maxiter_anls
    
    % store previous values of the variables
    Psi_old    = Psi;
    A_old_anls = A;
    
	
    
	%% A update ----------------------------------
	if verbose
        fprintf('updating A, Zh, Zm...\n')
    end
    
    [A,Zh,Zm] = admm_a(Yh, Ym, Zh, A, Mh, Psi, lambda_a, lambda_m, H_blur, decimFactor, R);
    
	if verbose
        fprintf('Done!\n')
    end
	
    
    
    %% Psi update ----------------------------------
    if verbose
        fprintf('updating Psi...\n')
    end
    
    [Psi] = admm_psi(Ym, Psi, A, Mh, lambda_m, lambda_1, lambda_2, R);
	
	if verbose
        fprintf('Done!\n')
    end
	
	
    
    
	%% ------------------------------------------------
    % residuals of the ANLS loops --------------------
	
    ra(i) = norm(A(:)-A_old_anls(:),2)/norm(A_old_anls(:),2);
    rpsi(i) = norm(Psi(:)-Psi_old(:),'fro')/(norm(Psi_old(:)));
    
    % compute objective function value
	objective(i) = 0; %norm_fitting(i) + lambda_s * source_model(i) + lambda_a' * TV_a(i,:)' + lambda_psi' * smooth_psi(i,:)';
        
	% display
    if verbose
        fprintf('\n---------------------- Outer iteration ------------------------------\n')
        fprintf('iteration %d of %d, ra = %f, rpsi = %f \n\n',i,maxiter_anls,ra(i),rpsi(i));
    end
	
    
    % termination test ---------------------------------------------------
    if ((ra(i) < epsilon_a)) && (rpsi(i) < epsilon_psi) 
        fprintf('\n\n')
        break;
    end
    
end

% recover optimization results

optim_struct.obj = objective;
optim_struct.fit = norm_fitting;
optim_struct.regul = source_model;
optim_struct.TVa = TV_a;
optim_struct.smoothpsi = smooth_psi;

end



