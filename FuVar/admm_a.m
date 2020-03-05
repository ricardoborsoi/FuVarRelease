function [A,Zh,Zm] = admm_a(Yh, Ym, Zh_init, A_init, Mh, Psi, lambda_a, lambda_m, H_blur, decimFactor, R, ...
    maxiter_admm,epsilon_admm_abs,epsilon_admm_rel)
% =========================================================================
% Solve the optimization problem wrt. A, Zm and Zh
% 
% 
% Yh - N1 * N2 * L  : hyperspectral image
% Ym - M1 * M2 * L_l  : multispectral image
% Zh_init - L * M1*M2  : HR HS image initialization
% A_init - P * M1*M2  : initialization for the abundance maps
% Mh - L * P  : endmember matrix (for hyperspectral image)
% Psi - L * P  : Matrix with endmember scaling factors (for Mm)
% lambda_a - regularization parameter
% lambda_m - regularization parameter
% decimFactor - decimation factor from HR to LR
% H_blur - blurring mask of HS sensor PSF
% R - L_l * L Spectral response function of MS instrument (converts HS to MS bands)
% 
% 
% A - P * M1*M2  : abundance matrix
% Zh - L * M1*M2  : HR image (HS)
% Zm - L * M1*M2  : HR image (MS)
% 
% =========================================================================


% enforce sum-to-one constraint in the abundances if true 
flag_sumtoone = false;



if nargin < 12
    maxiter_admm = 50;
    epsilon_admm_abs = 1e-4;
    epsilon_admm_rel = 1e-4;
elseif nargin < 13
    epsilon_admm_abs = 1e-4;
    epsilon_admm_rel = 1e-4;
end




% number of endmembers
P = size(A_init,1);
%[N1,N2,L] = size(data_h); % dimensions of the data cube
%[M1,M2,L_l] = size(data_m); % dimensions of the data cube
[N1,N2,L]   = size(Yh); % dimensions of the data cube
[M1,M2,L_l] = size(Ym); % dimensions of the data cube
N = N1*N2; % number of LR pixels
M = M1*M2; % number of HR pixels

% Reorder images as matrices
Yh_r = reshape(Yh,N1*N2,L)';
Ym_r = reshape(Ym,M1*M2,L_l)';

verbose = true;
norm_sr = '2,1'; % '2,1'or '1,1'
% norm_sr = '1,1';



% ==> get M, N, M1 M2, N1, N2

% =========================================================================
% =========================================================================
% =========================================================================
% Varuables and methods for Drumetz's part of the code

% build handlers and necessary stuff

% forward first order horizontal difference operator
%FDh = zeros(m,n);
FDh = zeros(M1,M2);
FDh(end,1) = -1;
FDh(end,end) = 1;
FDh = fft2(FDh);
FDhC = conj(FDh);

% forward first order vertical difference operator
%FDv = zeros(m,n);
FDv = zeros(M1,M2);
FDv(1,end) = -1;
FDv(end,end) = 1;
FDv = fft2(FDv);
FDvC = conj(FDv);

% barrier parameter of ADMM and related stuff
rho = zeros(maxiter_admm,1);
rho(1) = 1;

tau_incr = 1.2;
tau_decr = 1.2;
nu = 2;

% auxiliary functions

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define a circular convolution (the same for all bands) accepting a
% matrix  and returning a matrix

% ConvC = @(X,FK)  reshape(real(ifft2(fft2(reshape(X', m,n,P)).*repmat(FK,[1,1,P]))), m*n,P)';
ConvC = @(X,FK)  reshape(real(ifft2(fft2(reshape(X', M1,M2,P)).*repmat(FK,[1,1,P]))), M1*M2,P)';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% convert matrix to image
%conv2im  = @(A)  reshape(A',m,n,P);
conv2im  = @(A)  reshape(A',M1,M2,P);

% convert image to matrix
%conv2mat = @(A)  reshape(A,m*n,P)';
conv2mat = @(A)  reshape(A,M1*M2,P)';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
% teeemp = conv2im(A_init);
% figure, imagesc(teeemp(:,:,1)), pause(0.1)

%%


% init variables

A  = A_init;
mu = zeros(1,N);



B2 = Zh_init;
B1 = spatialDegrade(B2,M1,M2,decimFactor,H_blur);
B3 = A;
B4 = ConvC(A,FDh);
B5 = ConvC(A,FDv);
B6 = A;

U1 = zeros(size(B1'));
U2 = zeros(size(B2));
U3 = zeros(size(A));
U4 = zeros(size(A));
U5 = zeros(size(A));
U6 = zeros(size(A));





% initialize primal and dual variables
r_primal = zeros(maxiter_admm,1);
r_dual   = zeros(maxiter_admm,1);







% =========================================================================
% =========================================================================
% =========================================================================
% Start the Beast!

for iter = 1:maxiter_admm
    
    % store values from the previous iterations ---------------------------
	A_old  = A;
	B1_old = B1;
	B2_old = B2;
	B3_old = B3;
	B4_old = B4;
	B5_old = B5;
	B6_old = B6;
	U1_old = U1;
	U2_old = U2;
	U3_old = U3;
	U4_old = U4;
	U5_old = U5;
	U6_old = U6;
	

	
    
    
    % min w.r.t. A and mu -------------------------------------------------
    if flag_sumtoone
        % enforce sum-to-unit constraint -------------------------
        Omega1  = lambda_m*(R*(Psi.*Mh))'*(R*(Psi.*Mh)) + rho(iter)*(Mh'*Mh) + 2*rho(iter)*eye(P);
        tempMtx = [Omega1, ones(P,1); ones(1,P) 0];
        tempMtx = inv(tempMtx);
        parfor nn=1:M
            Omega2 = rho(iter) * ( Mh'*(B2(:,nn)+U2(:,nn)) + (B3(:,nn)+U3(:,nn)) ...
                + (B6(:,nn)+U6(:,nn)) ) + lambda_m * (R*(Psi.*Mh))' * Ym_r(:,nn);

            tempVec = [Omega2; 1];
            tempRes = tempMtx * tempVec;

            % store results
            A(:,nn) = tempRes(1:end-1);
            mu(nn) = tempRes(end);
        end
        
    else
        % no sum-to-unit constraint ------------------------------
        Omega1  = lambda_m*(R*(Psi.*Mh))'*(R*(Psi.*Mh)) + rho(iter)*(Mh'*Mh) + 2*rho(iter)*eye(P);
        tempMtx = inv(Omega1);
        parfor nn=1:M
            Omega2 = rho(iter) * ( Mh'*(B2(:,nn)+U2(:,nn)) + (B3(:,nn)+U3(:,nn)) ...
                + (B6(:,nn)+U6(:,nn)) ) + lambda_m * (R*(Psi.*Mh))' * Ym_r(:,nn);

            tempRes = tempMtx * Omega2;

            % store results
            A(:,nn) = tempRes;
        end
    end
    
    
    
    % min w.r.t. B1 -------------------------------------------------------
    tempDegB2 = spatialDegrade(B2,M1,M2,decimFactor,H_blur);
    U1til = U1';
    for n=1:N
        B1(:,n) = (1/(1+rho(iter))) * (Yh_r(:,n) + rho(iter) * tempDegB2(:,n) ...
            + rho(iter) * U1til(:,n) );
    end
    
%     % override
%     B1 = Yh_r;
    

    
    % min w.r.t. B2 -------------------------------------------------------
    afun = @(zin)afunSpatial(zin,M1,M2,decimFactor,H_blur);
    Omega3 = (Mh*A)';
	
	% Convert to image to degrade (note that dims(U1)=dims(B1')=dims(Yh'))
	B1_im = reshape(B1',N1,N2,L); % conv2im  = @(A)  reshape(A',m,n,P);
	U1_im = reshape((U1')',N1,N2,L);
	
	% Apply FDx to these signals (the output is already in matrix form, L x M)
	B1_tdeg = spatialDegradeTranspose(B1_im,N1,N2,decimFactor,H_blur);
	U1_tdeg = spatialDegradeTranspose(U1_im,N1,N2,decimFactor,H_blur);
	
	% Transpose to M*L dims
    B1_tdeg = B1_tdeg';
    U1_tdeg = U1_tdeg';
    
	% Solve linear system for each band
	cg_tol   = 1e-6;
	cg_maxit = 30;
	%Z_temp = zeros(M1*M2,L);
    parfor l=1:L
        bvec    = Omega3(:,l) - U2(l,:)' + B1_tdeg(:,l) - U1_tdeg(:,l);
        xl_init = B2(l,:)'; % compute an initialization
        [zvec,flagout] = pcg(afun,bvec,cg_tol,cg_maxit,[],[],xl_init);
		%Z_temp(:,l) = zvec;
		B2(l,:) = zvec';
    end
    %Zh = Z_temp';
	
% 	% override
%     Zh = Yh_r;
    
	
    

    
    % min w.r.t. B3 -------------------------------------------------------
    % reshape necessary variables into images
    A_im  = conv2im(A);
    B4_im = conv2im(B4);
    B5_im = conv2im(B5);
    U3_im = conv2im(U3);
    U4_im = conv2im(U4);
    U5_im = conv2im(U5);

    % update in the Fourier domain
    for p = 1:P
        second_term_in_spectral_domain = fft2(squeeze(A_im(:,:,p)) - squeeze(U3_im(:,:,p))) + ...
            fft2(squeeze((B4_im(:,:,p)) + squeeze(U4_im(:,:,p)))).*FDhC + ...
            fft2(squeeze((B5_im(:,:,p)) + squeeze(U5_im(:,:,p)))).*FDvC;
        B3_im(:,:,p) = real(ifft2((second_term_in_spectral_domain)./(ones(M1,M2) + abs(FDh).^2+ abs(FDv).^2))); % this is the convolution performed in the fourier domain, done band by band.
    end

    % convert back necessary variables into matrices
    B3   = conv2mat(B3_im);
    Hvv2 = ConvC(B3,FDv);
    Hhv2 = ConvC(B3,FDh);
    
    
    
    
    % min w.r.t. B4 and B5 ------------------------------------------------
    if strcmp(norm_sr,'2,1')
        B4 = vector_soft_col((Hhv2-U4),lambda_a/rho(iter)); % l21 norm
        B5 = vector_soft_col((Hvv2-U5),lambda_a/rho(iter));
    elseif strcmp(norm_sr,'1,1')
        B4 = soft((Hhv2-U4),lambda_a/rho(iter)); % l11 norm
        B5 = soft((Hvv2-U5),lambda_a/rho(iter));
    end
    
	
    
    
    % min w.r.t. B6 -------------------------------------------------------
    B6 = max(A - U6, zeros(size(A)));
    
    
    
    
	
	
	% dual update U -------------------------------------------------------
    U1 = U1 - B1' + (spatialDegrade(B2,M1,M2,decimFactor,H_blur))';  
    U2 = U2 + B2 - Mh * A;
	U3 = U3 + B3 - A;
	U4 = U4 + B4 - Hhv2; % Hhv2 = ConvC(B3,FDh);
	U5 = U5 + B5 - Hvv2; % Hvv2 = ConvC(B3,FDv);
	U6 = U6 + B6 - A;
	
	
    
	
	
	% admm stopping criteria ----------------------------------------------
    % compute primal residue, ||Ax+Bz-c||
    r_primal(iter) = 0;
    r_primal(iter) = r_primal(iter) + norm(-B1' + (spatialDegrade(B2,M1,M2,decimFactor,H_blur))','fro')^2;
	r_primal(iter) = r_primal(iter) + norm(B2 - Mh * A,'fro')^2;
    r_primal(iter) = r_primal(iter) + norm(B3 - A,'fro')^2;
    r_primal(iter) = r_primal(iter) + norm(B4 - ConvC(B3,FDh),'fro')^2;
    r_primal(iter) = r_primal(iter) + norm(B5 - ConvC(B3,FDv),'fro')^2;
    r_primal(iter) = r_primal(iter) + norm(B6 - A,'fro')^2;
    r_primal(iter) = sqrt(r_primal(iter));
    
    
	% compute dual residue, ||rho*A'B(a-a_old)||
	deltaA = A-A_old;
	temp = 0;
    temp = temp + norm(Mh * deltaA)^2;
	temp = temp + 2 * norm(deltaA,'fro')^2;
	r_dual(iter) = rho(iter) * sqrt(temp);
	
	
	
	% compute feasibility tolerances --------------------------------------
	% ||Ax||^2
	normAx = 0;
	normAx = normAx + norm(-B1' + (spatialDegrade(B2,M1,M2,decimFactor,H_blur))', 'fro')^2;
	normAx = normAx + norm(B2, 'fro')^2;
	normAx = normAx + norm(B3, 'fro')^2;
	normAx = normAx + norm(B4 - Hhv2, 'fro')^2;
	normAx = normAx + norm(B5 - Hvv2, 'fro')^2;
	normAx = normAx + norm(B6, 'fro')^2;
	%normAx = sqrt(normAx);
	
	
	
	% ||Bz||^2
	normBz = 0;
    normBz = normBz + norm(Mh * A)^2;
	normBz = normBz + 2 * norm(A,'fro')^2;
	%normBz = sqrt(normBz);
	
	
	
	% ||rho A'u||^2
	normAtu = 0;
    normAtu = normAtu + norm(U1, 'fro')^2;
	U1_im      = reshape((U1')',N1,N2,L);
	U1_tdeg_im = spatialDegradeTranspose(U1_im,N1,N2,decimFactor,H_blur);
	U1_tdeg    = reshape(U1_tdeg_im,M1*M2,L)';
	normAtu = normAtu + norm(U1_tdeg+U2, 'fro')^2;
	normAtu = normAtu + norm(U3-ConvC(U4,FDh)-ConvC(U5,FDv), 'fro')^2;
    normAtu = normAtu + norm(U4, 'fro')^2;
	normAtu = normAtu + norm(U5, 'fro')^2;
	normAtu = normAtu + norm(U6, 'fro')^2;
	normAtu = rho(iter)^2 * normAtu;
    
	
	
	
	
	% update epsilons ----------------------
	epsilon_primal = sqrt(L*N + L*M + 4*P*M) * epsilon_admm_abs ...
	                 + epsilon_admm_rel * max(normAx, normBz);
	epsilon_dual   = sqrt(L*N + L*M + 4*P*M) * epsilon_admm_abs ...
	                 + epsilon_admm_rel * normAtu;
	
	
	
                 
                 
    %fprintf('---- Mighty ADMM at iteration %d ----\n',iter);
    if verbose
        rel_A_tmp = abs(norm(A,'fro')-norm(A_old,'fro'))/norm(A_old,'fro');
        rel_A = sum(rel_A_tmp);
        fprintf('iter %d of %d, rel_A = %f, primal = %f, eps_p = %d dual = %f, eps_d = %f, rho = %f \n',iter,maxiter_admm,rel_A,r_primal(iter),epsilon_primal,r_dual(iter),epsilon_dual,rho(iter));
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
    U1 = U1 / rhoVariation;
    U2 = U2 / rhoVariation;
    U3 = U3 / rhoVariation;
    U4 = U4 / rhoVariation;
    U5 = U5 / rhoVariation;
    U6 = U6 / rhoVariation;
	
	
    
end % -- end ADMM iteration -----------------------------------------------

% Attribute output variables ----------------------------------------------
Zh = B2;
Zm = (Psi .* Mh) * A;

end % -- end main function ------------------------------------------------












% % convert matrix to image
% conv2im  = @(A)  reshape(A',m,n,P);
% % convert image to matrix
% conv2mat = @(A)  reshape(A,m*n,P)';

% =========================================================================
function tempDegZh = spatialDegrade(Zh,M1,M2,decimFactor,H_blur)
% Performs spatial degradation (blur and downsample)
% Zh - M1 x M2 x L, or L x M1*M2
%
%   This operation corresponds to either x*F*D when x is a row vector image,
%   or to D'*F'*x when x is a column vector image 
%   (generalizes trivially to matrices).

% M1/decimFactor = N1,  M2/decimFactor = N2

N1 = M1/decimFactor;
N2 = M2/decimFactor;

if sum(length(size(Zh))) == 2
    L = size(Zh,1);
    % Convert Zh to image (cube)
    Zh_im = reshape(Zh',M1,M2,L);
else
    L = size(Zh,3);
    Zh_im = Zh;
end

tempDegZh_im = zeros(N1, N2, L);
% degrade each band
for l=1:L
    temp = Zh_im(:,:,l);
    temp = imfilter(temp, H_blur, 'symmetric', 'same');
    temp = temp(1:decimFactor:end, 1:decimFactor:end);
    
    tempDegZh_im(:,:,l) = temp;
end

% convert back to matrix
tempDegZh = reshape(tempDegZh_im, N1*N2, L)';
end


% =========================================================================
function tempDegT_Yh = spatialDegradeTranspose(Yh,N1,N2,decimFactor,H_blur)
% Performs the transpose of the spatial degradation (upsample and blur with flipped kernel)
% Yh - N1 x N2 x L, or L x N1*N2
%   This operation corresponds to either x*D'*F' when x is a row vector image,
%   or to F*D*x when x is a column vector image 
%   (generalizes trivially to matrices).

% M1/decimFactor = N1,  M2/decimFactor = N2

M1 = N1*decimFactor;
M2 = N2*decimFactor;

if sum(length(size(Yh))) == 2
    L = size(Yh,1);
    % Convert to image (cube)
    Yh_im = reshape(Yh',N1,N2,L);
else
    L = size(Yh,3);
    Yh_im = Yh;
end

tempDegT_Yh_im = zeros(M1, M2, L);
% Perform transpose degradation to each band
for l=1:L
    temp = zeros(M1,M2);
    temp(1:decimFactor:end, 1:decimFactor:end) = Yh_im(:,:,l);
    temp = imfilter(temp(end:-1:1,end:-1:1), H_blur, 'symmetric', 'same');
    temp = temp(end:-1:1,end:-1:1);
    
    tempDegT_Yh_im(:,:,l) = temp;
end

% convert back to matrix
tempDegT_Yh = reshape(tempDegT_Yh_im, M1*M2, L)';
end




% =========================================================================
function outp = afunSpatial(zin,M1,M2,decimFactor,H_blur)
% Performs the operation (I+FDD'F') * zin, where zin is an image ordered as
% a column-vector (of a single band), and in this case the operation FDD'F'
% corresponds to degTranspose(degrade(zin)).

% zin - M1 * M2 x 1 (vector)

% reshape as image
Zin = reshape(zin,M1,M2);

% Blur and downsample
temp = imfilter(Zin, H_blur, 'symmetric', 'same');
temp = temp(1:decimFactor:end, 1:decimFactor:end);

temp2 = zeros(M1,M2);
temp2(1:decimFactor:end, 1:decimFactor:end) = temp;
temp2 = imfilter(temp2(end:-1:1,end:-1:1), H_blur, 'symmetric', 'same');
temp2 = temp2(end:-1:1,end:-1:1);

% Convert back to vector
outp = reshape(temp2,M1*M2,1);
% Sum the input to account for the identity
outp = outp + zin;
end


