% =========================================================================
% This code corresponds to the example with the Paris image contained in
% the paper referenced below:
% 
% -------------------------------------------------------------------------
%   R.A. Borsoi, T. Imbiriba, J.C.M. Bermudez.
%    "Super-Resolution for Hyperspectral and Multispectral Image Fusion 
%     Accounting for Seasonal Spectral Variability"
%   IEEE Transactions on Image Processing, 2019.
% 
% 
% =========================================================================

clear all
close all
clc

addpath(genpath('utils'))
addpath(genpath('FuVar'))

clus = gcp('nocreate'); % If no pool, do not create new one.
if isempty(clus)
    c = parcluster('local');
    c.NumWorkers = 1; 5;
    parpool(c, c.NumWorkers);
end


%%
rng(10, 'twister') 

% load images
load('DATA/paris_data_hs_ms.mat')

% load SRF (estimated as in HySure paper)
load('DATA/R_HysureParis.mat') % normalized SRF

MSp = zeros(size(MSim));
for i=1:size(MSim,3)
    x = MSim(:,:,i);
    xmax  = quantile(x(:),0.999);
    % Normalize to 1
    x = x/xmax;
    MSp(:,:,i) = x;
end
MSI = MSp;

HSp = zeros(size(HSim));
for i=1:size(HSim,3)
    x = HSim(:,:,i);
    xmax  = quantile(x(:),0.999);
    %normalize to 1
    x = x/xmax;
    HSp(:,:,i) = x;
end
HSI = HSp;



% Signal to noise ratio of HS and MS images
SNR_h = 30;
SNR_m = 40;


decimFactor = 3;
H_blur = conv2(ones(decimFactor)/(decimFactor^2), fspecial('gaussian',3,1));


L   = size(HSI,3);
L_l = size(MSI,3);
M1  = size(HSI,1);
M2  = size(HSI,2);
N1  = M1/decimFactor;
N2  = M2/decimFactor;


% Get MSI and HR HSI
Ym = MSI;
Zh = denoising(HSI);


% --------------------
% Reorder as an image, blur and downsample to get HSI
Yh = zeros(N1, N2, L);
% degrade each band
for l=1:L
    temp = Zh(:,:,l);
    temp = imfilter(temp, H_blur, 'symmetric', 'same');
    temp = temp(1:decimFactor:end, 1:decimFactor:end);
    Yh(:,:,l) = temp;
end
Zh_th =  Zh;

% add noise
Yh = Yh + sqrt(sum((Yh(:)).^2)/N1/N2/L/10^(SNR_h/10)) * randn(size(Yh));
Ym = Ym + sqrt(sum((Ym(:)).^2)/M1/M2/L_l/10^(SNR_m/10)) * randn(size(Ym));








% % ----------------------------------------------
% Extract spectral basis
P = 30;
data_r = (reshape(Yh,N1*N2,L)')';

try 
	M0 = vca(data_r','Endmembers',P,'verbose','off');
catch
    try
        M0 = vca(data_r',P);
    catch
        error('Hum... seems like VCA is not working!')
    end
end



%% HySure

basis_type = 'VCA';
lambda_phi = 5e-4;
lambda_m   = 1;
P_hysure   = P;
try 
	M0_hysure = vca(data_r','Endmembers',P_hysure,'verbose','off');
catch
    try
        M0_hysure = vca(data_r',P_hysure);
    catch
        error('Hum... seems like VCA is not working!')
    end
end


Yhim = Yh;
Ymim = Ym;
downsamp_factor = decimFactor;
B_est = zeros(M1,M2); 
B_est(1:size(H_blur,1), 1:size(H_blur,2)) = H_blur;
% B_est(round(M1/2)-3:round(M1/2)+3, round(M2/2)-3:round(M2/2)+3) = H_blur;
% R_est = R;
shift = 2; % ???

R_est = R;

tic
Z_hysure = data_fusion_mod(Yhim, Ymim, downsamp_factor, R_est, B_est, P_hysure, basis_type, lambda_phi, lambda_m, shift, M0_hysure);
time_hysure = toc;


%% CNMF

% Chose a priori or estimated SRF
R_est = R;

HSI = Yh;
MSI = Ym;
P_cnmf = 30; 40;

tic
Z_cnmf = CNMF_fusion2(HSI,MSI,R_est,P_cnmf);
time_cnmf = toc;


%% GLPHS

hs_glphs = Yh;
ms_glphs = Ym;
ratio_glphs = decimFactor;
mode_glphs = 2;

tic
Z_glphs = MTF_GLP_wrapper(hs_glphs,ms_glphs,ratio_glphs,mode_glphs);
Z_glphs = denoising(Z_glphs);
time_glphs = toc;


%% FuVar algorithm

disp('FuVar...')

% initialization of the spatial coefficients
A_FCLSU = FCLSU(data_r',M0)';
A_init = reshape(A_FCLSU',N1,N2,P);
A_init = imresize(A_init, decimFactor);
A_init = reshape(A_init,M1*M2,P)';

lambda_m = 1;
lambda_a = 1e-4; 
lambda_1 = 0.01;
lambda_2 = 10000;

R_srr = R;

tic
Psi_init = ones(L,P);
[Zh,Zm,A,Psi]=FuVar(Yh,Ym,A_init,M0,Psi_init,R_srr,decimFactor,H_blur,...
    lambda_m,lambda_a,lambda_1,lambda_2);
time_srr = toc;

Zh_srr = reshape(Zh',M1,M2,L);
Zm_srr = reshape(Zm',M1,M2,L);



%% ------------------------------------
% Reorder data and compute errors

fprintf('\n\n\n--------------------------------------------------------\n')
fprintf('\n\n\n------------------------ times -------------------------\n')
fprintf('Time SRR........: %f \n',time_srr)
fprintf('Time HySure.....: %f \n',time_hysure)
fprintf('Time GLPHS......: %f \n',time_glphs)
fprintf('Time CNMF.......: %f \n',time_cnmf)


addpath Quality_Indices
qualIdx_srr    = QualityIndices_mod(Zh_srr, Zh_th, decimFactor);
qualIdx_hysure = QualityIndices_mod(Z_hysure, Zh_th, decimFactor);
qualIdx_glphs  = QualityIndices_mod(Z_glphs, Zh_th, decimFactor);
qualIdx_cnmf   = QualityIndices_mod(Z_cnmf, Zh_th, decimFactor);

fprintf('\n\n\n------------------------results ------------------------\n')
fprintf('\n------------------------ PSNR (> better) ---------------\n')
fprintf('Img rec PSNR SRR HS........: %f \n',qualIdx_srr.psnr)
fprintf('Img rec PSNR HySure HS.....: %f \n',qualIdx_hysure.psnr)
fprintf('Img rec PSNR GLPHS HS......: %f \n',qualIdx_glphs.psnr)
fprintf('Img rec PSNR CNMF HS.......: %f \n',qualIdx_cnmf.psnr)

fprintf('\n------------------------ SAM (< better) ----------------\n')
fprintf('Img rec SAM SRR HS........: %f \n',qualIdx_srr.sam)
fprintf('Img rec SAM HySure HS.....: %f \n',qualIdx_hysure.sam)
fprintf('Img rec SAM GLPHS HS......: %f \n',qualIdx_glphs.sam)
fprintf('Img rec SAM CNMF HS.......: %f \n',qualIdx_cnmf.sam)

fprintf('\n------------------------ ERGAS (< better) --------------\n')
fprintf('Img rec ERGAS SRR HS........: %f \n',qualIdx_srr.ergas)
fprintf('Img rec ERGAS HySure HS.....: %f \n',qualIdx_hysure.ergas)
fprintf('Img rec ERGAS GLPHS HS......: %f \n',qualIdx_glphs.ergas)
fprintf('Img rec ERGAS CNMF HS.......: %f \n',qualIdx_cnmf.ergas)

fprintf('\n------------------------ UIQI (> better)----------------\n')
fprintf('Img rec UIQI SRR MS.......: %f \n',qualIdx_srr.uiqi )
fprintf('Img rec UIQI HySure MS....: %f \n',qualIdx_hysure.uiqi )
fprintf('Img rec UIQI GLPHS MS.....: %f \n',qualIdx_glphs.uiqi )
fprintf('Img rec UIQI CNMF MS......: %f \n',qualIdx_cnmf.uiqi )






%%

figure;
[ha, pos] = tight_subplot(1, 5, 0.01, 0.1, 0.1);

% vbands = [57,30,20]; % RGB bands
vbands = [24,14,3]; % RGB bands OK visible
% vbands = [115,81,38]; % RGB bands OK IR
ddata = Zh_th;
rgbim(:,:,1) = imadjust(rescale(ddata(:,:,vbands(1)),1));
rgbim(:,:,2) = imadjust(rescale(ddata(:,:,vbands(2)),1));
rgbim(:,:,3) = imadjust(rescale(ddata(:,:,vbands(3)),1));
axes(ha(1)); imshow(rgbim); % display RGB image
title('Reference','interpreter','latex')

ddata = Z_hysure;
rgbim(:,:,1) = imadjust(rescale(ddata(:,:,vbands(1)),1));
rgbim(:,:,2) = imadjust(rescale(ddata(:,:,vbands(2)),1));
rgbim(:,:,3) = imadjust(rescale(ddata(:,:,vbands(3)),1));
axes(ha(2)); imshow(rgbim); % display RGB image
title('HySure','interpreter','latex')

ddata = Z_glphs;
rgbim(:,:,1) = imadjust(rescale(ddata(:,:,vbands(1)),1));
rgbim(:,:,2) = imadjust(rescale(ddata(:,:,vbands(2)),1));
rgbim(:,:,3) = imadjust(rescale(ddata(:,:,vbands(3)),1));
axes(ha(3)); imshow(rgbim); % display RGB image
title('GLP-HS','interpreter','latex')

ddata = Z_cnmf;
rgbim(:,:,1) = imadjust(rescale(ddata(:,:,vbands(1)),1));
rgbim(:,:,2) = imadjust(rescale(ddata(:,:,vbands(2)),1));
rgbim(:,:,3) = imadjust(rescale(ddata(:,:,vbands(3)),1));
axes(ha(4)); imshow(rgbim); % display RGB image
title('CNMF','interpreter','latex')

ddata = Zh_srr;
rgbim(:,:,1) = imadjust(rescale(ddata(:,:,vbands(1)),1));
rgbim(:,:,2) = imadjust(rescale(ddata(:,:,vbands(2)),1));
rgbim(:,:,3) = imadjust(rescale(ddata(:,:,vbands(3)),1));
axes(ha(5)); imshow(rgbim); % display RGB image
title('FuVar','interpreter','latex')

% print(['examples/recIm_ex2_Paris_vis'],'-dpdf')



figure;
[ha, pos] = tight_subplot(1, 5, 0.01, 0.1, 0.1);

% vbands = [57,30,20]; % RGB bands
% vbands = [24,14,3]; % RGB bands OK visible
vbands = [115,81,38]; % RGB bands OK IR
ddata = Zh_th;
rgbim(:,:,1) = imadjust(rescale(ddata(:,:,vbands(1)),1));
rgbim(:,:,2) = imadjust(rescale(ddata(:,:,vbands(2)),1));
rgbim(:,:,3) = imadjust(rescale(ddata(:,:,vbands(3)),1));
axes(ha(1)); imshow(rgbim); % display RGB image
title('Reference','interpreter','latex')

ddata = Z_hysure;
rgbim(:,:,1) = imadjust(rescale(ddata(:,:,vbands(1)),1));
rgbim(:,:,2) = imadjust(rescale(ddata(:,:,vbands(2)),1));
rgbim(:,:,3) = imadjust(rescale(ddata(:,:,vbands(3)),1));
axes(ha(2)); imshow(rgbim); % display RGB image
title('HySure','interpreter','latex')

ddata = Z_glphs;
rgbim(:,:,1) = imadjust(rescale(ddata(:,:,vbands(1)),1));
rgbim(:,:,2) = imadjust(rescale(ddata(:,:,vbands(2)),1));
rgbim(:,:,3) = imadjust(rescale(ddata(:,:,vbands(3)),1));
axes(ha(3)); imshow(rgbim); % display RGB image
title('GLP-HS','interpreter','latex')

ddata = Z_cnmf;
rgbim(:,:,1) = imadjust(rescale(ddata(:,:,vbands(1)),1));
rgbim(:,:,2) = imadjust(rescale(ddata(:,:,vbands(2)),1));
rgbim(:,:,3) = imadjust(rescale(ddata(:,:,vbands(3)),1));
axes(ha(4)); imshow(rgbim); % display RGB image
title('CNMF','interpreter','latex')

ddata = Zh_srr;
rgbim(:,:,1) = imadjust(rescale(ddata(:,:,vbands(1)),1));
rgbim(:,:,2) = imadjust(rescale(ddata(:,:,vbands(2)),1));
rgbim(:,:,3) = imadjust(rescale(ddata(:,:,vbands(3)),1));
axes(ha(5)); imshow(rgbim); % display RGB image
title('FuVar','interpreter','latex')

% print(['examples/recIm_ex2_Paris_IR'],'-dpdf')






% ---------------------------------
% repeat in a single figure
figure;
[ha, pos] = tight_subplot(2, 5, [0.01 0.01], 0.3, 0.1);

% vbands = [57,30,20]; % RGB bands
vbands = [24,14,3]; % RGB bands OK visible
ddata = Zh_th;
rgbim(:,:,1) = imadjust(rescale(ddata(:,:,vbands(1)),1));
rgbim(:,:,2) = imadjust(rescale(ddata(:,:,vbands(2)),1));
rgbim(:,:,3) = imadjust(rescale(ddata(:,:,vbands(3)),1));
axes(ha(1)); imshow(rgbim); % display RGB image
axis square, set(gca,'xtick',[]), set(gca,'xticklabel',[]), set(gca,'ytick',[]), set(gca,'yticklabel',[])
title('Reference','interpreter','latex')

ddata = Z_hysure;
rgbim(:,:,1) = imadjust(rescale(ddata(:,:,vbands(1)),1));
rgbim(:,:,2) = imadjust(rescale(ddata(:,:,vbands(2)),1));
rgbim(:,:,3) = imadjust(rescale(ddata(:,:,vbands(3)),1));
axes(ha(2)); imshow(rgbim); % display RGB image
axis square, set(gca,'xtick',[]), set(gca,'xticklabel',[]), set(gca,'ytick',[]), set(gca,'yticklabel',[])
title('HySure','interpreter','latex')

ddata = Z_glphs;
rgbim(:,:,1) = imadjust(rescale(ddata(:,:,vbands(1)),1));
rgbim(:,:,2) = imadjust(rescale(ddata(:,:,vbands(2)),1));
rgbim(:,:,3) = imadjust(rescale(ddata(:,:,vbands(3)),1));
axes(ha(3)); imshow(rgbim); % display RGB image
axis square, set(gca,'xtick',[]), set(gca,'xticklabel',[]), set(gca,'ytick',[]), set(gca,'yticklabel',[])
title('GLP-HS','interpreter','latex')

ddata = Z_cnmf;
rgbim(:,:,1) = imadjust(rescale(ddata(:,:,vbands(1)),1));
rgbim(:,:,2) = imadjust(rescale(ddata(:,:,vbands(2)),1));
rgbim(:,:,3) = imadjust(rescale(ddata(:,:,vbands(3)),1));
axes(ha(4)); imshow(rgbim); % display RGB image
axis square, set(gca,'xtick',[]), set(gca,'xticklabel',[]), set(gca,'ytick',[]), set(gca,'yticklabel',[])
title('CNMF','interpreter','latex')

ddata = Zh_srr;
rgbim(:,:,1) = imadjust(rescale(ddata(:,:,vbands(1)),1));
rgbim(:,:,2) = imadjust(rescale(ddata(:,:,vbands(2)),1));
rgbim(:,:,3) = imadjust(rescale(ddata(:,:,vbands(3)),1));
axes(ha(5)); imshow(rgbim); % display RGB image
axis square, set(gca,'xtick',[]), set(gca,'xticklabel',[]), set(gca,'ytick',[]), set(gca,'yticklabel',[])
title('FuVar','interpreter','latex')

% vbands = [57,30,20]; % RGB bands
vbands = [115,81,38]; % RGB bands OK IR
ddata = Zh_th;
rgbim(:,:,1) = imadjust(rescale(ddata(:,:,vbands(1)),1));
rgbim(:,:,2) = imadjust(rescale(ddata(:,:,vbands(2)),1));
rgbim(:,:,3) = imadjust(rescale(ddata(:,:,vbands(3)),1));
axes(ha(6)); imshow(rgbim); % display RGB image
axis square, set(gca,'xtick',[]), set(gca,'xticklabel',[]), set(gca,'ytick',[]), set(gca,'yticklabel',[])

ddata = Z_hysure;
rgbim(:,:,1) = imadjust(rescale(ddata(:,:,vbands(1)),1));
rgbim(:,:,2) = imadjust(rescale(ddata(:,:,vbands(2)),1));
rgbim(:,:,3) = imadjust(rescale(ddata(:,:,vbands(3)),1));
axes(ha(7)); imshow(rgbim); % display RGB image
axis square, set(gca,'xtick',[]), set(gca,'xticklabel',[]), set(gca,'ytick',[]), set(gca,'yticklabel',[])

ddata = Z_glphs;
rgbim(:,:,1) = imadjust(rescale(ddata(:,:,vbands(1)),1));
rgbim(:,:,2) = imadjust(rescale(ddata(:,:,vbands(2)),1));
rgbim(:,:,3) = imadjust(rescale(ddata(:,:,vbands(3)),1));
axes(ha(8)); imshow(rgbim); % display RGB image
axis square, set(gca,'xtick',[]), set(gca,'xticklabel',[]), set(gca,'ytick',[]), set(gca,'yticklabel',[])

ddata = Z_cnmf;
rgbim(:,:,1) = imadjust(rescale(ddata(:,:,vbands(1)),1));
rgbim(:,:,2) = imadjust(rescale(ddata(:,:,vbands(2)),1));
rgbim(:,:,3) = imadjust(rescale(ddata(:,:,vbands(3)),1));
axes(ha(9)); imshow(rgbim); % display RGB image
axis square, set(gca,'xtick',[]), set(gca,'xticklabel',[]), set(gca,'ytick',[]), set(gca,'yticklabel',[])

ddata = Zh_srr;
rgbim(:,:,1) = imadjust(rescale(ddata(:,:,vbands(1)),1));
rgbim(:,:,2) = imadjust(rescale(ddata(:,:,vbands(2)),1));
rgbim(:,:,3) = imadjust(rescale(ddata(:,:,vbands(3)),1));
axes(ha(10)); imshow(rgbim); % display RGB image
axis square, set(gca,'xtick',[]), set(gca,'xticklabel',[]), set(gca,'ytick',[]), set(gca,'yticklabel',[])

axes(ha(1)); ylabel('Visible','interpreter','latex')
axes(ha(6)); ylabel('Infrared','interpreter','latex')
% print(['examples/recIm_ex2_Paris_vis_IR'],'-dpdf')







figure;
plot(qualIdx_hysure.psnrall)
hold on
plot(qualIdx_glphs.psnrall)
plot(qualIdx_cnmf.psnrall)
plot(qualIdx_srr.psnrall)
xlim([1 length(qualIdx_hysure.psnrall)])
legend('HySure','GLP-HS','CNMF','FuVar')
xlabel('Spectral band')
ylabel('PSNR [dB]')
% print(['examples/PSNR_band_ex2_Paris'],'-dpdf')


figure;
[ha, pos] = tight_subplot(1, 4, 0.01, 0.1, [0.1 0.15]);
mmax = max([max(qualIdx_hysure.sammap(:))  max(qualIdx_glphs.sammap(:)) ...
    max(qualIdx_cnmf.sammap(:))  max(qualIdx_srr.sammap(:))]);
mmin = min([min(qualIdx_hysure.sammap(:))  min(qualIdx_glphs.sammap(:)) ...
    min(qualIdx_cnmf.sammap(:))  min(qualIdx_srr.sammap(:))]);

axes(ha(1)); imagesc(qualIdx_hysure.sammap, [mmin mmax])
title('HySure','interpreter','latex'), set(gca,'ytick',[],'xtick',[]), axis square
axes(ha(2)); imagesc(qualIdx_glphs.sammap, [mmin mmax])
title('GLP-HS','interpreter','latex'), set(gca,'ytick',[],'xtick',[]), axis square
axes(ha(3)); imagesc(qualIdx_cnmf.sammap, [mmin mmax])
title('CNMF','interpreter','latex'), set(gca,'ytick',[],'xtick',[]), axis square
axes(ha(4)); imagesc(qualIdx_srr.sammap, [mmin mmax])
title('FuVar','interpreter','latex'), set(gca,'ytick',[],'xtick',[]), axis square
originalSize2 = get(gca, 'Position');
h=colorbar; 
set(ha(4), 'Position', originalSize2);
set(h,'fontsize',12);
% print(['examples/SAM_map_ex2_Paris'],'-dpdf')




%%




