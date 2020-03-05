function Out = QualityIndices_mod(I_HS,I_REF,ratio)
%--------------------------------------------------------------------------
% Quality Indices
%
% USAGE
%   Out = QualityIndices(I_HS,I_REF,ratio)
%
% INPUT
%   I_HS     : target HS data (rows,cols,bands)
%   I_REF    : reference HS data (rows,cols,bands)
%   ratio    : GSD ratio between HS and MS imagers
%   blk_size : Block and shift size for Q2N metric (use 32 if nothing goes wrong)
%
% OUTPUT
%   Out.psnr : PSNR
%   Out.sam  : SAM
%   Out.ergas: ERGAS
%   Out.q2n  : Q2N
%
%--------------------------------------------------------------------------
blk_size = 32;

% % Q2n_index = q2n(I_REF, I_HS, blk_size, blk_size);
% % if abs(Q2n_index) < 1e-4
% %     Q2n_index = q2n(I_HS, I_REF, blk_size, blk_size);
% % end
% % Out.q2n = Q2n_index;
% [Q2n_index, ~] = q2n(I_REF, I_HS, blk_size, blk_size);
% Out.q2n = Q2n_index;
Out.q2n = 0; 


[angle_SAM,map] = SAM(I_HS,I_REF);
Out.sam = angle_SAM;
Out.ergas = ERGAS(I_HS,I_REF,ratio);
psnr = PSNR(I_REF,I_HS);
Out.psnrall = psnr.all;
Out.sammap = map;
Out.psnr = psnr.ave;




% UIQI - calls the method described in "A Universal Image Quality Index"
% by Zhou Wang and Alan C. Bovik
q_band = zeros(1, size(I_HS,3));
for idx1=1:size(I_HS,3)
    q_band(idx1)=img_qi(I_REF(:,:,idx1), I_HS(:,:,idx1), 32);
end
uiqi_idx = mean(q_band);
Out.uiqi = uiqi_idx;



% disp(['PSNR : ' num2str(Out.psnr)]);
% disp(['SAM  : ' num2str(Out.sam)]);
% disp(['ERGAS: ' num2str(Out.ergas)]);
% disp(['Q2n  : ' num2str(Out.q2n)]);