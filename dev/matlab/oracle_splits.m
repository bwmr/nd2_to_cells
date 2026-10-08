function oracle_splits(dumpdir, sspath, nthreads)
% oracle_splits : replay missingSeg2to1 calls dumped by dev/dump_splits.py.
%
% For each split_*.mat in dumpdir, runs SuperSegger's missingSeg2to1 and
% registerMasks_monointensity (r frame only) and writes split_*_ml.mat with
% ml_rl (resulting mask), ml_success, ml_T (affine2d T of the r registration)
% and ml_reg (registered r mask).
%
% nthreads (optional): computational threads; imregtform uses 1 on Windows.
%
% matlab -batch "oracle_splits('<dump dir>', '<omnisegger>/SuperSegger-master', 1)"

addpath(genpath(sspath));
suffix = '_ml.mat';
if nargin > 2
    maxNumCompThreads(nthreads);
    suffix = sprintf('_ml%d.mat', nthreads);
end
fprintf('numthreads = %d\n', feature('numthreads'));
files = dir(fullfile(dumpdir, 'split_*.mat'));
files = files(~cellfun(@isempty, regexp({files.name}, '^split_t\d{3}_\d{3}\.mat$')));
for i = 1:numel(files)
    d = load(fullfile(dumpdir, files(i).name));
    data_c = struct();
    data_c.regs.regs_label = double(d.rl);
    data_c.regs.num_regs = d.num_regs;
    data_r = struct();
    data_r.regs.regs_label = double(d.rlr);
    if isempty(d.rlf)
        data_f = [];
    else
        data_f = struct();
        data_f.regs.regs_label = double(d.rlf);
    end
    regR = d.regR(:)';
    regF = d.regF(:)';
    if isempty(regF)
        regF = [];
    end
    tic;
    [data_new, success] = missingSeg2to1(data_c, d.regC, data_r, regR, data_f, regF, []);
    ml_time = toc;
    maskc = double(data_c.regs.regs_label == d.regC);
    maskr = double(data_r.regs.regs_label == regR(1)) + ...
        2 * double(data_r.regs.regs_label == regR(2));
    regis = registerMasks_monointensity(maskr, maskc);
    ml_rl = uint16(data_new.regs.regs_label);
    ml_success = double(success);
    ml_T = regis.Transformation.T;
    ml_reg = single(regis.RegisteredImage);
    save(fullfile(dumpdir, strrep(files(i).name, '.mat', suffix)), ...
        'ml_rl', 'ml_success', 'ml_T', 'ml_reg', 'ml_time', '-v7');
    fprintf('%s: success %d (%.1f s)\n', files(i).name, success, ml_time);
end
end
