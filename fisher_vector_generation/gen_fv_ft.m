clear all;
close all;

run('vlfeat-0.9.21/toolbox/vl_setup') % load vlfeat
load trained_GMM_model.mat; % load trained GMM model
img_path='frames/'; % path to input frames
save_folder = "output/fv/"; % path to output folder

fv_gmm = gmm(1,1); kd=16; nc=64;

% Count the number of files
files_temp = dir(fullfile(img_path, '**', '*'));
total_files = sum(~[files_temp.isdir]); % Exclude directories

all_folders = dir(img_path);
all_folders = all_folders(~ismember({all_folders.name} ,{'.','..'}));
dirFlags = [all_folders.isdir];
n_classes = length(dirFlags);

total_time = 0;
a = fv_gmm.m; b = fv_gmm.cov; c = fv_gmm.prior;
for vid_class=1:n_classes
    folder_path = string(all_folders(vid_class).folder) + '/' + string(all_folders(vid_class).name);    
    mkdir(save_folder + '/' + all_folders(vid_class).name);
    all_files = dir(folder_path);
    all_files = all_files(~ismember({all_files.name} ,{'.','..'}));
    all_files = natsortfiles(all_files);
    n_files = length(all_files);
    f_cnt = 0;
    for file=1:n_files
        vid_folder_path = string(all_files(file).folder) + '/' + string(all_files(file).name);
        all_frame_path = dir(vid_folder_path);
        all_frame_path = all_frame_path(~ismember({all_frame_path.name} ,{'.','..'}));
        all_frame_path = natsortfiles(all_frame_path);
        n_frames = length(all_frame_path);
        mkdir(save_folder + '/' + all_folders(vid_class).name + '/' + all_files(file).name);
        xc = string(save_folder) + '/' + all_folders(vid_class).name + '/' + string(all_files(file).name);
        parfor frame=1:n_frames
%         for frame=1:n_frames # Uncomment for single core processing
            tic;
            f_scfv = single(zeros(1, kd*nc));
            frame_path = string(all_frame_path(frame).folder) + '/' + string(all_frame_path(frame).name);
            im = imread(frame_path);
            
            % pull sift
            [~, sift_f]=vl_sift(single(rgb2gray(im)), 'PeakThresh', 0);
            
            fv_sift = double(sift_f')*A0(:,1:kd); % pca
            
            % aggregation via fisher vec
            fv = vl_fisher(fv_sift', a, b, c);
            
            f_scfv(1:nc*kd) = fv(1:nc*kd);
            
            elapsed_time = toc;
            total_time = total_time + elapsed_time;
            [~, f_name, ~] = fileparts(all_frame_path(frame).name);

            temp_save_name = xc + '/' + string(f_name) + '.mat';
            savetofile(f_scfv,temp_save_name);
        end
        fprintf('Video class %d/%d (File %d/%d)\n', vid_class, n_classes, file, n_files);
    end
end % for k

%{
Uncomment the following o calculate average time consumption per fisher vector computation.
For accurate calculation: make sure to comment "parfor frame=1:n_frames" and uncomment "for
frame=1:n_frames" 
%}

% average_time = total_time / total_files;
% fprintf('Average time per iteration: %.4f seconds\n', average_time);
