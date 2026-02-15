# Fast Video Deduplication and Localization With Temporal Consistence Re-Ranking
This is the official implementation of our IEEE TCVST 2024 paper titled [Fast Video Deduplication and Localization With Temporal Consistence Re-Ranking](https://ieeexplore.ieee.org/document/10577179/).

<div align="center">
  <img src="overview_TCVST_2024.png" alt="Overview" width="550"/>
</div>

## Prerequisites
The code is tested on Linux with the following prerequisites:

#### Python Code
1. Python 3.13
2. PyTorch 1.11.0 (CUDA 11.3)
3. Numpy
4. Pillow
5. Joblib
6. Tqdm
7. Natsort
8. Scipy
9. Sci-kit Learn
10. Matplotlib
11. YAML
12. OpenCV Python
13. Sci-kit Image

#### MATLAB Code
1. MATLAB R2021a
2. VLFeat 0.9.21
 
## Dataset Preparation

<!-- ### Option 1: Use Pre-computed Data -->

<!--Download -->

<!--### Option 2: Prepare dataset from scratch -->

1. Download the [FIVR-200K dataset](https://github.com/MKLab-ITI/FIVR-200K/tree/master).

    The paper uses videos categorized as "Duplicate Scene Videos (DSVs)". The datasets contains a total of 7,558 DSVs labelled as 'ND' in [annotations.json](https://github.com/MKLab-ITI/FIVR-200K/blob/master/dataset/annotation.json). We provide [youtube_ids_ND.txt](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/data/youtube_ids_ND.txt) which contains IDs of all DSVs. Only 4,960 DSVs were available for download at the time of writing our paper. The list of 4,960 videos used in our experiments is provided in [FIVR_available_videos.txt](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/data/FIVR_available_videos.txt).
   
   > Note: Most video links might be unavailable for download. Contacting the FIVR-200K dataset authors may help.

2. Download the [VCSL Dataset](https://github.com/alipay/VCSL/tree/main).

    We used the urls in [videos_url_uuid.csv](https://github.com/alipay/VCSL/blob/main/data/videos_url_uuid.csv) to download the dataset. Only 6,649 videos were available for download at the time of writing our paper. The uuids for the 6,649 videos used in our experiments are provided in [VCSL_available_videos.txt](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/data/VCSL_available_videos.txt).

3. Extract frames.
   
   FIVR-200K dataset
   ```bash
   python extract_frames.py --data_path <path_to_fivr_videos> --frame_interval 0.5
   ```
    
   - `<path_to_fivr_videos>` must contain one subfolder per query ID.
   - All videos corresponding to the same query ID must be placed inside the same subfolder.

   VCSL Dataset
   ```bash
   python extract_frames.py --data_path <path_to_vcsl_videos> --frame_interval 0.5
   ```
   
   - `<path_to_vcsl_videos>` must contain a subfolder with any name. The subfolder must contain all the downloaded videos.

 4. Select test videos
    
    Run the following to select test videos based on criteria mentioned in our paper. Test data used for experiments in our paper can be found in [test_data_list_FIVR.txt](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/data/test_data_list_FIVR.txt) and [test_data_list_VCSL.txt](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/data/test_data_list_VCSL.txt)
    ```bash
    python select_test_videos.py --data_path <path_to_frames> --save_file <output_path_with_filename>
    ```

    > Note: [select_test_videos.py](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/select_test_video) requires the frames to be extracted via [extract_frames.py](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/extract_frames.py)

  6. Prepare test data
     
     Generate normal version of test data as mentioned in our paper.
     ```bash
     python gen_test_data_normal.py --test_vid_list <test_data_list_path> --frames_path <path_to_frames> --out_path <output_path>
     ```
     Generate hard version of test data as mentioned in our paper.
     ```bash
     python gen_test_data_hard.py --test_vid_list <test_data_list_path> --frames_path <path_to_frames> --out_path <output_path>
     ```

## Feature Generation

- ### Thumbnail Feature
  Generate thumbnail features
  ```bash
  python gen_thumb_ft.py --data_path <path_to_frames> --out_path <path_to_thumbnail_features> --global_mean <global_mean_value> --thumb_size 12
  ```
  `<global_mean_value>` required for [gen_thumb_ft.py](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/gen_thumb_ft.py) can be calculated via:
  ```bash
  python get_global_mean.py --data_path <path_to_frames> --thumb_size 12
  ```
  Generate a single thumbnail feature file
  ```bash
  python gen_single_feature_npy.py --feature_path <thumbnail_features_path> --save_file <output_path_with_filename> --frame_interval 0.5
  ```

- ### VGG Feature
  Generate VGG features
  ```bash
  python gen_vgg_ft.py --data_path <path_to_frames> --out_path <path_to_vgg_features> --batch_size 256
  ```
  Generate a single VGG feature file
  ```bash
  python gen_single_feature_npy.py --feature_path <VGG_features_path> --save_file <output_path_with_filename> --frame_interval 0.5
  ```

- ### Fisher Vector Feature

  Generate fisher vector features [MATLAB Script].
  - Download and install [VLFeat](https://www.vlfeat.org/download.html).
  - Download the trained GMM model [trained_GMM_model.mat](https://mailmissouri-my.sharepoint.com/:u:/g/personal/chffn_umsystem_edu/IQBzyZ_xdtdrSZqW3noZWlj7AQBpr8ZCxQ1SWm-PCCfGLgI?e=66hrQc).
  - Modify VLFeat, trained GMM model, img_path, save_folder paths in [gen_fv_ft.m](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/fisher_vector_generation/gen_fv_ft.m) from [fisher_vector_generation](https://github.com/engrchrishenry/loc_aware_video_dedup/tree/main/fisher_vector_generation) folder.
  - Run [gen_fv_ft.m](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/fisher_vector_generation/extract_fv_sift_direct.m) in MATLAB.

  Generate a single fisher vector feature file
  ```bash
  python gen_single_feature_npy.py --feature_path <fisher_vector_features_path> --save_file <output_path_with_filename> --frame_interval 0.5
  ```

## Efficient Frame Retrieval via Multiple k-d Tree Setup
The script [frame_retrieval.py](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/frame_retrieval.py) does the following:
- Uses the single feature files generated via [gen_single_feature_file.py](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/gen_single_feature_file.py) as input.
- Trains a PCA model for each feature to reduce the feature dimension.
- Projects the features to low-dimensional vectors using the trained PCA model.
- Builds k-d trees via the projected features.
- Query the k-d trees for frame retrieval.
- Calculates the recall and average time consumption per query frame.

To run this step:
- Modify the parameters in [config_frame_retrieval.yaml](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/configs/config_frame_retrieval.yaml).
- Run [frame_retrieval.py](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/frame_retrieval.py).
  ```bash
  python frame_retrieval.py --config <config_file_path>
  ```

## Video Localization and Retrieval


   
     
   
