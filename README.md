# Fast Video Deduplication and Localization With Temporal Consistence Re-Ranking
This is the official implementation of our IEEE TCVST 2024 paper titled [Fast Video Deduplication and Localization With Temporal Consistence Re-Ranking](https://ieeexplore.ieee.org/document/10577179/).

<div align="center">
  <img src="overview_TCVST_2024.png" alt="Overview" width="550"/>
</div>

## Prerequisites
The code is tested on Linux with the following prerequisites:

1. Python 3.13
2. MATLAB
3. PyTorch 1.11.0 (CUDA 11.3)
4. Numpy 1.26.4

np
pil
joblib
tqdm

vlfeat-0.9.21
 
## Dataset Preparation

### Option 1: Use Pre-computed Data

Download

### Option 2: Prepare dataset from scratch

1. Download the [FIVR-200K dataset](https://github.com/MKLab-ITI/FIVR-200K/tree/master).

    The paper uses videos categorized as "Duplicate Scene Videos (DSVs)". The datasets contains a total of 7,558 DSVs labelled as 'ND' in [annotations.json](https://github.com/MKLab-ITI/FIVR-200K/blob/master/dataset/annotation.json). We provide [youtube_ids_ND.txt](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/fivr_data_process/youtube_ids_ND.txt) which contains IDs of all DSVs. Only 4,960 DSVs were available for download at the time of writing our paper. IDs for the unavailable DSVs are provided in [missing_videos.txt](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/fivr_data_process/missing_videos.txt).

   > Note: Most video links might be unavailable for download. Contacting the FIVR-200K dataset authors may help.
   
2. Extract frames.
   ```bash
   python extract_frames_multcore.py --data_path <path_to_fivr_videos> --frame_interval 0.5
   ```
    
   - `<path_to_fivr_videos>` must contain one subfolder per query ID.
   - All videos corresponding to the same query ID must be placed inside the same subfolder.

4. Extract thumbnail features.
   ```bash
    python gen_thumb_ft.py --data_path <path_to_fivr_frames> --out_path <path_to_thumbnail_features> --global_mean <global_mean_value> --thumb_size 12
   ```
   `<global_mean_value>` required for [gen_thumb_ft.py](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/gen_thumb_ft.py) can be calculated via:
   ```bash
    python get_global_mean.py --data_path <path_to_fivr_frames> --thumb_size 12
   ```
5. Generate VGG features.
   ```bash
    python gen_vgg_ft.py --data_path <path_to_fivr_frames> --out_path <path_to_vgg_features> --batch_size 256
   ```
6. Generate fisher vector features.
   ```bash
    python gen_vgg_ft.py --data_path <path_to_fivr_frames> --out_path <path_to_vgg_features> --batch_size 256
   ```
     
   

### Thumbnail FG

#### Feature Generation
- To generate thumbnail feature:
```
