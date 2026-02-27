# Fast Video Deduplication and Localization With Temporal Consistence Re-Ranking
This is the official implementation of the IEEE TCSVT 2024 paper titled [Fast Video Deduplication and Localization With Temporal Consistence Re-Ranking](https://ieeexplore.ieee.org/document/10577179/).

<div align="center">
  <img src="figures/overview_TCSVT_2024.png" alt="overview_TCSVT_2024.png" width="590"/>
</div>

## Prerequisites
This code was tested on Linux with the following prerequisites:
1. Python 3.12
2. PyTorch 1.11.0 (CUDA 11.3)
3. MATLAB R2021a
4. VLFeat 0.9.21

Remaining libraries are available in [requirements.txt](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/requirements.txt)

## Installation

- Clone this repository
   ```bash
   git clone https://github.com/engrchrishenry/loc_aware_video_dedup.git
   cd loc_aware_video_dedup
   ```

- Create conda environment
   ```bash
   conda create --name dedup python=3.12
   conda activate dedup
   ```

- Install dependencies
  1. Install [PyTorch](https://pytorch.org/get-started/locally/).
  2. Install [FFmpeg](https://www.ffmpeg.org/download.html).
  3. The remaining packages can be installed via:
     ```bash
     pip install -r requirements.txt
     ```
  4. For running MATLAB scripts, you are required to install [VLFeat](https://www.vlfeat.org/download.html).

## Quick Demo

- Download the pre-generated FIVR-200K and VCSL datasets ([here](https://mailmissouri-my.sharepoint.com/:f:/g/personal/chffn_umsystem_edu/IgBK1Ogmv9s8SLPBKsCBa4MzAW9IBIsXk2lrVQTvU9WDTiE?e=WqdRb0)).
- Unzip the downloaded .zip files
  ```bash
  unzip FIVR_200K_dataset_processed.zip
  unzip VCSL_dataset_processed.zip
  ```
  Dataset folder structure:
  
  Filename suffix format is `<Fisher_Vector_Dimension>_<Thumbnail_Dimension>_<VGG_Dimension>`

      FIVR_200K_dataset_processed/VCSL_dataset_processed
      ├── features_projected                                      # Features (fisher vector, thumbnail, and VGG)
      │   ├── fv                                                  # Fisher vector features
      │   ├── thumb                                               # Thumbnail features
      │   └── vgg                                                 # VGG features
      ├── indexes                                                 # Index files
      ├── kdtrees                                                 # k-d Trees (repositories)
      └── PCA_models_trained                                      # Trained PCA models
- Use the provided [config_video_retrieval.yaml](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/configs/config_video_retrieval.yaml) file to test on the FIVR-200K Normal Testset with `fisher vector feature dimension=128`, `thumbnail feature dimension=64`, and `VGG feature dimension=64`. Modify the parameters in [config_video_retrieval.yaml](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/configs/config_video_retrieval.yaml) to test on other testsets. Read the comments in the [config_video_retrieval.yaml](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/configs/config_video_retrieval.yaml) for guidance on modifying the parameters.
- Run [video_retrieval.py](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/video_retrieval.py) as follows:
  ```bash
  python video_retrieval.py --config configs/config_video_retrieval.yaml
  ```
- Sample output in `results_all.txt` file:
  ```text
    k = 512, ts 10.0, tolerance 1.5
    Avg node search time/frame = 95.0128 ms
    Avg node search time/query = 475.0642 ms
    Avg prune time/frame = 15.1487 ms
    Avg prune time/query = 75.7435 ms
    Avg total time/frame = 110.2708 ms
    Avg total time/query = 551.3542 ms
    Recall = 98.8
    Timestamp accuracy at varying tolerance levels:
      threshold=0.0   → 96.6000%
      threshold=1.0   → 97.3000%
      threshold=5.0   → 97.5000%
    ------------------------------------------
  ```

## Dataset Preparation

We provide the pre-generated FIVR-200K and VCSL datasets ([Click Here](https://mailmissouri-my.sharepoint.com/:f:/g/personal/chffn_umsystem_edu/IgBK1Ogmv9s8SLPBKsCBa4MzAW9IBIsXk2lrVQTvU9WDTiE?=WqdRb0)) for result reproducibility. We are unable to provide the raw video files as each video file belongs to its respective owner.

To prepare the dataset from scratch, follow the steps below:

### Download Dataset
- FIVR-200K Dataset ([Download Here](https://github.com/MKLab-ITI/FIVR-200K/tree/master))

   The paper uses videos categorized as "Duplicate Scene Videos (DSVs)". The dataset contains a total of 7,558 DSVs labeled as 'ND' in [annotation.json](https://github.com/MKLab-ITI/FIVR-200K/blob/master/dataset/annotation.json). We provide [youtube_ids_ND.txt](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/data/youtube_ids_ND.txt) which contains IDs of all DSVs. Only 4,960 DSVs were available for download at the time of writing our paper. The list of 4,960 videos used in our experiments is provided in [FIVR_available_videos.txt](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/data/FIVR_available_videos.txt).

- VCSL Dataset ([Download Here](https://github.com/alipay/VCSL/tree/main))
  
  We used the urls in [videos_url_uuid.csv](https://github.com/alipay/VCSL/blob/main/data/videos_url_uuid.csv) to download the dataset. Only 6,649 videos were available for download at the time of writing our paper. The UUIDs for the 6,649 videos used in our experiments are provided in [VCSL_available_videos.txt](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/data/VCSL_available_videos.txt).

  > Note: Most video links might be unavailable for download. Contacting the FIVR-200K dataset and VCSL authors may help.

### Extract frames.
- FIVR-200K Dataset
  ```bash
  python extract_frames.py --data_path <path_to_fivr_videos> --out_path <path_to_frames> --frame_interval 0.5
  ```
  
  - `<path_to_fivr_videos>` must contain one subfolder per query ID.
  - All videos corresponding to the same query ID must be placed inside the same subfolder.

- VCSL Dataset
  ```bash
  python extract_frames.py --data_path <path_to_vcsl_videos> --out_path <path_to_frames> --frame_interval 0.5
  ```
 
  - `<path_to_vcsl_videos>` must contain a subfolder named '1'. The subfolder must contain all the downloaded videos.

### Generate test data
- Select test videos

  Run the following to select test videos based on criteria mentioned in our paper. Skip this step to use the test data used for experiments in our paper: [test_data_normal_list_FIVR.txt](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/data/test_data_normal_list_FIVR.txt), [test_data_hard_list_FIVR.txt](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/data/test_data_hard_list_FIVR.txt), [test_data_normal_list_VCSL.txt](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/data/test_data_normal_list_VCSL.txt), and [test_data_hard_list_VCSL.txt](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/data/test_data_hard_list_VCSL.txt)
  ```bash
  python select_test_videos.py --data_path <path_to_frames> --save_file <output_path_with_filename>
  ```
  Each line in the resulting .txt file will have the following format: `"video_class/video_name frame_id_1 frame_id_2 ... frame_id_n"`.

    > Note: [select_test_videos.py](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/select_test_videos.py) requires the frames to be extracted via [extract_frames.py](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/extract_frames.py)

- Generate normal version (as mentioned in our paper)
  ```bash
  python gen_test_data_normal.py --test_vid_list <test_data_list_path> --frames_path <path_to_frames> --out_path <output_path> --frame_interval 0.5
  ```
- Generate hard version (as mentioned in our paper)
  ```bash
  python gen_test_data_hard.py --test_vid_list <test_data_list_path> --frames_path <path_to_frames> --out_path <output_path> --frame_interval 0.5
  ```

## Feature Generation
<div align="center">
  <img src="figures/feature_generation.jpg" alt="feature_generation.jpg" width="680"/>
</div>

- ### Thumbnail Feature
  Generate thumbnail features
  ```bash
  python gen_thumb_ft.py --data_path <path_to_frames> --out_path <path_to_thumbnail_features> --global_mean <global_mean_value> --thumb_size 12
  ```
  `<global_mean_value>` required for [gen_thumb_ft.py](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/gen_thumb_ft.py) can be calculated via:
  ```bash
  python get_global_mean.py --data_path <path_to_frames> --thumb_size 12
  ```
  
  Generate a single thumbnail feature file (.h5)
  ```bash
  python make_hdf5.py --feature_path <thumbnail_features_path> --save_file <output_path_with_filename>
  ```

- ### VGG Feature
  Generate VGG features
  ```bash
  python gen_vgg_ft.py --data_path <path_to_frames> --out_path <path_to_vgg_features> --batch_size 256
  ```
  Generate a single VGG feature file (.h5)
  ```bash
  python make_hdf5.py --feature_path <VGG_features_path> --save_file <output_path_with_filename>
  ```

- ### Fisher Vector Feature

  Generate fisher vector features (MATLAB Script).
  - Download the trained GMM model [trained_GMM_model.mat](https://mailmissouri-my.sharepoint.com/:f:/g/personal/chffn_umsystem_edu/IgBK1Ogmv9s8SLPBKsCBa4MzAW9IBIsXk2lrVQTvU9WDTiE?e=WqdRb0).
  - Modify VLFeat, trained GMM model, img_path, save_folder paths in [gen_fv_ft.m](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/fisher_vector_generation/gen_fv_ft.m) from [fisher_vector_generation](https://github.com/engrchrishenry/loc_aware_video_dedup/tree/main/fisher_vector_generation) folder.
  - Run [gen_fv_ft.m](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/fisher_vector_generation/gen_fv_ft.m) in MATLAB.

  Generate a single fisher vector feature file (.h5)
  ```bash
  python make_hdf5.py --feature_path <fisher_vector_features_path> --save_file <output_path_with_filename>
  ```

## Build repository (k-d Trees)

- ### Build repository

  Generate indexes file (.pickle)
  ```bash
  python gen_index_file.py --frames_path <path_to_frames> --save_file <pickle_file_path> --frame_interval 0.5
  ```
  Generate k-d Trees
  ```bash
  python build_repo.py \
    --thumb_file <thumbs_h5_file_path> \
    --fv_file <fv_h5_file_path> \
    --vgg_file <vgg_h5_file_path> \
    --index_file <index_file_path> \
    --out_path <output_path> \
    --pca_train_samp <number_of_train_samples_for_pca> \
    --comps <fv_pca_components thumb_pca_components vgg_pca_components>
  ```
  
  - `--thumb_file`, `--fv_file`, and `--vgg_file` are the paths to the .h5 files generated via [make_hdf5.py](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/make_hdf5.py).
  - `--index_file` is the .pickle file generated via [gen_index_file.py](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/gen_index_file.py).
  - `<number_of_train_samples_for_pca>` value of 150000 and 200000 was used for FIVR-200K dataset and VCSL dataset, respectively, in our experiments.

- ### Build test repository

  Generate test indexes file (.pickle)
  ```bash
  python gen_index_file.py --frames_path <path_to_test_frames> --save_file <pickle_file_path> --frame_interval 0.5
  ```
  
  Generate k-d Trees
  ```bash
  python build_test_repo.py \
    --thumb_file <thumbs_test_h5_file_path> \
    --fv_file <fv_test_h5_file_path> \
    --vgg_file <vgg_test_h5_file_path> \
    --index_file <index_test_file_path> \
    --out_path <output_test_path> \
    --pca_model <pca_model_path> \
    --comps <fv_pca_components thumb_pca_components vgg_pca_components>
  ```
  
  - `--thumb_file`, `--fv_file`, and `--vgg_file` are the paths to the .h5 files generated via [make_hdf5.py](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/make_hdf5.py).
  - `--index_file` is the .pickle file generated via [gen_index_file.py](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/gen_index_file.py).
  - `--pca_model` is the path to the trained PCA model generated after running [build_repo.py](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/build_repo.py).
  
## Video Retrieval with Localization

<table align="center" border="0" cellspacing="0" cellpadding="0">
  <tr>
    <td align="center" valign="middle">
      <img src="figures/temporal_consistence_pruning.png" width="380"/><br>
      <sub><b>(a)</b> Temporal Consistence Pruning Algorithm</sub>
    </td>

  <td width="25"></td> <!-- spacing -->

  <td align="center" valign="middle">
    <img src="figures/temporal_consistence_pruning_intuition.png" width="380"/><br>
    <sub><b>(b)</b> Intuition of Temporal Consistence Pruning</sub>
  </td>
  </tr>
</table>


- Modify the parameters in [config_video_retrieval.yaml](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/configs/config_video_retrieval.yaml). Read the comments in the [config_video_retrieval.yaml](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/configs/config_video_retrieval.yaml) for guidance on modifying the parameters.  
- Run [video_retrieval.py](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/video_retrieval.py) as follows:
  
  ```bash
  python video_retrieval.py --config <config_file_path>
  ```
  [video_retrieval.py](https://github.com/engrchrishenry/loc_aware_video_dedup/blob/main/video_retrieval.py) can produce results for varying k values. It outputs results in .txt files in `--out_path`.


## Results

<div align="center">
  <img src="figures/results_table_I_TCSVT_2024.png" alt="results_table_I_TCSVT_2024.png" width="620"/>
  <p>Recall (%) at different K values for positive queries.</p>
</div>

<div align="center">
  <img src="figures/results_table_II_TCSVT_2024.png" alt="results_table_II_TCSVT_2024.png" width="550"/>
  <p>Table showing timestamp accuracy (%) at varying tolerance values.</p>
</div>

<div align="center">
  <img src="figures/results_TCSVT_2024.png" alt="results_TCSVT_2024.png" width="650"/>
  <p>Video retrieval results by our system.</p>
</div>


## Citation

If you use this work, please cite:

```bibtex
@ARTICLE{10577179,
  author={Henry, Chris and Song, Li and Li, Zhu},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Fast Video Deduplication and Localization With Temporal Consistence Re-Ranking}, 
  year={2024},
  volume={34},
  number={11},
  pages={12006-12018},
  keywords={Vectors;Feature extraction;Task analysis;Principal component analysis;Web sites;Video on demand;Computational modeling;Video deduplication;video retrieval;near-duplicate video retrieval;video copy detection;fisher vector},
  doi={10.1109/TCSVT.2024.3420422}}
```

You may also explore our other work on video deduplication [here](https://ieeexplore.ieee.org/document/10095417/). Consider citing the following:

```bibtex
@INPROCEEDINGS{10095417,
  author={Henry, Chris and Liao, Rijun and Lin, Ruiyuan and Zhang, Zhebin and Sun, Hongyu and Li, Zhu},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Lightweight Fisher Vector Transfer Learning for Video Deduplication}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  keywords={Computational modeling;Transfer learning;Transforms;Multilayer perceptrons;Signal processing;Robustness;Encoding;Video deduplication;near-duplicate video detection;near-duplicate video copy detection;fisher vector aggregation},
  doi={10.1109/ICASSP49357.2023.10095417}}
```

## Acknowledgement

This work was supported in part by the National Science Foundation (NSF) under Award 2148382.

## Contact
In case of questions, feel free to contact at chffn@umsystem.edu or engr.chrishenry@gmail.com
   
