## DDESeg

This repository provides the PyTorch implementation for the **CVPR2025** paper "Dynamic Derivation and Elimination: Audio Visual Segmentation with Enhanced Audio Semantics". [[Arxiv]](https://arxiv.org/abs/2503.12840)

### Approach
![Pipeline](https://github.com/YenanLiu/DDESeg/blob/main/ddeseg.png)
DDESeg reconstructs the semantic content of the mixed audio signal by enriching the distinct semantic information of each individual source, deriving representations that preserve the unique characteristics of each sound. To reduce the matching difficulty, we introduce a discriminative feature learning module, which enhances the semantic distinctiveness of generated audio representations. Considering that not all derived audio representations directly correspond to visual features (e.g., off-screen sounds), we propose a dynamic elimination module to filter out non-matching elements. This module facilitates targeted interaction between sounding regions and relevant audio semantics. By scoring the interacted features, we identify and filter out irrelevant audio information, ensuring accurate audio-visual alignment.

### 1. Environment Preparation

Instructions for preparing your environment to run the code:
```
   $ conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0  pytorch-cuda=11.8 -c pytorch -c nvidia
   $ conda env create -f ddeseg_conda.yaml
```
### 2. Data Preparation
We reorganized the directory structures for AVSS and VPO to unify the data loading process. To help you get started quickly, you can directly utilize our reorganized [dataset here](https://huggingface.co/datasets/Yenan/DDESeg/tree/main). Additionally, you can download the original datasets from the AVSS and VPO repositories and implement your own data loader.

To facilitate a quick start, I newly generate the audio memory bank and bounding box files (all in zip files) for the sounding objects (the bbox file is used for data augmentation constraints). The audio memory bank is constructed using K-Means (details are provided in our paper). You have the option to generate your own audio memory bank and adjust the number of clusters for better performance.

### 3. Pretrained Model Preparation
We provide the pretrained visual and audio [backbones here](https://huggingface.co/datasets/Yenan/DDESeg/tree/main). The visual backbone for DDESeg is trained on [ImageNet](https://image-net.org/index.php), while the audio backbone is trained on [AudioSet](https://research.google.com/audioset/). Additionally, you can experiment with other versions and replace the backbones as needed.

### 4. Training Instructions
**First** [Necessary], replace all data file paths in the code with your own saving root.

**Second** [Optional], adjust the hyperparameters in `configs/ddeseg.yaml` and `runner.py` as needed.

**Tip**: We observed that using a larger batch size can improve performance. This phenomenon occurs because a larger batch size helps ensure more stable training, especially considering that the AVS datasets contain completely silent cases.

#### AVSS Setting
```
python runner.py -ddp -wandb -wn "avss_train" -nc 71 --task 'v2' --cfg './configs/ddeseg.yaml' -train_b 50 -val_b 1
```
#### AVS-Object-V1s (s4) Setting
```
python runner.py -ddp -wandb -wn "avs_v1s_train" -nc 1 --task 'v1s' --cfg './configs/ddeseg.yaml' -train_b 50 -val_b 1
```
#### AVS-Object-V1m (MS3) Setting
```
python runner.py -ddp -wandb -wn "avs_v1m_train" -nc 1 --task 'v1m' --cfg './configs/ddeseg.yaml' -train_b 50 -val_b 1 
```

#### VPO-SS Setting
```
python runner.py -ddp -wandb -wn "VPO-SS_train" -nc 22 --task 'VPO-SS' --cfg './configs/ddeseg.yaml' -train_b 50 -val_b 1 
```
#### VPO-MS Setting
```
python runner.py -ddp -wandb -wn "VPO-MS_train" -nc 22 --task 'VPO-MS' --cfg './configs/ddeseg.yaml' -train_b 50 -val_b 1
```
#### VPO-MSMI Setting
```
python runner.py -ddp -wandb -wn "VPO-MSMI_train" -nc 22 --task 'VPO-MSMI' --cfg './configs/ddeseg.yaml' -train_b 50 -val_b 1
```
### 5. Citation

```
@inproceedings{liu2025dynamic,
  title={Dynamic Derivation and Elimination: Audio Visual Segmentation with Enhanced Audio Semantics},
  author={Liu, Chen and Yang, Liying and Li, Peike and Wang, Dadong and Li, Lincheng and Yu, Xin},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={3131--3141},
  year={2025}
}
@misc{liu2025dynamicderivationeliminationaudio,
      title={Dynamic Derivation and Elimination: Audio Visual Segmentation with Enhanced Audio Semantics}, 
      author={Chen Liu and Liying Yang and Peike Li and Dadong Wang and Lincheng Li and Xin Yu},
      year={2025},
      eprint={2503.12840},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2503.12840}, 
}
```
### License
This project is licensed under the MIT License - see the [LICENSE file](https://github.com/YenanLiu/DDESeg_TPAMI/blob/main/LICENSE) for details.
