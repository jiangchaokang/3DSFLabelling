<div align="center">    
<img src="images/logo.jpg" width="600" height="120" alt="Celebration"/>   

## *CVPR 2024* | 3DSFLabelling: Boosting 3D Scene Flow Estimation by Pseudo Auto-labelling 
[![CVPR](http://img.shields.io/badge/CVPR-2024-4b44ce.svg)](https://arxiv.org/pdf/2402.18146.pdf)
[![Arxiv](http://img.shields.io/badge/Arxiv-2402.10668-B31B1B.svg)](https://arxiv.org/abs/2402.18146)
<img src="images/celebration.gif" width="35" height="35" alt="Celebration"/>

<a href='[https://jiangchaokang.github.io/3DSFLabelling-Page/](https://jiangchaokang.github.io/3DSFLabelling-Page/)' style='padding-left: 0.5rem;'>
<img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'> </a>

![GitHub stars](https://img.shields.io/github/stars/jiangchaokang/3DSFLabelling)
![GitHub contributors](https://img.shields.io/github/contributors/jiangchaokang/3DSFLabelling)
![GitHub issues](https://img.shields.io/github/issues-raw/jiangchaokang/3DSFLabelling)
![GitHub release (custom)](https://img.shields.io/badge/release-V0.1-blue)
![Downloads](https://img.shields.io/github/downloads/jiangchaokang/3DSFLabelling/total)


Check out the project demo here: [3DSFLabelling-Page](https://jiangchaokang.github.io/3DSFLabelling-Page/)

#### The code is gradually being released, please be patient.
[poster coming soon] [video coming soon]

| Description | Simple Example of the Auto-Labelling |
|-------------|-------|
| The proposed 3D scene flow pseudo-auto-labelling framework. Given point clouds and initial bounding boxes, both global and local motion parameters are iteratively optimized. Diverse motion patterns are augmented by randomly adjusting these motion parameters, thereby creating a diverse and realistic set of motion labels for the training of flow estimation models. | ![The proposed 3D scene flow pseudo-auto-labelling framework. Given point clouds and initial bounding boxes, both global and local motion parameters are iteratively optimized. Diverse motion patterns are augmented by randomly adjusting these motion parameters.](images/abstract.jpg) |
</div>



## Highlights <a name="highlights"></a>

:fire: **GENERATE FULLY ALIGNED INTER-FRAME POINT CLOUDS**

We propose a new framework for the automatic labelling of 3D scene flow pseudo-labels, significantly enhancing the accuracy of current scene flow estimation models, and effectively addressing the scarcity of 3D flow labels in autonomous driving.

:fire 

:star2: **Plug-and-play & Novel motion augmentation**

We propose a universal 3D box optimization method with multiple motion attributes. Building upon this, we further introduce a plug-and-play 3D scene flow augmentation module with global-local motions and motion status. This allows for flexible motion adjustment of ego-motion and dynamic environments, setting a new benchmark for scene flow data augmentation.


## News <a name="news"></a>

- `[2024/4]` :fire: Open source example model ([GMSF](https://github.com/ZhangYushan3/GMSF/tree/main), [MSBRN](https://github.com/cwc1260/MSBRN) and [FLOT](https://github.com/valeoai/FLOT/tree/master)), Training and evaluation [code](https://github.com/jiangchaokang/3DSFLabelling/tree/main/Baseline). 
- `[2024/4]` :fire: Open sourced a [new data](https://github.com/jiangchaokang/3DSFLabelling/tree/main/Data) set with 3D scene flow GT, The data comes from [KITTI Odometry Dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php), [Argoverse Dataset](https://www.argoverse.org/av2.html#download-link) and [nuScenes Dataset](https://www.argoverse.org/av2.html#download-link) respectively.
- `[2024/4]` :fire: Open sourced the [code](https://github.com/jiangchaokang/3DSFLabelling/tree/main/Gen_SF_label) for the **3D Scene Flow Label Generation**
- `[2024/03]` 3DSFLabelling code and models initially released.
- `[2024/02]` 3DSFLabelling is accepted by [CVPR 2024](https://cvpr.thecvf.com/virtual/2024/papers.html?filter=titles&search=3DSFLabelling:+Boosting+3D+Scene+Flow+Estimation+by+Pseudo+Auto-labelling).
- `[2024/02]` 3DSFLabelling [paper](https://arxiv.org/abs/2402.18146) released.

## TODO List <a name="TODO List"></a>

Still in progress:
- [ ] Datasets are easier to use.
- [x] The validity of the generated labels is verified on motion segmentation and LiDAR odometry.
- [ ] Readability optimization of configuration files and data reading code sections.


## Table of Contents

1. [Results and Model Zoo](#models)
2. [License and Citation](#license-and-citation)
3. [Comparative Results](#Comparative)

## Results and Model Zoo <a name="models"></a>

| Method | Dataset | Pre-trained Model | EPE3D |
|:------:|:--------:|:------------------:|:-----:|
| [GMSF](https://proceedings.neurips.cc/paper_files/paper/2023/hash/cb1c4782f159b55380b4584671c4fd88-Abstract-Conference.html)   | lidarKITTI | [gmsf_lidarKITTI_epe0.008.pth](https://drive.google.com/file/d/1EhkAP1cPZt3OgLW9Vw5l8qfBTAMtEwtG/view?usp=drive_link) | 0.008 |
| [GMSF](https://proceedings.neurips.cc/paper_files/paper/2023/hash/cb1c4782f159b55380b4584671c4fd88-Abstract-Conference.html)   | Argoverse   | [gmsf_argoverse_epe0.013.pth](https://drive.google.com/file/d/1fbQqIgPguFhFjA_0pNrM1rTmvAgejgIZ/view?usp=drive_link) | 0.013 |
| [GMSF](https://proceedings.neurips.cc/paper_files/paper/2023/hash/cb1c4782f159b55380b4584671c4fd88-Abstract-Conference.html)   | nuScenes    | [gmsf_nuScene_epe0.018.pth](https://drive.google.com/file/d/1YzewTDHpYEyPAn7iIg8Mvy6hmSguDZx-/view?usp=drive_link) | 0.018 |
| [FLOT](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730528.pdf)   | lidarKITTI | [flot_lidarKITTI_epe0.018.tar](sf_model/checkpoints/flot_lidarKITTI_epe0.018.tar) | 0.018 |
| [FLOT](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730528.pdf)   | Argoverse   | [flot_argoverse_epe0.043.tar](sf_model/checkpoints/flot_argoverse_epe0.043.tar) | 0.043 |
| [FLOT](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730528.pdf)   | nuScenes    | [flot_nuScenes_epe0.061.tar](sf_model/checkpoints/flot_nuScenes_epe0.061.tar) | 0.061 |
| [MSBRN](https://openaccess.thecvf.com/content/ICCV2023/papers/Cheng_Multi-Scale_Bidirectional_Recurrent_Network_with_Hybrid_Correlation_for_Point_Cloud_ICCV_2023_paper.pdf)  | lidarKITTI | [msbrn_lidarKITTI_epe0.011.pth](sf_model/checkpoints/msbrn_lidarKITTI_epe0.011.pth) | 0.0110 |
| [MSBRN](https://openaccess.thecvf.com/content/ICCV2023/papers/Cheng_Multi-Scale_Bidirectional_Recurrent_Network_with_Hybrid_Correlation_for_Point_Cloud_ICCV_2023_paper.pdf)   | Argoverse   | [msbrn_argoverse_epe0.017.pth](sf_model/checkpoints/msbrn_argoverse_epe0.017.pth) | 0.017 |
| [MSBRN](https://openaccess.thecvf.com/content/ICCV2023/papers/Cheng_Multi-Scale_Bidirectional_Recurrent_Network_with_Hybrid_Correlation_for_Point_Cloud_ICCV_2023_paper.pdf)   | nuScenes    | [msbrn_nuScenes_epe0.076.pth](sf_model/checkpoints/msbrn_nuScenes_epe0.076.pth) | 0.076 |


## Comparative results <a name="Comparative"></a>
#### The comparative results between our method and baseline.  "↑" signifies accuracy enhancement. In real-world LiDAR scenarios, our method markedly improves the 3D flow estimation accuracy across three datasets on the three baselines. This demonstrates that the proposed pseudo-auto-labelling framework can substantially boost the accuracy of existing methods, even without the need for ground truth.

| Dataset | Method | EPE3D↓ | Acc3DS↑ | Acc3DR↑ |
|:---:|:---:|:---:|:---:|:---:|
|  | <span style="background-color:#000000">FLOT [1]</span> | <span style="background-color:#000000">0.6532</span> | <span style="background-color:#000000">0.1554</span> | <span style="background-color:#000000">0.3130</span> |
|  | FLOT+3DSFlabelling | **0.0189** **↑97.1%** | **0.9666** | **0.9792** |
|  | <span style="background-color:#000000">MSBRN [2]</span> | <span style="background-color:#000000">0.0139</span> | <span style="background-color:#000000">0.9752</span> | <span style="background-color:#000000">0.9847</span> |
| <div style="text-align:left">LiDAR<br>KITTI</div> | MSBRN+3DSFlabelling | **0.0123** **↑11.5%** | **0.9797** | **0.9868** |
|  | <span style="background-color:#000000">GMSF [3]</span> | <span style="background-color:#000000">0.1900</span> | <span style="background-color:#000000">0.2962</span> | <span style="background-color:#000000">0.5502</span> |
|  | GMSF+3DSFlabelling | **0.0078** **↑95.8%** | **0.9924** | **0.9947** |


| Dataset | Method | EPE3D↓ | Acc3DS↑ | Acc3DR↑ |
|:---:|:---:|:---:|:---:|:---:|
|  | <span style="background-color:#000000">FLOT [1]</span> | <span style="background-color:#000000">0.2491</span> | <span style="background-color:#000000">0.0946</span> | <span style="background-color:#000000">0.3126</span> |
|  | FLOT+3DSFlabelling | **0.0107** **↑95.7%** | **0.9711** | **0.9862** |
| Argoverse | <span style="background-color:#000000">MSBRN [2]</span> | <span style="background-color:#000000">0.8691</span> | <span style="background-color:#000000">0.2432</span> | <span style="background-color:#000000">0.2854</span> |
|  | MSBRN+3DSFlabelling | **0.0150** **↑98.3%** | **0.9482** | **0.9601** |
|  | <span style="background-color:#000000">GMSF [3]</span> | <span style="background-color:#000000">7.2776</span> | <span style="background-color:#000000">0.0036</span> | <span style="background-color:#000000">0.0144</span> |
|  | GMSF+3DSFlabelling | **0.0093** **↑99.9%** | **0.9780** | **0.9880** |

| Dataset | Method | EPE3D↓ | Acc3DS↑ | Acc3DR↑ |
|:---:|:---:|:---:|:---:|:---:|
|  | <span style="background-color:#000000">FLOT [1]</span> | <span style="background-color:#000000">0.4858</span> | <span style="background-color:#000000">0.0821</span> | <span style="background-color:#000000">0.2669</span> |
|  | FLOT+3DSFlabelling | **0.0554** **↑88.6%** | **0.7601** | **0.8909** |
| nuScenes | <span style="background-color:#000000">MSBRN [2]</span> | <span style="background-color:#000000">0.6137</span> | <span style="background-color:#000000">0.2354</span> | <span style="background-color:#000000">0.2924</span> |
|  | MSBRN+3DSFlabelling | **0.0235** **↑96.2%** | **0.9413** | **0.9604** |
|  | <span style="background-color:#000000">GMSF [3]</span> | <span style="background-color:#000000">9.4231</span> | <span style="background-color:#000000">0.0034</span> | <span style="background-color:#000000">0.0086</span> |
|  | GMSF+3DSFlabelling | **0.0185** **↑99.8%** | **0.9534** | **0.9713** |

<a id="1">[1]</a> Puy G, Boulch A, Marlet R. Flot: Scene flow on point clouds guided by optimal transport[C]//European conference on computer vision. Cham: Springer International Publishing, 2020: 527-544.    
<a id="2">[2]</a> Cheng W, Ko J H. Multi-scale bidirectional recurrent network with hybrid correlation for point cloud based scene flow estimation[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023: 10041-10050.    
<a id="3">[3]</a> Zhang Y, Edstedt J, Wandt B, et al. Gmsf: Global matching scene flow[J]. Advances in Neural Information Processing Systems, 2024, 36.


## License and Citation <a name="license-and-citation"></a>

All assets and code are under the [Apache 2.0 license](./LICENSE) unless specified otherwise.

If this work is helpful for your research, please consider citing the following BibTeX entry.

``` bibtex
@inproceedings{yang2023vidar,
  title={3DSFLabelling: Boosting 3D Scene Flow Estimation by Pseudo Auto-labelling},
  author={Jiang, Chaokang and Wang, Guangming and Liu, Jiuming and Wang, Hesheng and Ma, Zhuang and Liu, Zhenqiang and Liang, Zhujin and Shan, Yi and Du, Dalong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```
