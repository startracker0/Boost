# Boosting Single-Domain Generalized Object Detection via Vision-Language Knowledge Interaction

**Official Implementation of the paper:**  
📄 [_"Boosting Single-Domain Generalized Object Detection via Vision-Language Knowledge Interaction"_](https://arxiv.org/pdf/2504.19086)  
✅ **Accepted at ACM Multimedia (ACMMM) 2025**

---

## 🔍 Introduction

This is the official PyTorch implementation of our ACMMM 2025 paper.  
We study the challenging setting of **Single-Domain Generalized Object Detection (S-DGOD)**, where the detector must generalize to unseen domains without accessing any target domain data.

---

## 🚀 Getting Started

### 1. Environment Setup
```bash
git clone https://github.com/startracker0/Boost.git
cd PhysAug

conda create -n boost python=3.8 -y
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116

pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"
pip install -v -e .

pip install -r requirements.txt
```
To ensure reproducibility, the detailed environment dependencies are provided in requirements.txt and environment.yaml

### 2. Prepare Data
Download and prepare the dataset required for the experiments. Update the dataset path in the configuration file.

#### DWD Dataset
You can download the DWD dataset from the following link:
[Download DWD Dataset](https://drive.google.com/drive/folders/1IIUnUrJrvFgPzU8D6KtV0CXa8k1eBV9B)

#### Cityscapes-C Dataset
The Cityscapes dataset can be downloaded from the official website:
[Download Cityscapes Dataset](https://www.cityscapes-dataset.com/)

We generate the Cityscapes-C validation set based on the cityscapes/leftImg8bit/val portion of the dataset.
You can create this dataset using the [imagecorruptions](https://github.com/bethgelab/imagecorruptions) library, which provides various corruption functions to simulate adverse conditions such as noise, blur, weather, and digital artifacts.

```bash
git clone https://github.com/bethgelab/imagecorruptions.git
cd imagecorruptions
pip install -v -e .
python gen_cityscapes_c.py
```

The datasets should be organized as follows:
```bash
datasets/
├── DWD/
│   ├── daytime_clear/
│   ├── daytime_foggy/
│   ├── dusk_rainy/
│   ├── night_rainy/
│   └── night_sunny/
├── Cityscapes-c/
│   ├── brightness/
│   ├── contrast/
│   ├── defocus_blur/
........
│   └── zoom_blur/
```
### 3. Training the Model

To train the model using PhysAug, follow these steps:

1. Ensure the dataset paths are correctly configured in `configs/_base_/datasets/dwd.py` and `configs/_base_/datasets/cityscapes_detection.py`.
2. Run the following command to start training:

```bash
bash train_dwd.sh
bash train_cityscapes_c.sh
```

The checkpoints are available in [this link](https://pan.baidu.com/s/1MQCG-u2nylz4ii3qy4i8Xw?pwd=xuja)
### 4. Evaluating the Model

To evaluate the trained model, follow these steps:

1. Specify the dataset to evaluate (e.g., DWD, Cityscapes, or Cityscapes-C).
2. Run the evaluation script with the following command:

```bash
bash test.sh
```
---

## 📬 Contact

For questions or collaborations, feel free to contact me.

---

## 📌 Acknowledgements

This repository is built upon [MMDetection](https://github.com/open-mmlab/mmdetection) and [OpenCLIP](https://github.com/mlfoundations/open_clip). We thank the authors for their contributions to the community.
