# Zero-shot Referring Image Segmentation with Global-Local Context Features
This repogitory store the code for implementing the Global-Local CLIP algorithm for zero-shot referring image segmentation.

<p align="center"> <img src="https://user-images.githubusercontent.com/75726938/222959862-51826d1e-b082-4f58-8e91-65abcc6d4a5c.PNG" width="700" align="center"> </p>

> [**Zero-shot Referring Image Segmentation with Global-Local Context Features**](https://openaccess.thecvf.com/content/CVPR2023/html/Yu_Zero-Shot_Referring_Image_Segmentation_With_Global-Local_Context_Features_CVPR_2023_paper.html)  
> [Seonghoon Yu](https://scholar.google.com/citations?user=VuIo1woAAAAJ&hl=ko), [Paul Hongsuck Seo](https://phseo.github.io/), [Jeany Son](https://jeanyson.github.io/)  
> AI graduate school, GIST and Google Research  
> CVPR 2023  

[paper](https://openaccess.thecvf.com/content/CVPR2023/html/Yu_Zero-Shot_Referring_Image_Segmentation_With_Global-Local_Context_Features_CVPR_2023_paper.html) | [arxiv](https://arxiv.org/abs/2303.17811) | [video](https://www.youtube.com/watch?v=X_37jodjz2Y) | [poster](https://github.com/Seonghoon-Yu/Zero-shot-RIS/assets/75726938/9ac2da28-f522-4fef-b672-fbd078d40155) | [tutorial](https://github.com/Seonghoon-Yu/Zero-shot-RIS/blob/master/KCCV2023_tutorial.ipynb) | [bibtex](#citation)


## Installation
### 1. Environment
```shell
# cteate conda env
conda create -n zsref python=3.8

# activate the environment
conda activate zsref

# Install Pytorch 1.10 version with GPU
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge

# Install spacy for language processing
conda install -c conda-forge spacy
pip install pydantic==1.10.11 --upgrade
python -m spacy download en_core_web_lg

# Install required package
pip install opencv-python
pip install scikit-image
pip install h5py
conda install -c conda-forge einops
pip install markupsafe==2.0.1
```
### 2. Third Party
```shell
# Install modified CLIP in a dev mode
cd third_parth
cd modified_CLIP
pip install -e .

# Install detectron2 for FreeSOLO
cd ..
cd old_detectron2
pip install -e .
pip install pillow==9.5.0
```

### 3. Download FreeSOLO pre-trained weiths
we use [FreeSOLO](https://github.com/NVlabs/FreeSOLO) which is an unsupervised instance segmentation model as the mask generator
```shell
mkdir checkpoints
cd checkpoints
wget https://cloudstor.aarnet.edu.au/plus/s/V8C0onE5H63x3RD/download
mv download FreeSOLO_R101_30k_pl.pth
```

## Dataset
we follow [dataset setup](https://github.com/yz93/LAVT-RIS/tree/main/refer) in [LAVT](https://github.com/yz93/LAVT-RIS)
### 1. Download COCO 2014 train images
In "./refer/data/images/mscoco/images" path
```shell
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014
```

### 2. Download RefCOCO, RefCOCO+, and RefCOCOg annotations 
In "./refer/data" path
```shell
# RefCOCO
wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip
unzip refcoco.zip

# RefCOCO+
wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip
unzip refcoco+.zip

# RefCOCOg
wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip
unzip refcocog.zip
```

## Evaluation
To evaluate a model's performance on RefCOCO variants, use
```shell
python Our_method_with_free_solo.py --dataset refcoco --split val
```
For options,  
--dataset: refcoco, refcoco+, refcocog  
--split: val, testA, testB for refcoco and val, test for refcocog  

## Citation
Please consider citing our paper in your publications, if our findings help your research.
```
@InProceedings{Yu_2023_CVPR,
    author    = {Yu, Seonghoon and Seo, Paul Hongsuck and Son, Jeany},
    title     = {Zero-Shot Referring Image Segmentation With Global-Local Context Features},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {19456-19465}
}
```

## Acknowledgements
Code is built upon several public repositories.
- Evaluation Metric and Dataset Preparation: [LAVT](https://github.com/yz93/LAVT-RIS)  
- Base Backbone code: [CLIP](https://github.com/openai/CLIP)  
- Mask Generator: [FreeSOLO](https://github.com/NVlabs/FreeSOLO)  

Thanks.
