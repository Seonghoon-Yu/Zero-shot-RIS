# Zero-shot Referring Image Segmentation with Global-Local Context Features
This repogitory store the code for implementing the Global-Local CLIP algorithm for zero-shot referring image segmentation.

<p align="center"> <img src="https://user-images.githubusercontent.com/75726938/222959862-51826d1e-b082-4f58-8e91-65abcc6d4a5c.PNG" width="700" align="center"> </p>

> [**Zero-shot Referring Image Segmentation with Global-Local Context Features**](https://arxiv.org/abs/2303.17811)  
> [Seonghoon Yu](https://scholar.google.com/citations?user=VuIo1woAAAAJ&hl=ko), [Paul Hongsuck Seo](https://phseo.github.io/), [Jeany Son](https://jeanyson.github.io/)  
> AI graduate school, GIST and Google Research  
> CVPR 2023  

[arxiv](https://arxiv.org/abs/2303.17811) | [pdf](https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_Zero-Shot_Referring_Image_Segmentation_With_Global-Local_Context_Features_CVPR_2023_paper.pdf) | [video](https://www.youtube.com/watch?v=X_37jodjz2Y) | [poster](https://github.com/Seonghoon-Yu/Zero-shot-RIS/assets/75726938/d9973d1d-d764-4dbf-bcff-384e48ff52b5) | [bibtex](#citation)


## Installation
```shell
# cteate conda env
conda create -n zsref python=3.8

# activate the environment
conda activate zsref

# Install Pytorch 1.10 version with GPU
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge

# Install spacy for language processing
conda install -c conda-forge spacy
pip install -U pydantic
python -m spacy download en_core_web_lg

# Install required package
pip install opencv-python
pip install scikit-image
pip install h5py
conda install -c conda-forge einops
pip install markupsafe==2.0.1

```

## Citation
