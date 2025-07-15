# eSPARK

The source code of article 'A Multimodal Synergistic Model for Personalized Neoadjuvant Immunochemotherapy in Esophageal Cancer'.

## System requirements
This code was developed and tested in the following settings. 
### OS
- Ubuntu 20.04
### GPU
- Nvidia GeForce RTX 2080 Ti
### Dependencies
- Python (3.8.12)
- Pytorch install = 1.10.1
- torchvision (0.10.0)
- CUDA (11.3)
## Installation guide

- Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html#linux-installers) on your machine (download the distribution that comes with python3).  
  
- After setting up Miniconda, install OpenSlide (3.4.1):  
```
apt-get install openslide-python
```
- Create a conda environment with espark.yaml:
```
conda env create -f espark.yaml
```  
- Activate the environment:
```
conda activate espark
```
- Typical installation time: 1 hour

## Preprocessing
We use the python files to convert the WSI to patches with size 256*256 pixels and taking color normalization for comparison.
### Slide directory structure
```
DATA_ROOT_DIR/
    └──case_id/
        ├── slide_id.svs
        └── ...
    └──case_id/
        ├── slide_id.svs
        └── ...
    ...
```
### Generating patches
- /preprocessing/generate_patch.py
## Development of eSPARK
Training of MScaleCT
```
python single_modal_ct.py
```
Training of CytoPath
```
python single_modal_patho.py
```
Training of eSPARK
```
python multimodal.py

