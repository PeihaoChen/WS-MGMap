# Setup

## clone this code
```bash
git clone https://github.com/PeihaoChen/WS-MGMap.git
cd WS-MGMap
```

## Python
This project is developed with Python 3.6.13. If you are using miniconda or anaconda, you can create an environment:

```bash
conda create -n wsmgmap python==3.6.13
conda activate wsmgmap
```

## Pytorch
VLN-CE uses Pytorch 1.6.0 & Cuda 10.2 which can be built installed from conda:

```bash
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
```

## Habitat
VLN-CE uses Habitat-Sim 0.1.5 which can be built from source or installed from conda:

```bash
conda install -y -c aihabitat -c conda-forge bullet=2.88 habitat-sim=0.1.5 headless withbullet python=3.6
```
Tips: You'd better to install bullet and withbulllet simultaneously, in order to avoid ImportError at run time.

Then install Habitat-Lab:

```bash
git clone --branch v0.1.5 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
# installs both habitat and habitat_baselines
pip install --upgrade pip   # update pip
python -m pip install -r requirements.txt

python -m pip install -r habitat_baselines/rl/requirements.txt
python -m pip install -r habitat_baselines/rl/ddppo/requirements.txt
python setup.py develop --all
```

## WS-MGMap for VLN
```bash
cd ..
pip install -r requirements.txt

# requirements
conda install psutil 
pip install einops 

# torch_scatter
cd data
wget https://data.pyg.org/whl/torch-1.6.0%2Bcu102/torch_scatter-2.0.6-cp36-cp36m-linux_x86_64.whl
pip install torch_scatter-2.0.6-cp36-cp36m-linux_x86_64.whl
cd ..
```

# Data 
```bash
# Fisrt install the gdown to download data in google drive.
pip install gdown

mkdir data
cd data
```

## Semantic Map
```bash
# Download map_data.tar.gz
gdown https://drive.google.com/uc?id=1pJwx0E95WsJXThcx8tPrUTB_6gTlryoy
tar -xvf map_data.tar.gz

# Unzip all train files
cd map_data/semantic/train
find . -name '*.tar.gz' -print0 | xargs -0 -I {} -P 10 tar -zvxf {}

# Unzip all train_aug files
cd ../train_aug
find . -name '*.tar.gz' -print0 | xargs -0 -I {} -P 10 tar -zvxf {}
```

## Pre-Trained Model
```bash
gdown https://drive.google.com/uc?id=1DYkXbRIBVgMU1qHF_mLT41esSAdcQJaf
tar -zxvf pretrain_model.tar.gz
```

## Trained model
```bash
gdown https://drive.google.com/uc?id=1HcD8s-tyBeH2LsXs6Rj5x5DC1hVD4GNs
tar -zxvf trained_model.tar.gz
```
