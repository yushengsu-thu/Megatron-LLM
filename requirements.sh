####SERVER#####
#No LSB modules are available.
#Distributor ID:	Ubuntu
#Description:	Ubuntu 22.04.2 LTS
#Release:	22.04
#Codename:	jammy

####GITHUB Repo#####
#https://github.com/epfLLM/Megatron-LLM

####CUDA and GPU VSERSION####
#CUDA Version: 12.0
#GPU A100 80G


## load env and create the same env



##Install by yuorself




###################
# Install conda env
conda create --name myenv python=3.12.2
###################

# install cuda
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

#In my case, I removed "flash-attn" from requirements.txt and ran
pip install -r requirements.txt
#After installation of the other packages, then ran
pip install flash-attn --no-build-isolation

## Must follow
pip install packaging==24.0

## Must follow (can use the existing apex. You alos clone from the github website and find use the following commit id)
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
cd..

pip install tensorboard>=2.16

## Build Megatron
python setup.py install

# compile helper.cpp
cd megatron/data
make
cd ../..



#######################################################
#####The newer megatron might need: magatron.core######
#######################################################


'''
pip install six==1.16.0
pip install regex==2023.12.25
'''


