#https://github.com/epfLLM/Megatron-LLM

'''
# load env and create the same env
conda env create -f environment_megatron_lm.yml


## Must follow
pip install packaging==24.0

## Must follow
#cd apex
wget https://github.com/NVIDIA/apex/archive/refs/tags/23.08.zip
unzip 23.08
pip uninstall apex
cd apex-23.08
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
cd..
'''

pip install -r requirement.txt

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


