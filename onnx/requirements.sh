# !/bin/sh
wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
bash Anaconda3-2021.05-Linux-x86_64.sh -b
rm Anaconda3-2021.05-Linux-x86_64.sh
export PATH=$PATH:/root/anaconda3/bin
conda install cudatoolkit=10.2 -c pytorch
conda install --insecure pytorch -c pytorch
conda install --insecure torchvision torchaudio -c pytorch

# pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
~  
