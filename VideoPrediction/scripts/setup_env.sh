# Setting up Pytorch using Conda
# Originally written by Wonkwang Lee (https://github.com/1Konny)
# Edited by Yunseok Jang

if [ -z $1 ]; then
    echo -e '\033[0;31mYou should specify the name of the conda environment.\033[0m'
    echo 'E.g. setup_env.sh MY_ENV_NAME'
else
    source deactivate
    conda create --name $1 -y
    source deactivate
    source activate $1
    conda config --add channels conda-forge
    conda install -c conda-forge python numba scipy scikit-image opencv=2.4 ffmpeg -y
    conda install -c anaconda tensorflow-gpu=1.5 mkl cudatoolkit=8.0 cudnn=7.0 -y
    # upgrade pip/pip3
    pip install --upgrade pip --user

    # install python packages
    echo -e '\033[1;33install other dependencies\033[0m'
    pip install -r requirements.txt --user
fi
