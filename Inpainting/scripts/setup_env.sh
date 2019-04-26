#! /bin/bash

# Setting up Pytorch using Conda
# Originally written by Wonkwang Lee (https://github.com/1Konny)

function checkconda(){
    if type "conda" > /dev/null 2>&1; then
        retval="true"
    else
        retval="false"
    fi

    echo "$retval"
}


function mkenv(){
    # mkenv : make conda environemnt
    # usage : mkcenv [env_name] [cuda_version]
    envname=$1
    cudaver=$2
    if [ -z $envname -o -z $cudaver ]; then
        echo 'You should specify the name of the conda environment with cuda version.'
        echo 'Usage: mkcenv [env_name] [cuda_version]'
        echo 'E.g. mkenv my_env 9'
    else
        if [ $(checkconda) = "false" ]; then
            echo "Can't find conda. check if conda is installed or its path is set properly."
        else
            if [ $cudaver = "8" ]; then
                echo 'setup conda environment including pytorch with cuda 8'
            elif [ $cudaver = "9" ]; then
                echo 'setup conda environment including pytorch with cuda 9'
            else
                echo 'cuda version should be either 8 or 9'
                exit 1
            fi

            conda create --name $envname python='3.6' --yes
            source deactivate
            source activate $envname
            conda install --yes mkl mkl-include ipython jupyter numpy matplotlib scipy cython

            if [ $cudaver = "8" ]; then
                echo 'install pytorch with cuda 8'
                conda install --yes pytorch=0.3.1 torchvision cuda80 -c pytorch
            elif [ $cudaver = "9" ]; then
                echo 'install pytorch with cuda 9'
                conda install --yes pytorch=0.3.1 torchvision cuda90 -c pytorch
            else
                echo 'cuda version should be either 8 or 9'
                exit 1
            fi

            # upgrade pip/pip3
            pip install --upgrade pip --user
            pip3 install --upgrade pip --user

            # install python packages
            pip install tqdm tensorflow-gpu tensorflow tensorboardX pyyaml --user
        fi
    fi
}

mkenv $1 $2
