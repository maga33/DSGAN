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
        echo -e '\033[0;31mYou should specify the name of the conda environment with cuda version.\033[0m'
        echo 'Usage: mkcenv [env_name] [cuda_version]'
        echo 'E.g. mkenv my_env 9'
    else
        if [ $(checkconda) = "false" ]; then
            echo -e "\033[0;31mCan't find conda. check if conda is installed or its path is set properly.\033[0m"
        else
            if [ $cudaver = "8" ]; then
                echo -e '\033[1;33msetup conda environment including pytorch with cuda 8\033[0m'
            elif [ $cudaver = "9" ]; then
                echo -e '\033[1;33msetup conda environment including pytorch with cuda 9\033[0m'
            elif [ $cudaver = "10" ]; then
                echo -e '\033[1;33msetup conda environment including pytorch with cuda 10\033[0m'
            elif [ $cudaver = "0" ]; then
                echo -e '\033[1;33mstep up conda enviroment including with no cuda support\033[0m'
            else
                echo -e '\033[0;31mcuda version should be either 8 or 9 or 10\033[0m'
                exit 1
            fi

            conda create --name $envname python='2.7' --yes
            source deactivate
            source activate $envname
            conda install --yes mkl mkl-include ipython jupyter numpy matplotlib scipy cython

            if [ $cudaver = "8" ]; then
                echo -e '\033[1;33install pytorch with cuda 8\033[0m'
                conda install pytorch torchvision cudatoolkit=8.0 -c pytorch
            elif [ $cudaver = "9" ]; then
                echo -e '\033[1;33install pytorch with cuda 9\033[0m'
                conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
            elif [ $cudaver = "10" ]; then
                echo -e '\033[1;33install pytorch with cuda 10\033[0m'
                conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
            elif [ $cudaver = "0" ]; then
                echo -e '\033[1;33install pytorch with only cpu support\033[0m'
                conda install pytorch-cpu torchvision-cpu -c pytorch
            else
                echo -e '\033[0;31mcuda version should be either 8 or 9 or 10\033[0m'
                exit 1
            fi

            # upgrade pip/pip3
            pip install --upgrade pip --user

            # install python packages
            echo -e '\033[1;33install other dependencies\033[0m'
            pip install -r ../requirements.txt --user
        fi
    fi
}

mkenv $1 $2
