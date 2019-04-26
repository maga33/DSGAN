# Image-to-Image Translation

This code is the implementation of image-to-image translation task in [Diversity-Sensitive Conditional Generative Adversarial Networks](https://openreview.net/forum?id=rJliMh09F7) in ICLR 2019 [[Project Page]](https://sites.google.com/view/iclr19-dsgan/) [[Paper]](https://arxiv.org/abs/1901.09024).
This repository is based on [Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), [BicycleGAN](https://github.com/junyanz/BicycleGAN) and [Pix2PixHD](https://github.com/NVIDIA/pix2pixHD).

<img src='example/im2im.png'>

## Setup
1. Install `pytorch` and other dependencies except `tensorflow` use script: 

	```bash
	cd code && bash scripts/setup_env.sh $CONDA_ENV_NAME $CUDA_VERSION
	```
	This will create a conda enviroment named `$CONDA_ENV_NAME` to manage relavent packages.
	
	`CUDA_VERSION` supports `10`, `9`, `8`, `0`. `0` means installation with only CPU support.
2. Install `tensorflow` for [tensorboardX](https://github.com/lanpa/tensorboardX) usage. To install tensorflow, please refer to [the official webpage](https://www.tensorflow.org/install)

3. Set up datasets.
Put the dataset under `./data` folder. To download the datasets about **pix2pix**, please use `data/download_dataset.sh`. For example:

	```bash
	# To download dataset "facades"
	bash data/download_dataset.sh facades
	```
	For cityscapes dataset, please refer to the official website [cityscapes](https://www.cityscapes-dataset.com/). Download fine semantic segmentation dataset. Split all the files into `data/CityScapes/train` and `data/CityScapes/val` according to file names in pickle files [train](https://umich.box.com/shared/static/x2p68sl6sid0xdyq6zaewvnw7oqh4xq1.pickle) [val](https://umich.box.com/shared/static/1s8zquqa1zjzngrt74odsj1096a4644i.pickle). Put the two pickle files under `data/CityScapes`.
	
## File structure

```
root ----> data --> dataset folders
     ----> code --> main.py (Training and evaluation entrance)
     			--> trainer.py (Main training procedures and class 'GANTrainer' defined there)
     			--> cfg --> *.yml (Configuration file)
     			--> miscc 
     				--> utils.py (Visualization, parameter parsing, model loading and saving etc.)
     			        --> config.py (The definition file of hyperparameter setting)
     				--> layer_utils.py (Pytorch related layer small tools)
     				--> html.py (The file used to create a simple log website during training)
     				--> datasets_*.py (Data input pipeline files, defined under torch.data.Dataset class)
     				--> choose_dataset.py (Choose which data input pipeline according to given dataset name)
     			--> LPIPS (A submodule, for LPIPS evaluation)
                        --> FID (A submodule, for FID evaluation)   
                        --> exterior_models (Baseline models)
     				
     
```


## Training

```bash
cd code
# python main.py --cfg cfg/$YML_FILE_NAME --output_dir $MODEL_OUTPUT_DIRECTORY
python main.py --cfg cfg/facades_bicycleGAN_NE.yml --output ../output/facades_checkpoints
```
If there are already checkpoints in `$MODEL_OUTPUT_DIRECTORY`, this script will try to load the newest checkpoint according to the time stamp and continue training. Add `--no_cuda` flag to train or evaluate on CPU. 

For more other flags about training, please refer to `main.py, miscc/utils.py and miscc/config.py`

## Inference
1. Download pretrained model's checkpoint file using script:
	
	```bash
	cd code && bash scripts/download_pretrained.sh facades
	```
2. Infer some results from the pretrained model:

	```bash
	# under 'code' subfolder
	python main.py --cfg cfg/facades_bicycleGAN_NE.yml --eval --eval_num 10 --sample_num 20
	```
	Add `--no_cuda` to infer on CPU. 
	
	`--eval_num` means how many inputs from dataset needs to be inferred. `--sample_num` means for each input how many sample it will generate.
	
	After inference, under checkpoint folder there will be a folder named `eval`. There is a simple webpage under the `eval` folder for the convenience of browsing the inference results.
	
## Evaluation

There are three more evaluation flags. `--inter_eval` `--LPIPS_eval`, `--FID_eval`. If you don't specify any of the three, you just do random sampling.  

Evaluation (apart from LPIPS evaluation, FID evaluation), will make a folder at your checkpoint folder of corresponding experiment named *inter_eval* or *eval* (`$OUTPUT_DIR/eval`, `$OUTPUT_DIR/inter_eval`, related generated image results will be under `$EVALUATION_FOLDER/Image`). Evaluation of the metrics (FID or LPIPS scores) will create a `.txt` file recording the evaluation scores.

For example, 

```bash
# Under 'code' folder

# FID evaluation
# Generate 100 * 20 images and calculate the FID score between the generated images and training dataset.
python main.py --fid_eval --cfg cfg/$YML_FILE_NAME --output_dir $OUTPUT_DIR --eval_num 100 --sample_num 20

# LPIPS evaluation
# On 100 inputs (conditions), generate 20 pairs of images, calculate their LPIPS score. 
# Run 20 times (--try_num) and get the mean and variance. 
python main.py --LPIPS_eval --cfg cfg/$YML_FILE_NAME --output_dir $OUTPUT_DIR --eval_num 100 --sample_num 20 --try_num 20

# Interpolation evaluation
# On 100 inputs, for each select 2 random latent vectors, and generate 20 interpolation results between the two vectors.
python main.py --inter_eval --cfg cfg/$YML_FILE_NAME --output_dir $OUTPUT_DIR --eval_num 100 --sample_num 20

```

For more information of evaluartion or flags like `--eval_all, --gen_paths` etc, please refer to `main.py, miscc/utils.py` to see related evaluation flag definitions.

