# download the pretrained model
if [ ! -d checkpoints/CENoise_noiseDim_32_lambda_5.000_outputDistperceptual_trans_ctr/ ]; then
    mkdir -p checkpoints/CENoise_noiseDim_32_lambda_5.000_outputDistperceptual_trans_ctr/
fi
wget https://www.dropbox.com/s/ec1yhfi716o65ur/netG_latest.pth -c0 checkpoints/CENoise_noiseDim_32_lambda_5.000_outputDistperceptual_trans_ctr/netG_latest.pth
