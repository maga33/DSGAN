# download the pretrained model
if [ ! -d checkpoints/CENoise_noiseDim_32_lambda_5.000_outputDistperceptual_trans_ctr/ ]; then
    mkdir -p checkpoints/CENoise_noiseDim_32_lambda_5.000_outputDistperceptual_trans_ctr/
fi
wget https://umich.box.com/shared/static/ot4sng6jsdgyvdpcfv0qzn6paswus1jw.pth -c0 checkpoints/CENoise_noiseDim_32_lambda_5.000_outputDistperceptual_trans_ctr/netG_latest.pth
