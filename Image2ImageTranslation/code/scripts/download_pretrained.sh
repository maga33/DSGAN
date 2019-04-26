EXP_NAME=$1

# Available checkpoints of experiments
if [[ $EXP_NAME != "cityscapes" &&  $EXP_NAME != "facades" && $EXP_NAME != "maps" ]]; then
  echo "Availabel pretrained models are cityscapes, facades, maps"
  exit 1
fi

echo "Specified [$EXP_NAME]"

# Get the download url according to the experiment name
url_parse() {
    case $1 in
        'cityscapes') URL="https://umich.box.com/shared/static/9jtuz3nhfeoed2us4s8p98lia9k5nsy6.gz";;
        'facades') URL="https://umich.box.com/shared/static/yyjvlmw22tz2hmddiggydwidrgedh3o9.gz";;
        'maps') URL="https://umich.box.com/shared/static/nliphdbk968p5cy65vy5gksgq872288n.gz";;
    esac
}
url_parse $EXP_NAME


if [ ! -d ../output/ ]; then
    mkdir -p ../output
fi

# Download and decompress the checkpoints folder
wget $URL -O ../output/${EXP_NAME}_checkpoints.tar.gz
tar -xvzf ../output/${EXP_NAME}_checkpoints.tar.gz -C ../output
rm ../output/${EXP_NAME}_checkpoints.tar.gz
