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
        'cityscapes') URL="https://www.dropbox.com/s/bc2bg9736jppshn/cityscapes_ne_checkpoint.tar.gz";;
        'facades') URL="https://www.dropbox.com/s/6lfiuuam4imbyv9/facades_ne_checkpoint.tar.gz";;
        'maps') URL="https://www.dropbox.com/s/5e4fhhs4ten1b57/maps_ne_checkpoint.tar.gz";;
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
