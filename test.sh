#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
# Maximum is currently 30g, configurable in your algorithm image settings on grand challenge
MEM_LIMIT="30g"

docker volume create lion-output-$VOLUME_SUFFIX

# Do not change any of the parameters to docker run, these are fixed
docker run --rm \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        --cpus=4 \
        --gpus=all \
        -v $SCRIPTPATH/test/input/:/input/ \
        -v lion-output-$VOLUME_SUFFIX:/output/ \
        lion

# compare the outputs in the Docker volume with the outputs in ./test/expected_output/
docker run --rm \
        -v lion-output-$VOLUME_SUFFIX:/output/ \
        -v $SCRIPTPATH/test/expected_output/:/expected_output/ \
        biocontainers/simpleitk:v1.0.1-3-deb-py3_cv1 python3 -c """
        
import SimpleITK as sitk

output = sitk.ReadImage('/output/images/automated-petct-lesion-segmentation/PET_VPat002.mha')
expected_output = sitk.ReadImage('/expected_output/images/automated-petct-lesion-segmentation/PET_VPat002.mha')

label_filter = sitk.LabelOverlapMeasuresImageFilter()
label_filter.Execute(output, expected_output)
dice_score = label_filter.GetDiceCoefficient()

if dice_score == 1.0:
    print('Test passed!')
else:
    print('Test failed!')
"""

docker volume rm lion-output-$VOLUME_SUFFIX
