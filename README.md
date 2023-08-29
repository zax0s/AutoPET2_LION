# LION Algorithm autopet-ii grand-challenge

The source code for the algorithm container for
LION (MOOSE 2.0 tumour segmentation), generated with
evalutils version 0.4.2 using Python 3.10.

## Build Guide 🛠️
To build the docker image , simply run the [build.sh](https://github.com/zax0s/AutoPET2_LION/blob/main/build.sh) script. 
Building the docker image will triger the [model_download.py](https://github.com/zax0s/AutoPET2_LION/blob/main/model_download.py) file which will force download the model within the container. 
The built container includes the model which was dowloaded during built time. 

## Export Guide 
Run the [export.sh](https://github.com/zax0s/AutoPET2_LION/blob/main/export.sh) script to save the docker image to a .tar.gz file. 

## Prediction :computer: 
A wraper class was built to handle the data I/O in the container, within the [process.py](https://github.com/zax0s/AutoPET2_LION/blob/main/process.py)
Moose prediction is handled within the wraper and then the output is set to the challenge required directory. 

## Testing :computer: 
Testing is needed to make sure the Docker image is not using more than the allowed resources. 
It also tests the naming and location of the output. 
To run the test, use [test.sh](https://github.com/zax0s/AutoPET2_LION/blob/main/test.sh)

For more details on the challenge requirements see [autopet-ii.grand-challenge](https://autopet-ii.grand-challenge.org/submission)



