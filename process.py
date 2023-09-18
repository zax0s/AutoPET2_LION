from lionz import lion
import contextlib

import os
import torch
import SimpleITK
import glob
import shutil

def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))

TRACER_MODEL = "fdg"
ACCELERATOR = "cuda"

class Lion():
    def __init__(self):

        self.input_path_pet = '/input/images/pet/'  # according to the specified grand-challenge interfaces
        self.input_path_ct = '/input/images/ct/'  # according to the specified grand-challenge interfaces
        self.output_path = '/output/images/automated-petct-lesion-segmentation/'  # according to the specified grand-challenge interfaces
        self.lion_work_dir_input = '/workdir_input/'
        self.lion_work_dir_output = '/workdir_output/'
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        os.environ['OMP_NUM_THREADS']="1"
        os.environ['nnUNet_n_proc_DA']="0"
        os.environ['nnUNet_compile'] = 'F'

    def check_gpu(self):
        """
        Check if GPU is available
        """
        print('Checking GPU availability')
        is_available = torch.cuda.is_available()
        print('Available: ' + str(is_available))
        print(f'Device count: {torch.cuda.device_count()}')
        if is_available:
            print(f'Current device: {torch.cuda.current_device()}')
            print('Device name: ' + torch.cuda.get_device_name(0))
            print('Device memory: ' + str(torch.cuda.get_device_properties(0).total_memory))

    def convert_mha_nii(self, mha_input_path, nii_out_path):  #nnUNet specific
        img = SimpleITK.ReadImage(mha_input_path)
        SimpleITK.WriteImage(img, nii_out_path, True)

    def load_inputs(self):
        pet_mha = glob.glob(os.path.join(self.input_path_pet)+"*.mha")[0]
        self.convert_mha_nii(pet_mha, os.path.join(self.lion_work_dir_input, 'PT_input.nii.gz'))
        ct_mha = glob.glob(os.path.join(self.input_path_ct)+"*.mha")[0]
        self.convert_mha_nii(ct_mha, os.path.join(self.lion_work_dir_input, 'CT_input.nii.gz'))
        
    def set_output(self):
        pet_mha = os.path.basename(glob.glob(os.path.join(self.input_path_pet)+"*.mha")[0])
        print("pet_mha: ", pet_mha)
        prediction_nii = glob.glob(os.path.join(self.lion_work_dir_output)+"*.nii.gz")[0]
        print("prediction_nii: ", prediction_nii)
        print("Setting output to file: ", os.path.join(self.output_path, pet_mha))
        self.convert_mha_nii(prediction_nii, os.path.join(self.output_path, pet_mha))

    def predict(self):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                lion(TRACER_MODEL, self.lion_work_dir_input, self.lion_work_dir_output,ACCELERATOR)

    def clean_workdir(self):
        for f in os.listdir(self.lion_work_dir_input):
            remove(os.path.join(self.lion_work_dir_input, f))
        for f in os.listdir(self.lion_work_dir_output):
            remove(os.path.join(self.lion_work_dir_output, f))

    def process(self):
        self.check_gpu()
        print('Copy inputs to lion workdir')
        self.load_inputs()
        print('Start prediction')
        self.predict()
        print('Prediction complete')
        print('Copy outputs to grand-challenge output directory')
        self.set_output()
        print('Process complete')
        self.clean_workdir()

if __name__ == "__main__":
    Lion().process()




