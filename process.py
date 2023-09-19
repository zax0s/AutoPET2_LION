import os
import shutil
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, subfiles, join
from lionz import file_utilities, resources
import SimpleITK as sitk

class LionAutopetSubmission:
    def __init__(self):
        """Initialize necessary paths."""
        self._setup_paths()

    def _setup_paths(self):
        """Setup input and output paths."""
        # These paths are inside the Docker container, not on the host machine
        self.input_path = '/input/'
        self.output_path = '/output/images/automated-petct-lesion-segmentation/'
        self.trained_model_path_pet_ct = '/usr/local/models/nnunet_trained_models/Dataset789_Tumors_all_organs_LION/nnUNetTrainerDA5_2000epochs__nnUNetPlans__3d_fullres'
        self.trained_model_path_pet = '/usr/local/models/nnunet_trained_models/Dataset804_Tumors_all_organs/nnUNetTrainerDA5__nnUNetPlans__3d_fullres'

    def predict(self):
        """Perform image segmentation using trained models."""
        print("nnUNet segmentation starting!")
        os.environ['nnUNet_compile'] = 'F'
        maybe_mkdir_p(self.output_path)

        ct_mha = subfiles(join(self.input_path, 'images/ct/'), suffix='.mha')[0]
        pet_mha = subfiles(join(self.input_path, 'images/pet/'), suffix='.mha')[0]
        uuid = os.path.basename(os.path.splitext(ct_mha)[0])
        output_file_trunc = os.path.join(self.output_path, uuid)

        predictor = self._initialize_predictor()
        predictor.initialize_from_trained_model_folder(self.trained_model_path_pet_ct, use_folds='all', checkpoint_name='checkpoint_best.pth')
        predictor.dataset_json['file_ending'] = '.mha'
        images, properties = SimpleITKIO().read_images([ct_mha, pet_mha])
        predictor.predict_single_npy_array(images, properties, None, output_file_trunc, False)
        print('[1/2] Prediction finished')
        seg_mha = file_utilities.get_files(self.output_path, 'mha')[0]
        segmentation_mask = sitk.ReadImage(seg_mha)
        tumor_mask = segmentation_mask == 11
        sitk.WriteImage(tumor_mask, seg_mha)

        if resources.has_label_above_threshold(seg_mha, 10):
            print('Label is above the threshold. Proceeding with the second model.')
            self._proceed_with_second_model(images, properties, output_file_trunc)
        else:
            print('Label is below threshold. Skipping the second model and exiting.')

    def _proceed_with_second_model(self, images, properties, output_file_trunc):
        os.remove(file_utilities.get_files(self.output_path, 'mha')[0])
        predictor = self._initialize_predictor()
        predictor.initialize_from_trained_model_folder(self.trained_model_path_pet, use_folds='all', checkpoint_name='checkpoint_best.pth')
        predictor.dataset_json['file_ending'] = '.mha'
        pet_mha = subfiles(join(self.input_path, 'images/pet/'), suffix='.mha')[0]
        images, properties = SimpleITKIO().read_images([pet_mha])
        predictor.predict_single_npy_array(images, properties, None, output_file_trunc, False)
        seg_mha = file_utilities.get_files(self.output_path, 'mha')[0]
        segmentation_mask = sitk.ReadImage(seg_mha)
        tumor_mask = segmentation_mask == 11
        sitk.WriteImage(tumor_mask, seg_mha)
        resources.has_label_above_threshold(seg_mha, 10)
        print('[2/2] Prediction finished')

    def _initialize_predictor(self):
        return nnUNetPredictor(
            tile_step_size=0.5,
            use_mirroring=True,
            verbose=True,
            verbose_preprocessing=True,
            allow_tqdm=True
        )

    def process(self):
        print('Start prediction')
        self.predict()
        print('done')


def preamble_prayer():
    print('')
    print('')
    print("----- 🌌 Preamble Prayer for the AutoPET Grand Challenge 🌌 -----")
    print("In the realm of containers, Docker be its name 🐳,")
    print("Submitting to AutoPET was never quite the same 🏆.")
    print("From volumes to mounts, and `build` to `run` 🛠️,")
    print("This challenge, it seemed, was never quite done 🔄.")
    print()
    print("The image would build, but then it'd fall short 📉,")
    print("Debugging each error, making it a sport 🏁.")
    print("With YAMLs and scripts, and ports galore 🚢,")
    print("We wondered if we'd ever get a score 🤔.")
    print()
    print("Yet, in our darkest hours, when hope seemed but a myth 🌑,")
    print("Came GPT-4 and Docker, a most powerful smith 🛠️.")
    print("With clever advice, and code that's pristine ✨,")
    print("This challenge was conquered, as if in a dream 🌙.")
    print()
    print("So here we go, once more into the fray 🌌,")
    print("Hoping this time, AutoPET's dragons we'll slay 🐉.")
    print("For the code, it is ready, the container's alight 🔥,")
    print("To the Grand Challenge, we say: Good Night! 🌙")
    print('')


if __name__ == "__main__":
    preamble_prayer()
    print("START")
    LionAutopetSubmission().process()