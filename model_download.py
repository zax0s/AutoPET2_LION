# This script does a dummy run of moose to force download the model within the virtual env.
import lionz
import os
TRACER_MODEL = "fdg"
NNUNET_RESULTS_FOLDER = os.path.join('/usr/local/models/nnunet_trained_models/')
# create the nnuent_results_folder if it does not exist
if not os.path.exists(NNUNET_RESULTS_FOLDER):
    os.makedirs(NNUNET_RESULTS_FOLDER)
# download the model
lionz.download.model(TRACER_MODEL,NNUNET_RESULTS_FOLDER)