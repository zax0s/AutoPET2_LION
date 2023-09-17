# This script does a dummy run of moose to force download the model within the virtual env.
import lionz
import os
TRACER_MODEL = "fdg"
NNUNET_RESULTS_FOLDER = os.path.join('/usr/local', 'models', 'nnunet_trained_models')
lionz.download.model(TRACER_MODEL,NNUNET_RESULTS_FOLDER)