# This script does a dummy run of moose to force download the model within the virtual env.
import moosez
import os
MOOSE_MODEL = "clin_pt_fdg_tumor"
NNUNET_RESULTS_FOLDER = os.path.join('/usr/local', 'models', 'nnunet_trained_models')
moosez.download.model(MOOSE_MODEL,NNUNET_RESULTS_FOLDER)