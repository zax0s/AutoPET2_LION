# Starting with the Python bookworm image
FROM python:3.10.12-bookworm

# Installing required libraries
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# Create a non-root user
RUN groupadd -r user && useradd -m --no-log-init -r -g user user

# Create necessary directories and change their ownership to our user
RUN mkdir -p /input /output /opt/app /opt/app/empty /usr/local/models/nnunet_trained_models \
    && chown user:user /input /output /opt/app /opt/app/empty /usr/local/models/nnunet_trained_models

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"
ENV nnUNet_results='/usr/local/models/nnunet_trained_models/'

# Update pip and install pip-tools
RUN python -m pip install --user -U pip && python -m pip install --user pip-tools

# Copy requirements and install them
COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install torch torchvision torchaudio
RUN python -m pip install -r /opt/app/requirements.txt

# Copying the custom trainer into the nnunetv2 path
COPY --chown=user:user LION_custom_trainers.py /home/user/.local/lib/python3.10/site-packages/nnunetv2/training/nnUNetTrainer/variants/LION_custom_trainers.py

# Install LION and nnUNet from GitHub
RUN pip install git+https://github.com/LalithShiyam/LION.git
RUN pip install --upgrade git+https://github.com/MIC-DKFZ/nnUNet.git

# Download models
COPY --chown=user:user model_download.py /opt/app/
RUN python model_download.py
RUN python -m pip install --upgrade batchgenerators

# Copy process script
COPY --chown=user:user process.py /opt/app/

# Using bash as the default command when the container starts
ENTRYPOINT ["python", "/opt/app/process.py"]