FROM python:3.10.12-bookworm
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /input /output \
    && chown user:user /input /output

RUN mkdir -p /opt/app /workdir_input /workdir_output /opt/app/empty \
    && chown user:user /opt/app /workdir_input /workdir_output /opt/app/empty

RUN mkdir -p /usr/local/models/nnunet_trained_models \
    && chown user:user /usr/local/models/nnunet_trained_models    

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install torch torchvision torchaudio
RUN python -m pip install --user -rrequirements.txt

COPY --chown=user:user LION_custom_trainers.py /home/user/.local/lib/python3.10/site-packages/nnunetv2/training/nnUNetTrainer/variants/LION_custom_trainers.py
# COPY --chown=user:user LION /opt/app/LION
# RUN python -m pip install --user /opt/app/LION

COPY --chown=user:user model_download.py /opt/app/
RUN python model_download.py
RUN python -m pip install --upgrade batchgenerators

COPY --chown=user:user process.py /opt/app/

ENTRYPOINT [ "python", "-m", "process" ]
#ENTRYPOINT ["/bin/bash"]