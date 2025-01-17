#FROM ubuntu:18.04

FROM nvidia/cuda:11.1.1-base-ubuntu20.04
RUN rm /etc/apt/sources.list.d/cuda.list
#RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub
CMD nvidia-smi

#ARG USER_ID
#ARG GROUP_ID
ENV USER srikanth
ENV HOME /home/$USER
ENV MPI_DIR=/opt/ompi
ENV PATH="$MPI_DIR/bin:$HOME/.local/bin:$PATH"
ENV LD_LIBRARY_PATH="$MPI_DIR/lib:$LD_LIBRARY_PATH"

RUN apt-get -q update \
    && apt-get install -y \
    python3 python3-dev python3-pip \
    gcc gfortran binutils \
    && pip3 install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ADD https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.3.tar.bz2 .
RUN tar xf openmpi-4.1.3.tar.bz2 \
    && cd openmpi-4.1.3 \
    && ./configure --prefix=$MPI_DIR \
    && make -j4 all \
    && make install \
    && cd .. && rm -rf \
    openmpi-4.1.3 openmpi-4.1.3.tar.bz2 /tmp/*

RUN pip install --upgrade pip
RUN pip install setuptools mpi4py
RUN which mpicc
RUN apt-get update && apt-get install -y openssh-client

ARG USER_ID
ARG GROUP_ID
RUN if getent user $USER ; then userdel $USER; fi
RUN if getent group $USER ; then groupdel $USER; fi
RUN groupadd -o --gid $GROUP_ID $USER
RUN useradd -l --uid $USER_ID --gid $GROUP_ID $USER
#RUN useradd -l --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER

#RUN groupadd -r $USER \
#    && useradd -r -g $USER $USER \
#    && chown -R $USER:$USER $HOME

RUN chown -R $USER:$USER $HOME
RUN chmod -R g+rwx $HOME


# If WORKDIR creates the directory, it's owned by root, but it needs to be owned by $USER
RUN mkdir $HOME/app && chown -R $USER:$USER $HOME/app/

USER $USER

RUN pip install --user torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
#RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
#RUN apt update
#RUN apt -y install wget
#RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
#RUN dpkg -i cuda-keyring_1.0-1_all.deb
#RUN apt update
#RUN apt-get install libnccl2 libnccl-dev
#ADD https://files.pythonhosted.org/packages/82/c8/13d273d42f60153b157a5d193e74ae3e1749745f3c2bf3c360b6d131d16e/blobfile-1.3.1-py3-none-any.whl .
#RUN pip install blobfile-1.3.1-py3-none-any.whl 


WORKDIR $HOME/app
ADD --chown=$USER:$USER requirements.txt .
#RUN ls -lh .
RUN pip install --user -r requirements.txt
# [No need now] Need to install gym_minigrid after installing babyai as babyai installs default gym_minigrid (without my modifications)
#RUN pip install gym_minigrid -e git+https://github.com/omsrisagar/gym-minigrid.git@master#egg=gym_minigrid
#RUN pip install jericho -e git+https://github.com/microsoft/jericho.git@6f761073ef064e62412c36cc8de569f57b39561c#egg=jericho



#USER srikanth


# Create a working directory and CD to it
#WORKDIR $HOME/app
#RUN ls -la /app
#RUN pwd
#RUN chown $USER:$USER -R /app/

#USER $USER
#RUN pip3 install --user -U setuptools \
#    && pip3 install --user mpi4py

#RUN pwd
#RUN ls -la .

# ADD babyai . will copy all contnents of babyai into current wdir. It will not copy the folder as a whole!
RUN echo $PATH
ADD --chown=$USER:$USER setup.py .
ADD --chown=$USER:$USER improved_diffusion ./improved_diffusion
ADD --chown=$USER:$USER scripts ./scripts

# Good cmds to debug
#RUN pwd
#RUN ls -la .

RUN pip install --user --editable .

RUN echo $PATH

#USER srikanth
#RUN pip install babyai -e git+https://github.com/omsrisagar/babyai.git@kg#egg=babyai

#CMD ["pwd"]
#RUN ls -la /babyai_kg/*

# Set the working directory to babayai/scripts
#WORKDIR /app/scripts/
#RUN pwd
#RUN ls -la .
#RUN chown /app
# Launch the run command
#CMD ["python", "train_rl.py", "--model debug_graph", "--procs 2", "--val-episodes 1"]
#CMD python train_rl.py --model debug_graph --procs 2 --val-episodes 1
