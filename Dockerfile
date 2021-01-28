FROM python:3.8.2-slim-buster

RUN apt-get update && \
    apt-get install -y apt-transport-https build-essential cmake curl gcc g++ git python3-numpy tree sudo unzip wget

RUN useradd -m user -u 1000 && \
    echo 'user:user' | chpasswd user && \
    echo "user ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/user && \
    chmod 0440 /etc/sudoers.d/user && \
    chown -R user /home
USER user

COPY ./requirements.txt /tmp/requirements.txt
RUN sudo pip install --upgrade pip && \
    sudo pip install --ignore-installed -r /tmp/requirements.txt

RUN sudo jupyter contrib nbextension install && \
    jupyter nbextension enable toc2/main

ARG GIT_CONFIG_USER_NAME
ARG GIT_CONFIG_EMAIL
RUN git config --global user.name $GIT_CONFIG_USER_NAME && \
    git config --global user.email $GIT_CONFIG_EMAIL


WORKDIR /home/tutorial_predict_device_change
ENV PYTHONPATH=/home/tutorial_predict_device_change

CMD jupyter-notebook --ip=0.0.0.0 --port 8888 --no-browser --allow-root --NotebookApp.token=''