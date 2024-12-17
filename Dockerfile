FROM nvcr.io/nvidia/pytorch:23.11-py3
WORKDIR /workspace
ADD ./LLM-common-eval/requirements.txt r1.txt
RUN pip install -r r1.txt
ADD ./eagle/requirements.txt r2.txt
RUN pip install -r r2.txt
RUN pip install datasets==2.18
# setup the shell
RUN git clone https://github.com/t-k-cloud/tkarch.git && \
    pushd tkarch && cd dotfiles && ./overwrite.sh && \
    popd && rm -rf tkarch/.git
RUN apt update && apt install -y tmux git-lfs exuberant-ctags
RUN pip install nvitop
# add code
ADD . s3d
WORKDIR /workspace/s3d
CMD /bin/bash
