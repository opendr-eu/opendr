FROM ubuntu:20.04

ARG branch=master
ARG ros_distro=noetic

# Install dependencies
RUN apt-get update && \
    apt-get --yes install git sudo && \
    DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata

# Add Tini
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]

# Clone the repo and install the toolkit
RUN git clone --depth 1 --recurse-submodules -j8 https://github.com/opendr-eu/opendr -b $branch
WORKDIR "/opendr"
ENV ROS_DISTRO=$ros_distro
RUN chmod +x ./bin/install.sh && ./bin/install.sh && rm -rf /root/.cache/* && apt-get clean

# Create script for starting Jupyter Notebook
RUN /bin/bash -c "source ./bin/activate.sh; pip3 install jupyter" && \
    echo "#!/bin/bash\n source ./bin/activate.sh\n ./venv/bin/jupyter notebook --port=8888 --no-browser --ip 0.0.0.0 --allow-root" > start.sh && \
    chmod +x start.sh

# Start Jupyter Notebook inside OpenDR
CMD ["./start.sh"]
