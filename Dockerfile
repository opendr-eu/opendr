FROM ubuntu:20.04

# Install dependencies
RUN apt-get update && \
    apt-get --yes install git sudo
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata

# Add Tini
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]

# Clone the repo and install the toolkit
RUN git clone https://github.com/tasostefas/opendr_internal
WORKDIR "/opendr_internal"
RUN git checkout install_scripts
RUN ./bin/install.sh

# Create script for starting Jupyter Notebook
RUN pip3 install jupyter
RUN echo "#!/bin/bash\n source ./bin/activate.sh\njupyter notebook --port=8888 --no-browser --ip 0.0.0.0 --allow-root" > start.sh
RUN chmod +x start.sh

# Start Jupyter Notebook inside OpenDR
CMD ["./start.sh"]
