FROM nvidia/cuda:9.1-devel-ubuntu16.04

RUN apt-get update && \
    apt-get install -y software-properties-common python-software-properties apt-transport-https

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9 && \
    add-apt-repository 'deb [arch=amd64] https://cran.rstudio.com/bin/linux/ubuntu xenial/' && \
    apt-get update && \
    apt-get install -y r-base

RUN apt-get install -y git wget libcurl4-openssl-dev default-jdk-headless libssl-dev

## xgboost: CMake 3.12 or higher is required
RUN apt-get install -y python-pip && pip install cmake --upgrade && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN R CMD javareconf

ENV MAKE="make -j$(nproc)"

RUN R -e 'install.packages(c("ROCR","data.table","curl","R6","devtools","RCurl","jsonlite"), repos = "https://cran.rstudio.com/")'


ARG CACHE_DATE

RUN R -e 'install.packages("h2o", repos = "http://h2o-release.s3.amazonaws.com/h2o/latest_stable_R")'

RUN git clone --recursive https://github.com/dmlc/xgboost && \
    cd xgboost && \
    mkdir build && cd build && cmake .. -DUSE_CUDA=ON -DR_LIB=ON && \
    make install -j

RUN apt-get install -y libboost-dev libboost-system-dev libboost-filesystem-dev ocl-icd-opencl-dev opencl-headers clinfo
RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd   ## otherwise lightgm segfaults at runtime (compiles fine without it)
RUN R -e 'devtools::install_github("Laurae2/lgbdl"); lgbdl::lgb.dl(use_gpu = TRUE)'


CMD ["R","--vanilla"]
