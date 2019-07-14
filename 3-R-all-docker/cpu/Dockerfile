FROM rocker/tidyverse

RUN apt-get update && \
    apt-get install -y default-jdk-headless cmake && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN R CMD javareconf

ENV MAKE="make -j4"

RUN install2.r ROCR data.table curl


ARG CACHE_DATE

RUN install2.r -r http://h2o-release.s3.amazonaws.com/h2o/latest_stable_R h2o

RUN git clone --recursive https://github.com/dmlc/xgboost && \
    cd xgboost && git submodule init && git submodule update && \
    cd R-package && R CMD INSTALL .

RUN R -e 'devtools::install_github("Laurae2/lgbdl"); lgbdl::lgb.dl()'


CMD ["R","--vanilla"]
