
install.packages(c("data.table","curl"))
install.packages("ggplot2")
install.packages("ROCR")


install.packages("xgboost")

## until lightgbm gets (back) to CRAN:
PKG_URL <- "https://github.com/microsoft/LightGBM/releases/download/v3.0.0/lightgbm-3.0.0-r-cran.tar.gz"
install.packages("remotes")
remotes::install_url(PKG_URL)
## more info: https://lightgbm.readthedocs.io/en/latest/R/index.html

install.packages(c("RCurl","h2o"))



