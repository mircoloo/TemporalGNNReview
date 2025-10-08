#!/bin/bash

pwd
bash ./submit_job.sh nasdaq dgdnn zscore False
#bash ./submit_job.sh sse dgdnn zscore False
bash ./submit_job.sh nyse dgdnn zscore False

bash ./submit_job.sh nasdaq dgdnn minmax False
bash ./submit_job.sh sse dgdnn minmax False
bash ./submit_job.sh nyse dgdnn minmax False

bash ./submit_job.sh nasdaq dgdnn log1p False
bash ./submit_job.sh sse dgdnn log1p False
bash ./submit_job.sh nyse dgdnn log1p False



#bash ./submit_job.sh nasdaq dgdnn zscore False
#bash ./submit_job.sh nyse dgdnn zscore False
#bash ./submit_job.sh sse dgdnn zscore False

#bash ./submit_job.sh nasdaq darnn zscore False
#bash ./submit_job.sh nyse darnn  zscore False
#bash ./submit_job.sh sse darnn zscore False

#bash ./submit_job.sh nasdaq dgdnn zscore
#bash ./submit_job.sh nyse dgdnn zscore
#bash ./submit_job.sh sse dgdnn zscore

#bash ./submit_job.sh nasdaq graphwavenet zscore False
#bash ./submit_job.sh nyse graphwavenet  zscore False
#bash ./submit_job.sh sse graphwavenet zscore False

#bash ./submit_job.sh nasdaq hyperstockgat zscore False
#bash ./submit_job.sh nyse hyperstockgat  zscore False
#bash ./submit_job.sh sse hyperstockgat zscore False
