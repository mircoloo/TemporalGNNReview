#!/bin/bash

pwd
bash ./submit_job.sh nyse dgdnn zscore False
bash ./submit_job.sh nyse dgdnn none False
bash ./submit_job.sh nyse dgdnn log1p False
bash ./submit_job.sh nyse dgdnn minmax False