#!/bin/bash

pwd
bash ./submit_job.sh nasdaq dgdnn zscore False
bash ./submit_job.sh nasdaq dgdnn none False
bash ./submit_job.sh nasdaq dgdnn log1p False
bash ./submit_job.sh nasdaq dgdnn minmax False