#!/bin/bash

pwd
bash ./submit_job.sh sse dgdnn zscore False
bash ./submit_job.sh sse dgdnn none False
bash ./submit_job.sh sse dgdnn log1p False
bash ./submit_job.sh sse dgdnn minmax False