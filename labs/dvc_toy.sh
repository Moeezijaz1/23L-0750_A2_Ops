#!/usr/bin/env bash
# Task 1.4 toy part - run in a NEW EMPTY folder. Replace <...> placeholders.
set -ex
mkdir dvc-toy && cd dvc-toy
git init && dvc init
printf "id,value\n1,10\n2,20\n3,30\n" > data.csv
dvc add data.csv
git add data.csv.dvc .gitignore .dvc && git commit -m "v1: initial CSV tracked by DVC"
printf "4,40\n5,50\n" >> data.csv
dvc add data.csv && git add data.csv.dvc && git commit -m "v2: added rows"
git log --oneline
git checkout HEAD~1 -- data.csv.dvc && dvc checkout      # back to v1
cat data.csv                                              # 3 rows again
git checkout HEAD -- data.csv.dvc && dvc checkout         # forward to v2
# remote storage (DagsHub storage, S3-compatible):
dvc remote add -d origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/<user>/<repo>.s3
dvc remote modify --local origin access_key_id <DAGSHUB_TOKEN>
dvc remote modify --local origin secret_access_key <DAGSHUB_TOKEN>
dvc push
rm -rf data.csv .dvc/cache                                # delete local data + cache
dvc pull && cat data.csv                                  # restored from remote
