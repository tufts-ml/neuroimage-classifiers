# Kaiser Permanente Southern California (KPSC)

## Downloading data from KPSC
```
stfp <uuid>@sftp.kp-scalresearch.org
get <remote_path> <local_path>
```

## Logging in to KPSC
Log in to virtual machine. Then, ssh into KPSC high-performance computing (HPC) cluster.
```
ssh <uuid>@<ip_address>
```

## Installing enviroment
```
python -m venv brain-scan-classifiers-venv
source brain-scan-classifiers-venv/bin/activate
pip install -r requirements.txt
```

## Starting Jupyter Notebook on KPSC
```
source /home/<uuid>/brain-scan-classifiers-venv/bin/activate
jupyter notebook --ip=0.0.0.0 --no-browser
```
Open `cscdclppa30.kp-scalresearch.org` link.

## Running job on KPSC
Create a `script.sh` file that specifies the GPU devices, thread counts, and CPU affinity for your job.
```
#!/bin/bash

CUDA_VISIBLE_DEVICES=<gpu_ids> \
OMP_NUM_THREADS=<num_threads> \
MKL_NUM_THREADS=<num_threads> \
taskset --cpu-list <cpu_list> \
python main.py
```
Start the job in the background with `nohup`.
```
nohup script.sh > output.log &
disown
```
