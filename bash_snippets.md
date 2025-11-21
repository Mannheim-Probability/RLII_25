# Helpful Code-snippets for bash scripts

## General Stuff

Always make sure to activate your venv!

Multiple commands can be executed with a ";" separating them.

## SLURM-commands

### Get manual for command
```
man command_name
```

### Queue a job
```
sbatch path/to/job.sh
```

### Check on all running and queued jobs
```
squeue
```

### Check planned starting time of all queued jobs
```
squeue --start
```

### Cancel a job
```
scancel
```

### Check for currently idle nodes
```
sinfo_t_idle
```

### Check Fairshare-values of account
Max: 10000 (Multiplied with own Fairshare-value to get job value)
```
sshare
```

### Get a report on resource usage
```
sreport --tres cpu,gres/gpu cluster AccountUtilizationByUser Start=2020-03-01 End=now -t hours
```

## SLURM helpful batch commands

### Submit all .sh jobs in a folder
```
for script in path/to/folder/*.sh; do
  sbatch "$script"
  done
```

### Cancel all Jobs with certain numbers
```
for i in $(seq number1 number2); do
  scancel $i
  done
```

### Get a random seed in the range of the Gym-seeds
```
seed=$(shuf -i 0-4294967295 -n 1)
echo "Random seed: $seed"
```

### Use Rsync to get the results folder without Zip-files
Uses absolute paths and copies every subfolder of the last folder in /source/directory/ into the last folder of /destination/directory/ while skipping folders and files that are already present in /destination/directory/ . It does not modify or delete files of /desination/directory/ that are not in /source/directory/

On local machine (use "." when already in goal-directory):
```
rsync -av --exclude='*.zip' -e ssh ma_user_name@uc3.scc.kit.edu:/source/directory/ /local/goal/directory/
```

On Cluster in order to then download the folder:
```
rsync -av --exclude='*.zip' /source/directory/ /local/goal/directory/
```

## Workspaces commands

Work in progress