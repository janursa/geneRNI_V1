#!/bin/tcsh
#SBATCH --job-name=testser
#SBATCH --partition=pAll
#SBATCH --nodes=2
#SBATCH --ntasks=80
#SBATCH --time=02:00:00
#SBATCH --mail-user=jalil.nourisa@hereon.de
#SBATCH --mail-type=ALL
#SBATCH --output=job.o%j
#SBATCH --error=job.e%j
unset LD_PRELOAD
# source /etc/profile.d/modules.sh
module purge

module load applications/python/3.8
# setenv OMP_NUM_THREADS 20
python3 scripts/tune_dream4.py 80
