#!/bin/bash -l

module load python
source activate /global/common/software/m1727/n2j

cat << EOF > prepare-env.sl
#!/bin/bash
#SBATCH -J data_gen
#SBATCH -C knl
#SBATCH --nodes=1
#SBATCH --time=36:00:00
#SBATCH --time-min=12:00:00
#SBATCH --qos=low

srun python generate_data.py 
EOF

sbatch prepare-env.sl