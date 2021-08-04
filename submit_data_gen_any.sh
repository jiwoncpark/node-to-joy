#!/bin/bash -l

module load python
source activate /global/common/software/m1727/n2j
echo "$1"

cat << EOF > prepare-env-${1}.sl
#!/bin/bash
#SBATCH -J data_gen_${1}
#SBATCH -C knl
#SBATCH --nodes=1
#SBATCH --time=07:00:00
#SBATCH --qos=regular

srun python generate_data_any.py ${1}
EOF

sbatch prepare-env-${1}.sl