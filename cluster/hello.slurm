#/bin/bash
#SBATCH -job-name="Hello World"
#SBATCH -error=hellp.err 
#SBATCH -output=hello.out
#SBATCH -partition=allgroups
#SBATCH -ntasks=1
#SBATCH -mem=1G
#SBATCH -time=01:00:00
echo "HELLO WORLD"
