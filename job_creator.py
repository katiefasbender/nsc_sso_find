#!/usr/bin/env python

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
from astropy.io import fits
import sys
import os
import time 

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------
def makedir(dir):
	if not os.path.exists(dir):
		os.mdkir(dir)


#-----------------------------------------------------------------------------
# Main Code
#-----------------------------------------------------------------------------

if __name__ == "__main__":

	#put slurm stuff in here 
	#format: python nsc_sso_find.py <#_of_healpix> <marker>
	#and then i need to separate the total healpix list into batches of maybe 10ish
	input_file=sys.argv[1] #your input file would be 'path_to_file/healpix_good.fits'
	if not os.path.exists(input_file): #to check whether there actually is a file.  change print to file output.
		print("You messed up!!!!!!!  %s doesn't exist!" %input_file)
		sys.exit(1)

	hdul=fits.open(input_file)
	data=hdul[1]
	max_data=len(hdul[1].data)

#this is in the "job_name".sh file:

##
##
## Lines starting with #SBATCH are read by Slurm. Lines starting with ## are comments.
## All other lines are read by the shell.
	
	maximum_range=2658
	
	m=0
	for my_set in range(0,maximum_range):
		job_file='hpix_job_%d.sh' % my_set
		with open(job_file,'w') as fh: #even if the file doesn't exist, it will be created and the lines will be written to it!!!!!! <3
			fh.writelines("#!/bin/bash\n")
			fh.writelines("#SBATCH --job-name=hpix_job_%d" % my_set)      # job name, pix is the #_of_healpix
			fh.writelines("#SBATCH --output=nft-%j.out\n") # standard output file (%j = jobid)
			fh.writelines("#SBATCH --error=nft-%j.err\n") # standard error file
			fh.writelines("#SBATCH --partition=unsafe\n")	   # queue partition to run the job in
			fh.writelines("#SBATCH --ntasks=1") #for running in parallel? no... I think I need srun?
			fh.writelines("#SBATCH --nodes=1\n")            # number of nodes to allocate
			fh.writelines("#SBATCH --ntasks-per-node 1\n")        # number of cores to allocate; set with care 
			fh.writelines("#SBATCH --mem=200\n")         # 2000 MB of Memory allocated; set --mem with care (200 for 1)
			fh.writelines("#SBATCH --time=00:30:00\n")     # Maximum job run time
			#fh.writelines("#SBATCH --mail-user=katiefasbender@montana.edu\n") # user to send emails to
			#fh.writelines("#SBATCH --mail-type=FAIL\n")   # Email on: BEGIN, END, FAIL & REQUEUE
			fh.writelines("module load Anaconda3/5.1.0\n") #load anaconda
			for i in range(0,50): #loop through elements of input file
				counter=i+(50*my_set)
				num_pix=hdul[1].data[counter][0]
				mark_pix=hdul[1].data[counter][1]
				subdir=int(int(num_pix)//1000)
				makedir("../../../../mnt/lustrefs/scratch/katie.fasbender/nsc_fast_track_healpix/hgroup_%d" % subdir)
				if counter<132885:
					fh.writelines("python nsc_sso_find.py %d %d\n" % (num_pix,mark_pix)) #this is the command that runs the python script to find SSOs for 1 healpix			
#fh.writelines("date\n")                            # print out the date
#fh.writelines("hostname -s\n")                     # print a message from the compute node
#fh.writelines("date\n")                         # print the date again
		m=m+1
		os.system("sbatch %s" %job_file)
		if m==500:
			time.sleep(900)
			m=0
