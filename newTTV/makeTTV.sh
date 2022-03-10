ds_dir="/rds/user/$USER/hpc-work/personality-machine/firstimpressions/"
files=`find $ds_dir -name *.mp4`
TTV.py $files

