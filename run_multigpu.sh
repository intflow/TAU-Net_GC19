#!/bin/bash

testdir=/home/Gist/data_10TB/GC_2019/t3_audio
testdir_basename=t3_audio
NR_type=16k_TAUnet.H1.newmodel2.IBM.v2

input_dir=$testdir/$testdir_basename
output_dir=$testdir/${testdir_basename}_$NR_type
	
	 
python3 run_proc_stereo.py --dir=$input_dir --out_dir=$output_dir --db_split=1/4 --gpu_idx=0&
python3 run_proc_stereo.py --dir=$input_dir --out_dir=$output_dir --db_split=2/4 --gpu_idx=1&
python3 run_proc_stereo.py --dir=$input_dir --out_dir=$output_dir --db_split=3/4 --gpu_idx=2&
python3 run_proc_stereo.py --dir=$input_dir --out_dir=$output_dir --db_split=4/4 --gpu_idx=3&
