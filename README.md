#
cd experiments/probe_controlled_tsv/scripts/

# 
./01_generate_samples.sh     
./02_train_tsv.sh            
./03_train_probe.sh          
./04_run_experiments.sh      

# may not work
./run_all.sh
