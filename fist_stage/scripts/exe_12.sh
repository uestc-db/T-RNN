exp_dir="./experiment/params_12"
echo $exp_dir
mkdir $exp_dir
python src/main.py --cuda-use --checkpoint-dir-name params_12 --mode 0 --teacher-forcing-ratio 0.83 --cuda-id 1 --input-dropout 0.4 --encoder-hidden-size 1024 --decoder-hidden-size 2048
