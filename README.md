### This is the code for T-RNN, a recursive math word problem solver
cd first_stage  

run code in py2.7 environment with pytorch

Training in first stage:
```
sh script/exe_12.sh 
```
Test in first stage:
```
sh script/step_one_test_12.sh 
```


cd second_stage

run code in py3.5 environment with pytorch

Training and test integrated in second stage:
```
python  src/main.py
```

In next code version, we will update the code for runing beautifully
Be careful, due to data limitation in github, some data we do not upload. If you need, be free to email us.



