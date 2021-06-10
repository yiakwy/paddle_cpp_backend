PAddle CPP backend
==================

## Chapter 2: Training PAddle program in cpp backend

### step 1. build paddle lib

Make sure you have successfully compiled PAddle as I talked in my presentation in our Ubuntu 18.04 machine.

Here is the system configuration:

```
Paddle : Released 2.0
OS: Ubuntu 18.04
# cmake 3.16 compiled with ssl; you can also use gcc 8.4 but you have to CHANGE THE SOURCE CODE
gcc: 8.2

# RTX 2080Ti, PV100, Quad P4000..., NV Driver 400.100
CUDA: 10.2
CUDNN: 8.1 OR 9.0

# used for multi gpu cards communication
NCCL: v2.8

# conda 
conda installed python: 3.6
```

copy scripts from `paddle_cpp_backend/scripts` to `Paddle`:

```bash
cd $ROOT/scripts
source env.sh

# build gcc-8.2 in temp_gcc82
bash install_gcc_8.2.sh

# install nccl
bash install_nccl_2.8.sh

# build cmake project; use ccmake to see whether values are set correct
bash build.sh
```

generate output:

```bash
cd $PADDLE_BUILD
make -j8
cd paddle/dist
pip install *.whl

# we have handled output from paddle 
# but you are free to pack paddle output
# using following tools

# cd $PADDLE_SRC/paddle/scripts

# see `main` from paddle_build.sh for how to
# pack paddle output

# bash paddle_bash.sh gen_fluid_lib
```

### step 2. generate network intermediate description

```
conda activate py36
cd python/network
python least_square_network.py
```

### step 3. train the least square object in CPP backend:

```
cd paddle/network
../../build/bin/least_square_trainer
``` 

The output will be:
```
Op_desc : mul Op_desc : elementwise_add Op_desc : elementwise_sub Op_desc : square Op_desc : mean 
loss_name: mean_0.tmp_0
step: 0 loss: 486.927
step: 1 loss: 484.665
step: 2 loss: 482.413
step: 3 loss: 480.172
step: 4 loss: 477.942
step: 5 loss: 475.721
step: 6 loss: 473.512
step: 7 loss: 471.312
step: 8 loss: 469.123
step: 9 loss: 466.943

------------------------->     Profiling Report     <-------------------------

Place: CPU
Time unit: ms
Sorted by total time in descending order in the same thread

-------------------------     Overhead Summary      -------------------------

Total time: 1.19622
  Computation time       Total: 0.51573     Ratio: 43.1134%
  Framework overhead     Total: 0.680487    Ratio: 56.8866%

-------------------------     GpuMemCpy Summary     -------------------------

GpuMemcpy                Calls: 0           Total: 0           Ratio: 0%

-------------------------       Event Summary       -------------------------

Event                                Calls       Total       Min.        Max.        Ave.        Ratio.      
thread0::mul                         10          0.30546     0.00746     0.219246    0.030546    0.255355    
thread0::sgd                         20          0.198225    0.004167    0.053196    0.00991125  0.16571     
thread0::elementwise_add             10          0.095019    0.005441    0.028407    0.0095019   0.0794329   
thread0::mul_grad                    10          0.082498    0.006979    0.017608    0.0082498   0.0689657   
thread0::elementwise_sub_grad        10          0.080829    0.00529     0.022378    0.0080829   0.0675705   
thread0::elementwise_sub             10          0.079912    0.004637    0.029711    0.0079912   0.0668039   
thread0::elementwise_add_grad        10          0.068009    0.005656    0.012705    0.0068009   0.0568534   
thread0::fill_constant               10          0.067041    0.004544    0.01799     0.0067041   0.0560442   
thread0::square                      10          0.065969    0.004125    0.022021    0.0065969   0.055148    
thread0::mean_grad                   10          0.062919    0.004719    0.018939    0.0062919   0.0525983   
thread0::square_grad                 10          0.051058    0.004429    0.009672    0.0051058   0.0426829   
thread0::mean                        10          0.039278    0.003525    0.00637     0.0039278   0.0328352   
run_time = 2093

```

## Chapter 3: 
 Subscribe my wechat public account: `after midnight: bug-driven ML devleloping lab` for updates
