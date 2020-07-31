# FortNN
FortNN is an attempt to create a Fortran library for dealing with Neural Networks and Deep Learning. It has the following features :

  - A simple way to define layers and their activation functions, plus other parameters
  - Blasized
  - Uses OpenMP and along that, there is optional MPI usability, if needed
  - A command line interface, if needed

For now, the following technical features are supported:
  - activation functions: relu , lrelu , smax (softmax) , sigmoid , htan
  - loss functions:  mse (mean squared error) , sce (softmax cross entropy)
  - optimizers : sgd (stochastic gradient descent)
  - learning decay
  - droprates
  - momentum
  - epochs
  - online training or batach training using MPI
    
GPU support is not present, but it is planned. As FortNN is CPU-based for now, the performance will largely depend on the implementation of Blas routines you utilize (OpenBlas, Atlas, MKL-DNN, etc.).

## Prerequisites
You would obviously need a Fortran 2003 compiler such as `gfortran`. `OpenMP` and an a `blas` implementation must be present. If `MPI` is used, an `MPI` library such as `Open-MPI` must be present as well. To compile:
```sh
gfortran -Ofast -O3 -march=native -fopenmp -x f95-cpp-input yourprogram.f90 FortNN.f90 -o yourexe -lopenblas 
```
If you want to enable a feature like `MPI` or `CLI`, use `-D MPI` or `-D CLI`.
```sh
mpifort -Ofast -O3 -march=native -fopenmp -x f95-cpp-input -D MPI yourprogram.f90 FortNN.f90 -o yourexe -lopenblas && mpirun yourexe
```
FortNN relies on [M_CLI.f90](https://github.com/urbanjost/M_CLI/blob/master/src/M_CLI.f90) for deploying a command line interface. Download the file and include it in your project, if you want the command line interface(`-D CLI`).
## How to use

![The NN structure](/NN.png)

 Suppose that we want to have the MNIST deep learning model with the input layer consisting of 784 neuron, then 3 hidden layers of size 500 and the output layer containing 10 neurons (5 layers total). Define a neural network using:
```fortran
USE fortnn
TYPE (nn) :: mynn
TYPE (pointerproc1) , DIMENSION(:) , ALLOCATABLE :: afuncs    ! for the activation functions
```
 Define the activation function for each of the layers by
```fortran
ALLOCATE(afuncs(5))   
afuncs(1)%f => NULL() ! the inputs usually don't take a function
afuncs(2)%f => htan
afuncs(3)%f => lrelu
afuncs(4)%f => smax
afuncs(5)%f => sigmoid
```
Now we want to start with a learning rate of `0.05` and end with `0.001` within `30` epochs. Also, a momentum of `0.9` will be used. We will utilize `sce` for the loss function and `sgd` for the optimizer. Furthermore, we will ustilize `0.8` as the droprate for the hidden layers.
Initialize the network using:
```fortran
   CALL mynn%init( layers=[784,500,500,500,10] , droprates=[1.,0.8,0.8,0.8,1.] , activ_func=afuncs ,&
        loss_func=sce , optimizer=sgd , lrs=0.05 , lrf=0.001 , mu=0.9 , epoch=30 )
```
Train the network using 
```fortran
CALL mynn%train(traindata , targets , testnn)
```
If MPI is intended:
```fortran
CALL mynn%train_mpi(traindata , targets , testnn)
```
where `traindata` is a matrix in which each column represents one set of data and the corresponding column in `targets` indicates the results. Note that because Fortran is column-major, data are meant to be packed in columns rather than in rows. Also, 'testnn' is a subroutine that does the calculations of the performance and accuracy and the user should feed it into the `train` or `train_mpi` subroutines. The general interface is as follows:
```fortran
   SUBROUTINE testnn ()
   ! this subroutine can be "contained" in the main program
      INTEGER :: score
      INTEGER :: nl , r1(1) , r2(2) , nl , nr
      nl = SIZE(mynn%layers) 
      nr = UBOUND(testdata,2)
      score = 0
      DO i = 1 , nr
         ! query the testdata
         CALL mynn%query(testdata(:,i))
         ! get the output result at the final layer i.e., mynn%layers(nl-1)%y
         ! layer index starts from 0 to nl-1
         r1 = MAXLOC(mynn%layers(nl-1)%y,dim=1)
         ! testresults has the test targets
         r2 = MAXLOC(testresults(:,i),dim=1)

         IF ( r1(1) == r2(1) ) score = score + 1
      ENDDO
      PRINT*, "The performance: " , REAL(score)/nr
   END SUBROUTINE testnn
```
where `testdata` and `testresults` are the matrices for performance evaulation of the network.
## Contributions
If you would like, you can expand FortNN by adding other capabilities such as more functions. Of course, bug submits and suggestions are welcomed.
