MODULE timermodule
   IMPLICIT NONE
   INTEGER , PARAMETER :: dp1 = SELECTED_REAL_KIND(13)
   TYPE , PUBLIC :: timer
      PRIVATE
      REAL(KIND=dp1) :: saved_time
      CHARACTER (len=100) :: mes=""
   CONTAINS
      PROCEDURE , PUBLIC :: tic => gettime
      PROCEDURE , PUBLIC :: toc => givetime
   END TYPE timer
   PRIVATE :: gettime , givetime
CONTAINS
   SUBROUTINE gettime(start , mes)
      CLASS(timer) , INTENT(OUT) :: start
      CHARACTER(len=*) , INTENT(in) , OPTIONAL :: mes
      INTEGER, DIMENSION(8) :: time1
      CALL DATE_AND_TIME(values=time1)
      start%saved_time = 86400._dp1*time1(3) + 3600._dp1*time1(5) +&
           60._dp1*time1(6) + time1(7) + time1(8)*1.e-3_dp1
      IF (PRESENT(mes)) THEN
         start%mes = TRIM(mes)
      ELSE
         start%mes = "this part"
      ENDIF
   END SUBROUTINE gettime
   !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   SUBROUTINE givetime(last_time)
      CLASS(timer) , INTENT(IN) :: last_time
      INTEGER , DIMENSION(8) :: time1
      REAL(KIND=dp1) :: stoptime, givetime1
      INTEGER :: mnt
      REAL(kind=dp1) :: scnd
      
      CALL DATE_AND_TIME(values = time1)
      stoptime = 86400._dp1*time1(3) + 3600._dp1*time1(5) + 60._dp1*time1(6) +&
           time1(7) + time1(8)*1.e-3_dp1
      givetime1 = stoptime - last_time%saved_time
      scnd = MOD(givetime1,60._dp1)
      mnt = INT(givetime1)/60
      IF (mnt < 1) THEN
         WRITE (*,100) TRIM(last_time%mes) , givetime1
      ELSEIF ( mnt == 1 ) THEN
         WRITE (*,200) TRIM(last_time%mes) , scnd
      ELSE
         WRITE (*,300) TRIM(last_time%mes) , mnt , scnd
      ENDIF
100   FORMAT ('  The time spent on ' , A , ' was ' , f6.3 , ' seconds.' )
200   FORMAT ('  The time spent on ' , A , ' was 1 minute and ' , &
           f5.2 , ' seconds.')
300   FORMAT ('  The time spent on ' , A , ' was ' , I2 , &
           ' minutes and ' , f5.2 , ' seconds.')
   END SUBROUTINE givetime
END MODULE timermodule
!----------------------------------------------------------------------------------
#ifdef CLI
   INCLUDE "M_CLI.f90"
#endif
!----------------------------------------------------------------------------------
MODULE FortNN
   USE timermodule
   USE OMP_LIB
   USE, INTRINSIC :: iso_c_binding
   IMPLICIT NONE
   INTEGER, PARAMETER :: dp=selected_real_KIND(13)

   ! activation function syntax
   TYPE :: pointerproc1 
      PROCEDURE(ndarray1) , POINTER, NOPASS  :: f
   END TYPE pointerproc1

   ! an array of function pointers for activation functions
   TYPE (pointerproc1) , DIMENSION(:) , ALLOCATABLE :: activfuncs
   
   TYPE :: layer
      ! each layer includes: its weight matrix w, bias neurons b, its output y,
      ! loss error el (used for the last layer), and the propagation error e.
      ! we have not defined an input element here, as the results of a layer
      ! would be the input for the next one. we would start the layers from
      ! index 0 in which y would correspond to the inputs of the network.
      ! wd is for storing dE/dW and bd is for dE/dB.
      REAL(kind=dp) , DIMENSION (:,:) , POINTER :: w , wd
      REAL(kind=dp) , DIMENSION (:) , POINTER :: b , bd , y , e , el
      ! we have flags for the droprate
      LOGICAL :: is_dr
      REAL(kind=dp) :: dr
      ! the mask
      INTEGER , DIMENSION(:) , POINTER :: drm
      ! each layer has its own activation function and its derivative. we define them
      ! as pointers so we can have a modular layer structure.
      PROCEDURE(ndarray1) , PUBLIC , NOPASS , POINTER :: af => NULL() , afd => NULL()
   END TYPE layer

   ! we declare the layer component as target so we can point to them efficiently.
   TARGET :: layer

   ! making pointers for working with input, output and error vectors.
   REAL(kind=dp) , DIMENSION (:) , POINTER ::  xp => NULL() , yp => NULL() , ep => NULL()
   
   ! creating the neural network type
   TYPE :: nn
      ! learning rate, starting learning rate and the final learning rate (learning decay)
      REAL(kind=dp) :: lr , lrs , lrf
      ! the momentum
      REAL(kind=dp) :: mu
      ! the epochs
      INTEGER :: epoch
      ! the array of layers
      TYPE(layer) , DIMENSION (:) , ALLOCATABLE :: layers
      ! average of loss errors
      REAL(kind=dp) :: err
      ! we define the loss functions and their derivatives along with the optimizer
      ! as pointers.
      ! in the init procedure, we would assign these according to what the user requested.
      ! for the whole network, one loss function and one optimizer are assigned.
      PROCEDURE(ndarray2) , PUBLIC , NOPASS , POINTER :: lf => NULL() , lfd => NULL()
      PROCEDURE(ndarray0) , PUBLIC , PASS , POINTER :: optimizer => NULL()
   CONTAINS 
      ! init the NN which means set the layers and assign its functions
      PROCEDURE , PUBLIC :: init
      ! query and train functions. train1 is for one input and one target.
#ifndef MPI
      PROCEDURE , PUBLIC :: query , train1 , train
#else
      PROCEDURE , PUBLIC :: query , train1 , train_mpi
#endif
   END TYPE nn

   TYPE :: dyn_arr
      REAL(kind=dp) , DIMENSION(:) , ALLOCATABLE :: v
      REAL(kind=dp) , DIMENSION(:,:) , ALLOCATABLE :: m
   END TYPE dyn_arr
      
   ABSTRACT INTERFACE
      ! the interfaces for loss and activation functions
      FUNCTION ndarray1 (n)
         IMPORT dp
         IMPORT pointerproc1
         REAL(kind=dp) , DIMENSION (:) , INTENT(in) :: n
         REAL(kind=dp) , DIMENSION (SIZE(n)) :: ndarray1
      END FUNCTION ndarray1
      
      FUNCTION ndarray2 ( n1 , n2 )
         IMPORT dp
         REAL(kind=dp) , DIMENSION (:) , INTENT(in) :: n1 , n2
         REAL(kind=dp) , DIMENSION (SIZE(n1)) :: ndarray2
      END FUNCTION ndarray2

      SUBROUTINE ndarray0 (this)
         IMPORT nn
         CLASS (nn), INTENT(inout) :: this
      END SUBROUTINE ndarray0
   END INTERFACE

   PRIVATE :: drop_mask , normaldist , init , query , train1

CONTAINS
   !==================================================================
   SUBROUTINE init (this , layers , droprates , activ_func , loss_func , &
        optimizer , lrs , lrf , mu , epoch)
      CLASS (nn) , INTENT(inout) :: this

      ! an array containing the structure of layers like [100,20,30,10]
      ! which means the input has 100 nodes; the first hidden layer has 20 nodes;
      ! the second hidden layer has 30 nodes, and the result layer has 10 ones.
      ! in this model we multiply weights by inputs not inputs by weights and  
      ! we assume the inputs and outputs will always be vectors like dim X 1 
      ! because Fortran is column-major.
      INTEGER , DIMENSION (:) , INTENT(in) :: layers

      ! an array of values indicating the droprates for all the layers
      REAL , DIMENSION (:) , INTENT(in) , OPTIONAL :: droprates

      REAL , INTENT(in) , OPTIONAL :: lrs , lrf , mu
      INTEGER , INTENT (in) , OPTIONAL :: epoch 
      INTEGER :: i , nin , nout , nl

      ! list of active functions passed in
      TYPE (pointerproc1) , DIMENSION(0:) , INTENT(in) :: activ_func
      PROCEDURE (ndarray2) :: loss_func ! the loss function assigned
      PROCEDURE (ndarray0) :: optimizer ! the optimizer procedure
      REAL(kind=dp) :: sd
      TYPE (dyn_arr) :: wb
      
      ! checks on the size of droprate and their values
      IF (PRESENT(droprates)) THEN
         IF ( SIZE(layers) .NE. SIZE(droprates) ) THEN
            PRINT*, "The size of the droprates and the layers do not match."
            STOP
         ENDIF
         IF ( MINVAL(droprates) < 0._dp .OR. MAXVAL(droprates) > 1._dp ) THEN
            PRINT*, "Droprates must be positive, real values less than 1."
         ENDIF
      ENDIF

      ! for every layer, an activation function must be present. for the first
      ! layer i.e., zero, the activation function is defined as null, but can be
      ! set to anything if desired.
      IF (SIZE(activ_func) /= SIZE(layers)) THEN
         PRINT*, "The number of layers and activation functions do not match."
         STOP
      ENDIF

      ! provide consistent random numbers. comment out for production use.
      CALL seeder

      ! setting the learning rate.
      IF (PRESENT(lrf) .AND. PRESENT(lrs)) THEN
         IF (lrf>lrs) THEN
            PRINT*,"The final learning rate should be smaller than the start rate."
            STOP
         END IF
      ENDIF

      IF (PRESENT(lrs)) THEN
         this%lrs=REAL(lrs,dp)
      ELSE
         ! default learning rate is 0.1
         this%lrs=0.1_dp
      ENDIF
      this%lr = this%lrs

      IF (PRESENT(lrf) .AND. this%lr > lrf) THEN
         this%lrf=REAL(lrf,dp)
      ELSE
         this%lrf=this%lrs
      ENDIF

      ! setting the momentum
      IF (PRESENT(mu)) THEN
         this%mu=REAL(mu,dp)
      ELSE
         ! default momentum is zero
         this%mu=0._dp
      ENDIF

      ! epoch
      IF (PRESENT(epoch)) THEN
         this%epoch = epoch
      ELSE
         ! by default we want one round of training
         this%epoch = 1
      ENDIF
      
      nl = SIZE(layers)
      nin = layers(1)
      nout = layers(nl)
      
      ! for convenience (subroutine trainer needs the vector of inputs),
      ! we will create 0:nl-1 layers. The layer zero will only hold inputs as
      ! its result with no error or weight matrix.
      ALLOCATE (this%layers(0:nl-1))

      DO i=1,nl-1
         
         ! each weight is constructed by (the next number X the former number) in the list
         ALLOCATE (this%layers(i)%w(layers(i+1),layers(i)))
         ALLOCATE (this%layers(i)%wd(layers(i+1),layers(i)))

         ! allocating the biases
         ALLOCATE (this%layers(i)%b(layers(i+1)))
         ALLOCATE (this%layers(i)%bd(layers(i+1)))

         ! we set the initial wd and bd to zero for use with the momentum.
         this%layers(i)%wd = 0._dp
         this%layers(i)%bd = 0._dp

         ! allocate y & e for the first hidden layer up to the final layer
         ALLOCATE (this%layers(i)%y(layers(i+1)))
         ALLOCATE (this%layers(i)%e(layers(i+1)))

         ! allocating space for the loss error.
         ALLOCATE (this%layers(i)%el(layers(i+1)))

         ! now we init the weights using a normal distribution.
         ! based on Glorot method which is to force SD=sqrt(2/nin+nout) for each layer.
         ! the standard deviation for creating the initial weights:
         sd = SQRT(2._dp/REAL(layers(i+1)+layers(i),dp))
         CALL normaldist( 0._dp , sd , SHAPE(this%layers(i)%w) , wb )
         CALL normaldist( 0._dp , sd , SHAPE(this%layers(i)%b) , wb )

         this%layers(i)%w = wb%m
         this%layers(i)%b = wb%v
         DEALLOCATE (wb%v,wb%m)

         ! based on the loss function and activation functions passed in,
         ! we assign their derivatives to the appropriate names for each layer.
         ! activ_func is supposed to have the same dimension bounds of this%layers
         ! (i.e., starting from 0)
         this%layers(i)%af => activ_func(i)%f
         
         ! ASSOCIATED does not work here (segfault).
         IF (same_proc(c_FUNLOC(activ_func(i)%f),c_FUNLOC(sigmoid))) THEN
            this%layers(i)%afd =>  sigmoid_d
         ELSEIF (same_proc(c_FUNLOC(activ_func(i)%f),c_FUNLOC(smax))) THEN
            this%layers(i)%afd => smax_d
         ELSEIF (same_proc(c_FUNLOC(activ_func(i)%f),c_FUNLOC(htan))) THEN
            this%layers(i)%afd => tanh_d
         ELSEIF (same_proc(c_FUNLOC(activ_func(i)%f),c_FUNLOC(relu))) THEN
            this%layers(i)%afd => relu_d
         ELSEIF (same_proc(c_FUNLOC(activ_func(i)%f),c_FUNLOC(lrelu))) THEN
            this%layers(i)%afd => lrelu_d
         ELSE
            PRINT*,"The activation function you have requested is invalid."
            STOP
         ENDIF

      ENDDO

      ! manually creating the layer zero
      ALLOCATE (this%layers(0)%y(nin))

      ! no error matrix in layer zero
      ALLOCATE (this%layers(0)%e(0))
      ALLOCATE (this%layers(0)%el(0))

      ! no weight matrix in layer zero
      ALLOCATE (this%layers(0)%w(0,0),this%layers(0)%wd(0,0)) 
      ALLOCATE (this%layers(0)%b(0),this%layers(0)%bd(0))

      ! setting the droprates in the layers
      IF (PRESENT(droprates)) THEN
         DO i=0,nl-1
            ! assign only when the drop rates are < 1.
            IF (  droprates (i+1) - 0.9999999 < 0. ) THEN
               this%layers(i)%dr = droprates (i+1)
               this%layers(i)%is_dr = .TRUE.

               ! we will set the inference mode later in the query function
               ! based on the mode of run
               ALLOCATE (this%layers(i)%drm(layers(i+1)))
               this%layers(i)%drm = drop_mask ( this%layers(i)%dr , &
                    SIZE(this%layers(i)%drm))
            ELSE
               this%layers(i)%dr = 1.
               this%layers(i)%is_dr = .FALSE.
               ALLOCATE (this%layers(i)%drm(0))
            ENDIF
         ENDDO
      ELSE
         DO i=0,nl-1
            this%layers(i)%is_dr = .FALSE.
         ENDDO
      ENDIF

      ! assigning the loss function and the optimizer
      this%lf => loss_func
      this%optimizer => optimizer

      ! assigning the derivative of loss function
      IF (same_proc(c_FUNLOC(loss_func),c_FUNLOC(mse))) THEN
         this%lfd => mse_d
      ELSEIF (same_proc(c_FUNLOC(loss_func),c_FUNLOC(sce))) THEN
         this%lfd => sce_d
      ELSE
         PRINT*,"The loss function you have requested is invalid."
         STOP
      ENDIF

      ! activation function for layer zero (very likely NULL)
      this%layers(0)%af => activ_func(0)%f

   CONTAINS
      FUNCTION same_proc (a,b)
         ! a logical function to check if two function pointers point
         ! to the same entity
         LOGICAL :: same_proc
         TYPE(c_funptr), INTENT(in) :: a,b
         same_proc = TRANSFER(a,0_C_INTPTR_T) == TRANSFER(b,0_C_INTPTR_T)
      END FUNCTION same_proc
   END SUBROUTINE init
   !==================================================================
   SUBROUTINE query ( this , x , im )
      ! given an input x, produces the intermediate and the final results in this%y
      CLASS(nn) , INTENT(inout) :: this
      REAL(kind=dp) , DIMENSION(:) , INTENT(in) :: x
      LOGICAL , INTENT(in) , OPTIONAL :: im
      LOGICAL :: infer_mode
      INTEGER :: i , n , lm , ln
      ! the number of weight matrices. because we included layer zero
      ! without a weight matrix, the number of weight matrices will be
      ! one unit less than the length.
      n=SIZE(this%layers)-1

      ! setting the infer mode
      IF (PRESENT(im)) THEN
         infer_mode = im
      ELSE
         ! by default True
         infer_mode = .TRUE.
      ENDIF
      
      ! copy x into y(0)
      this%layers(0)%y = x

      ! if an activation function for the inputs was intended (most of the times no)
      IF (ASSOCIATED(this%layers(0)%af)) &
           this%layers(0)%y = this%layers(0)%af (this%layers(0)%y)
      
      ! dropouts
      IF (this%layers(0)%is_dr) THEN
         ! if the query is requested during training which means im is False,
         ! we update the mask for each run of query here
         IF (infer_mode) THEN
            this%layers(0)%y = this%layers(0)%y * this%layers(0)%dr
         ELSE
            ! we are in training mode.
            ! each round of training, the mask is updated
            this%layers(0)%drm = drop_mask ( this%layers(0)%dr , &
                 SIZE(this%layers(0)%drm) )
            this%layers(0)%y = this%layers(0)%y * this%layers(0)%drm
         ENDIF
      END IF
      
      DO i=1,n
         xp => this%layers(i-1)%y
         yp => this%layers(i)%y

         lm = UBOUND(this%layers(i)%w,1)
         ln = UBOUND(this%layers(i)%w,2)

         !yp = this%af((this%layers(i)%w .dot. xp) + this%layers(i)%b)
         yp = this%layers(i)%b
         CALL dgemv( "N" , lm , ln , 1._dp , this%layers(i)%w , lm , xp , 1 , 1._dp , yp , 1 )
         yp = this%layers(i)%af(yp)
         
         ! dropouts
         IF (this%layers(i)%is_dr) THEN
            IF (infer_mode) THEN
               this%layers(i)%y = this%layers(i)%y * this%layers(i)%dr
            ELSE
               ! we update the mask in case the query is requested during training
               this%layers(i)%drm = drop_mask ( this%layers(i)%dr , &
                    SIZE(this%layers(i)%drm) )
               this%layers(i)%y = this%layers(i)%y * this%layers(i)%drm
            ENDIF
         END IF
         
      ENDDO
   END SUBROUTINE query
  !==================================================================
   SUBROUTINE train1 ( this , x , t )
      ! train1 is for training one vector of inputs and targets
      CLASS(nn) , INTENT(inout) ::  this
      REAL(kind=dp) , DIMENSION(:) , INTENT(in) :: x , t
      INTEGER :: n , i , lm , ln
      REAL(kind=dp) :: se
      
      n = SIZE(this%layers) - 1

      ! we query the input to create the intermediate results and the final output.
      ! we set the inference mode to false while training in case we have dropouts.
      CALL this%query( x , .FALSE. )
      
      ! create the propagation error in the final layer.
      ep => this%layers(n)%e
      ep = this%lfd ( this%layers(n)%y , t )
      ! using el as a temporary medium
      this%layers(n)%el = this%layers(n)%afd ( this%layers(n)%y )
      ep = ep * this%layers(n)%el

      ! the loss error
      this%layers(n)%el = this%lf ( this%layers(n)%y , t )
      ! saving maximum and minimum errors

      ! storing the average of loss errors
      se = SUM(this%layers(n)%el)/SIZE(this%layers(n)%el)
      this%err = se
      
      IF (this%layers(n)%is_dr) THEN
         ! enforcing the dropouts on the errors
         ep = ep * this%layers(n)%drm
      ENDIF
      
      ! now we create the propagation error for the other layers.
      ! when we want to create the error in the i-th layer, we multiply
      ! the transpose of the weight in the next layer by the error in the
      ! next layer again.
      DO i = n-1 , 1 , -1

         lm = UBOUND(this%layers(i+1)%w,1)
         ln = UBOUND(this%layers(i+1)%w,2)
         !this%layers(i)%e = TRANSPOSE(this%layers(i+1)%w) .dot. ep
         CALL dgemv( "T" , lm , ln , 1._dp , this%layers(i+1)%w , lm , &
              ep , 1 , 0._dp , this%layers(i)%e , 1 )
         this%layers(i)%el = this%layers(i)%afd (this%layers(i)%y) ! using el as a temp
         this%layers(i)%e =  this%layers(i)%el * this%layers(i)%e
         ep => this%layers(i)%e

         IF (this%layers(i)%is_dr) THEN
            ! enforcing the dropouts on the errors
            ep = ep * this%layers(i)%drm
         ENDIF
         
      ENDDO
      CALL this%optimizer()
   END SUBROUTINE train1
  !==================================================================
#ifndef MPI
   SUBROUTINE train ( this , d , t , test_results)
      ! does the training. subroutine test_results is the one that you set
      ! for measuring the performance, accuracy, etc.
      TYPE (timer) :: tw
      CLASS(nn) , INTENT(inout) :: this
      REAL(kind=dp) , DIMENSION(:,:) , INTENT(in) :: d , t
      INTEGER :: nl , nt , i , j
      PROCEDURE () :: test_results
      INTERFACE
         SUBROUTINE test_results ()
         END SUBROUTINE test_results
      END INTERFACE
      
      nl = SIZE (this%layers)
      IF (.NOT. ( ASSOCIATED (this%layers(0)%y) .AND. &
           ASSOCIATED (this%layers(nl-1)%y) ) ) THEN
         PRINT*,"You have not initialized the network. Call the init procedure first."
         STOP
      ENDIF

      IF (UBOUND(d,1) .NE. UBOUND(this%layers(1)%w,2)) THEN
         PRINT*, "The number of rows in the training data must be the same as &
              &the number of inputs."
         STOP
      ENDIF

      IF (UBOUND(d,2) .NE. UBOUND(t,2)) THEN
         PRINT*, "The number of training data (columns) does not equal that of the targets."
         STOP
      ENDIF

      nt = UBOUND (d,2)
      DO j = 1 , this%epoch

         CALL tw%tic ("this epoch")
         DO i = 1 , nt
            CALL this%train1 (d(:,i),t(:,i))
         ENDDO
         WRITE(*,200) j,this%lr,this%err
         CALL tw%toc
         
         ! test the training
         CALL test_results

         ! decay
         IF ( this%epoch .NE. 1 ) &
              this%lr = this%lr * ( this%lrf / this%lrs )**(1._dp / (this%epoch-1) )
      ENDDO
200   FORMAT ("epoch: " , I3 , ", learning rate:" , f6.3 , ", average error: " , g10.5)
   END SUBROUTINE train
  !==================================================================
#else
   SUBROUTINE train_mpi ( this , d , t , test_results)
      ! does the training utilizing MPI
      USE mpi
      TYPE (timer) :: tw
      CLASS(nn) , INTENT(inout) :: this
      REAL(kind=dp) , DIMENSION(:,:) , INTENT(in) :: d , t
      INTEGER :: nl , nt , i , j
      PROCEDURE () :: test_results
      ! mpi variables
      INTEGER :: merr , proc , nproc , np , root , ndims , wcount
      INTEGER , DIMENSION(2) :: array_of_sizes, array_of_subsizes , array_of_starts
      INTEGER , DIMENSION (:), ALLOCATABLE :: newtype
      TYPE(layer) , DIMENSION (:) , ALLOCATABLE :: new_layer

      INTERFACE
         SUBROUTINE test_results ()
         END SUBROUTINE test_results
      END INTERFACE

      CALL MPI_INIT (merr)
      CALL mpi_comm_size (mpi_comm_world , nproc , merr)
      CALL mpi_comm_rank (mpi_comm_world , proc , merr)
      ! force openblas to use one thread if MPI is enabled
      CALL openblas_set_num_threads(1)
      
      nl = SIZE (this%layers)
      IF (.NOT. ( ASSOCIATED (this%layers(0)%y) .AND. &
           ASSOCIATED (this%layers(nl-1)%y) ) ) THEN
         IF ( proc == 0 ) &
              PRINT*,"You have not initialized the network. Call the init procedure first."
         STOP
      ENDIF

      IF (UBOUND(d,1) .NE. UBOUND(this%layers(1)%w,2)) THEN
         IF ( proc == 0 ) &
              PRINT*, "The number of rows in the training data must be the same as &
              &the number of inputs."
         STOP
      ENDIF

      IF (UBOUND(d,2) .NE. UBOUND(t,2)) THEN
         IF ( proc == 0 ) &
              PRINT*, "The number of training data (columns) does not equal&
              & that of the targets."
         STOP
      ENDIF
      
      ! creating a new data type in 'newtype' for treating 2d arrays as a single item.
      ndims = 2
      ALLOCATE (newtype ( nl - 1 ))
      DO np = 1 , nl - 1
         array_of_sizes = [ SIZE(this%layers(np)%y) , SIZE(this%layers(np-1)%y) ]
         array_of_starts = [ 0 , 0 ]
         array_of_subsizes = array_of_sizes
         CALL mpi_type_create_subarray ( ndims , array_of_sizes , array_of_subsizes , &
              array_of_starts , mpi_order_fortran , &
              mpi_real8 , newtype(np) , merr )
         CALL mpi_type_commit ( newtype(np) , merr )

         root = 0
         ! broadcast the weights W in the root process to the others.
         CALL mpi_bcast ( this%layers(np)%w , 1 , newtype(np) , root , &
              mpi_comm_world , merr )
      ENDDO

      DO np = 1 , nl - 1

         root = 0
         ! broadcast the weights B in the root process to the others.
         ! no need to create a new type here.
         CALL mpi_bcast ( this%layers(np)%b , SIZE(this%layers(np)%b) , mpi_real8 , &
              root , mpi_comm_world , merr )
      ENDDO

      ! allocate new_layer for storing the intermediate results of dE/dW & dE/dB
      ALLOCATE ( new_layer(SIZE(this%layers)) )
      DO np = 1 , nl - 1
         ALLOCATE ( new_layer(np)%w , mold = this%layers(np)%w )
         ALLOCATE ( new_layer(np)%b , mold = this%layers(np)%b )
      ENDDO

      ! make all the threads wait for each other to arrive at this point.
      CALL mpi_barrier ( mpi_comm_world , merr )

      nt = UBOUND (d,2)
      DO j = 1 , this%epoch
         CALL tw%tic("this epoch")
         DO i = proc + 1 , nt , nproc
            CALL this%train1 (d(:,i),t(:,i))

            DO np = 1 , nl - 1
               
               new_layer(np)%w = 0._dp
               new_layer(np)%b = 0._dp
               
               wcount = SIZE(this%layers(np)%y) * SIZE(this%layers(np-1)%y)
               ! gather wd from the threads and put in new_layer

               CALL mpi_allreduce ( this%layers(np)%wd , new_layer(np)%w , wcount , &
                    mpi_real8 , mpi_sum , mpi_comm_world , merr )

               CALL mpi_allreduce ( this%layers(np)%bd , new_layer(np)%b , &
                    SIZE(this%layers(np)%b)  , &
                    mpi_real8 , mpi_sum , mpi_comm_world , merr )

               ! when a group is trained, all of the dE/dW and dE/dB add up.
               this%layers(np)%w =  this%layers(np)%w + new_layer(np)%w
               this%layers(np)%b =  this%layers(np)%b + new_layer(np)%b
               ! no need for a barrier as mpi_allreduce and all collective calls
               ! are of a blocking nature.
               
            ENDDO
         ENDDO

         WRITE(*,201) proc,j,this%lr,this%err

         IF (proc==0) CALL tw%toc
         ! test the training (for one process it is enough)
         IF (proc==0) CALL test_results
         
         ! decay
         IF ( this%epoch .NE. 1 ) &
              this%lr = this%lr * ( this%lrf / this%lrs )**(1._dp / (this%epoch-1) )

      ENDDO
201   FORMAT ("For process: " , I3 , ", epoch: " , I3 , ", learning rate:" , f6.3 , &
           ", average error: " , g10.5)

      CALL mpi_finalize(merr)
      
   END SUBROUTINE train_mpi
#endif
  !==================================================================
   SUBROUTINE sgd ( this )
      ! stochastic gradient descent optimizer. we will calculate dE/dW
      ! and dE/dB here and the new weights would be obtained.
      ! this routine is heavily 'blas'izes.
      ! dcopy and dscal hurt the performance but using daxpy improves it. also,
      ! simple element-wise whole vector multiplication is faster than using dsbmv.
      CLASS(nn) , INTENT(inout) :: this
      INTEGER :: nl , i , j
      INTEGER :: mm , nn , kk , lda , ldb , ldc
      REAL(kind=dp) :: alpha , beta
      
      nl = SIZE(this%layers) - 1 

      DO i = nl , 1 , -1

         mm = SIZE(this%layers(i)%y)
         kk = 1
         nn = SIZE(this%layers(i-1)%y)
         alpha = -this%lr ! minus lr
         beta = -this%mu
         lda = mm
         ldb = nn
         ldc = mm

         ! for transpose to work in dgemm we only need to change ld(a,b,c).
         ! no change in m,n,k is needed.
         CALL dgemm( "N" , "T" , mm , nn , kk , alpha , this%layers(i)%e , &
              lda , this%layers(i-1)%y , ldb , beta , this%layers(i)%wd , ldc )

         ! for the bias weights d/dB we have ( d(Wx+B) / dB = 1)
         !this%layers(i)%bd = alpha * this%layers(i)%e + beta * this%layers(i)%bd
         ! beta * bd -> bd 
         this%layers(i)%bd = beta * this%layers(i)%bd

         ! alpha * e + bd -> bd
         CALL daxpy ( mm , alpha , this%layers(i)%e , 1 , this%layers(i)%bd , 1 )

#ifndef MPI         
         ! we update the weights now. if MPI is enabled, the weight will be updated
         ! in train_mpi subroutine.
         !this%layers(i)%w = this%layers(i)%w + this%layers(i)%wd

         !$omp PARALLEL DO
         DO j = 1 , nn
            CALL daxpy ( mm , 1._dp , this%layers(i)%wd(:,j) , 1 , &
                 this%layers(i)%w(:,j) , 1 )
         ENDDO
         !$omp END PARALLEL DO

         !this%layers(i)%b = this%layers(i)%b + this%layers(i)%bd
         CALL daxpy ( mm , 1._dp , this%layers(i)%bd , 1 , this%layers(i)%b , 1 )
#endif
      ENDDO
   END SUBROUTINE sgd
   !---------------------------------------------------------------------
   ! <<<<<<<<<<<<<<<<<<<<<<< Activation Functions >>>>>>>>>>>>>>>>>>>>>>>
   !---------------------------------------------------------------------
   !==================================================================  
   PURE FUNCTION sigmoid (x)
      ! we will assume that all the inputs and outputs will be multidimensional
      ! like ndarray in python. this gives us more flexibility.
      ! so even a scalar input must be with dimension (1)
      REAL(kind=dp) , DIMENSION (:) , INTENT(in) :: x
      REAL(kind=dp) , DIMENSION (SIZE(x)) :: sigmoid
      sigmoid = EXP(-x)
      sigmoid = 1._dp + sigmoid
      sigmoid = 1._dp / sigmoid
   END FUNCTION sigmoid
   !==================================================================
   PURE FUNCTION sigmoid_d (y)
      ! calculates the 'equivalent' derivative of sigmoid function.
      ! used for neural networks only. it simplifies its output by
      ! rearranging the answer to use y itself instead of the input.
      REAL(kind=dp) , DIMENSION (:) , INTENT(in) :: y
      REAL(kind=dp) , DIMENSION (SIZE(y)) :: sigmoid_d
      ! actually the derivative of sigmoid is sigmoid(x)(1-sigmoid(x))
      ! but since sigmoid() here equals y, we use y itself.
      sigmoid_d = 1._dp - y
      sigmoid_d = sigmoid_d * y
   END FUNCTION sigmoid_d
   !==================================================================
   PURE FUNCTION sigmoid_rd (x)
      ! the real derivative of sigmoid.
      ! provided for reference and/or further work.
      REAL(kind=dp) , DIMENSION (:) , INTENT(in) :: x
      REAL(kind=dp) , DIMENSION (SIZE(x)) :: sigmoid_rd , s
      s = sigmoid(x)
      sigmoid_rd = (1._dp - sigmoid(x))
      sigmoid_rd = sigmoid_rd * s
   END FUNCTION sigmoid_rd
   ! ==================================================================
   PURE FUNCTION htan (x)
      ! A naming interface for tanh.
      REAL(kind=dp) , DIMENSION (:) , INTENT(in) :: x
      REAL(kind=dp) , DIMENSION (SIZE(x)) :: htan
      htan = TANH(x)
   END FUNCTION htan
   ! ==================================================================
   PURE FUNCTION tanh_d (y)
      ! calculates the 'equivalent' derivative of tanh function,
      ! used in neural networks.
      REAL(kind=dp) , DIMENSION (:) , INTENT(in) :: y
      REAL(kind=dp) , DIMENSION (SIZE(y)) :: tanh_d
      tanh_d = y**2
      tanh_d = 1._dp - tanh_d
   END FUNCTION tanh_d
   ! ==================================================================
   PURE FUNCTION tanh_rd (x)
      ! the actual derivative of tanh.
      ! provided for reference and/or further work.
      REAL(kind=dp) , DIMENSION (:) , INTENT(in) :: x
      REAL(kind=dp) , DIMENSION (SIZE(x)) :: tanh_rd
      tanh_rd = 1._dp - (TANH(x))**2
   END FUNCTION tanh_rd
   ! ==================================================================
   PURE FUNCTION smax ( y )
      ! the softmax function.
      REAL(kind=dp) , DIMENSION(:) , INTENT(in) :: y
      REAL(kind=dp) , DIMENSION (SIZE(y)) :: smax
      REAL(kind=dp) :: sum_smax , l1 , l2
      l1 = 1.e-9_dp ; l2 = 0.999999999_dp ! to avoid possible numerical instability
      smax = EXP(y)
      sum_smax = SUM(smax)
      smax = smax / sum_smax
      WHERE ( smax > l2 ) smax = l2
      WHERE ( smax < l1 ) smax = l1
   END FUNCTION smax
   ! ==================================================================
   PURE FUNCTION smax_d ( y )
      ! the 'equivalent' derivative of softmax function
      ! with regard to y (a.k.a. its output).
      REAL(kind=dp) , DIMENSION(:) , INTENT(in) :: y
      REAL(kind=dp) , DIMENSION (SIZE(y)) :: smax_d
      smax_d = 1._dp - y
      smax_d = smax_d * y
   END FUNCTION smax_d
   ! ==================================================================
   PURE FUNCTION smax_rd ( x )
      ! the real derivative of softmax function with regard to x
      ! provided for reference and/or further work.
      REAL(kind=dp) , DIMENSION(:) , INTENT(in) :: x
      REAL(kind=dp) , DIMENSION (SIZE(x)) :: smax_rd
      smax_rd = 1._dp - smax(x)
      smax_rd = smax_rd * smax(x)
   END FUNCTION smax_rd
   ! ==================================================================
   PURE FUNCTION lrelu (x)
      ! this actually represents the leaky ReLU function
      REAL(kind=dp) , DIMENSION (:) , INTENT(in) :: x
      REAL(kind=dp) , DIMENSION (SIZE(x)) :: lrelu
      lrelu = x
      WHERE ( x < 0._dp ) lrelu = 0.01_dp * x
   END FUNCTION lrelu
   ! ==================================================================
   PURE FUNCTION lrelu_d (y)
      ! the derivative of leaky ReLU function.
      REAL(kind=dp) , DIMENSION (:) , INTENT(in) :: y
      REAL(kind=dp) , DIMENSION (SIZE(y)) :: lrelu_d
      lrelu_d=1._dp
      WHERE ( y < 0._dp ) lrelu_d = 0.01_dp
   END FUNCTION lrelu_d
   ! ==================================================================
   PURE FUNCTION relu (x)
      ! the ReLU function. it may provoke numerical instabilities in certain
      ! situations
      REAL(kind=dp) , DIMENSION (:) , INTENT(in) :: x
      REAL(kind=dp) , DIMENSION (SIZE(x)) :: relu
      relu = x
      WHERE ( x < 0._dp ) relu = 0._dp
   END FUNCTION relu
   ! ==================================================================
   PURE FUNCTION relu_d (y)
      ! the derivative of ReLU function.
      REAL(kind=dp) , DIMENSION (:) , INTENT(in) :: y
      REAL(kind=dp) , DIMENSION (SIZE(y)) :: relu_d
      relu_d=1._dp
      WHERE ( y < 0._dp ) relu_d = 0._dp
   END FUNCTION relu_d
   !---------------------------------------------------------------------
   ! <<<<<<<<<<<<<<<<<<<<<<<<<< Loss Functions >>>>>>>>>>>>>>>>>>>>>>>>>
   !---------------------------------------------------------------------
   ! ==================================================================
   PURE FUNCTION mse ( y , t )
      ! mean squared error
      REAL(kind=dp) , DIMENSION (:) , INTENT(in) :: y , t
      REAL(kind=dp) , DIMENSION (SIZE(y)) :: mse
      mse = 0.5_dp*( y - t )**2
   END FUNCTION mse
   ! ==================================================================
   PURE FUNCTION mse_d ( y , t )
      ! derivative of mean squared error
      REAL(kind=dp) , DIMENSION (:) , INTENT(in) :: y , t
      REAL(kind=dp) , DIMENSION (SIZE(y)) :: mse_d
      mse_d = y - t
   END FUNCTION mse_d
   ! ==================================================================
   PURE FUNCTION sce ( y , t )
      ! the actual softmax cross entropy function
      REAL(kind=dp) , DIMENSION(:) , INTENT(in) :: y , t
      REAL(kind=dp) , DIMENSION (SIZE(y)) :: sce
      sce = smax(y)
      sce = -t * LOG(sce) - ( 1._dp - t ) * LOG(1._dp - sce)
   END FUNCTION sce
   ! ==================================================================
   PURE FUNCTION sce_d ( y , t )
      ! equivalent derivative of SCE
      REAL(kind=dp) , DIMENSION (:) , INTENT(in) :: y , t
      REAL(kind=dp) , DIMENSION (SIZE(y)) :: sce_d
      sce_d = smax ( y ) - t
   END FUNCTION sce_d
   ! ==================================================================
   !---------------------------------------------------------------------
   !---------------------------------------------------------------------
   SUBROUTINE normaldist (mean,sd,frm,arr)
      ! creates a normal distribution in the 'frm' form.
      ! frm is a list containing the dimensions of the array.
      ! using Marsaglia polar method (or Box-Muller method)
      REAL(kind=dp) , INTENT(in) :: mean , sd
      TYPE (dyn_arr) , INTENT(inout) :: arr
      
      REAL(kind=dp) , DIMENSION(:) , ALLOCATABLE :: nd
      INTEGER , DIMENSION(:) , INTENT(in) :: frm

      INTEGER, PARAMETER :: ki = selected_int_KIND(18)
      INTEGER (kind=ki) :: samples , n , nm , osize
      REAL(kind=dp) :: ur1 , ur2 , nr1 , nr2 , s

      nm = SIZE(frm)
      IF ( nm > 2 .OR. nm < 1 ) THEN
         PRINT *, "For now, just input an array indicating the &
              &dimensions for a 2D matrix or a vector only."
         STOP
      ENDIF

      IF ( nm ==2 ) THEN
         samples=INT(frm(1),ki)*INT(frm(2),ki)
         ! recording the original size
         osize = samples
         ! in case both the dimensions were odd
         IF (MOD(samples,2) .EQ. 1) samples=samples+1
         ALLOCATE (arr%m(frm(1),frm(2)))
      ELSE
         samples=INT(frm(1),ki)
         ! recording the original size
         osize = samples
         ! in case the dimension was odd
         IF (MOD(samples,2) .EQ. 1) samples=samples+1
         ALLOCATE (arr%v(frm(1)))
      ENDIF
      ALLOCATE (nd(samples))
      nd = 0
      n = 0
      DO WHILE (n < samples)
         CALL random_NUMBER(ur1)
         CALL random_NUMBER(ur2)
         ur1 = ur1 * 2.0 - 1.0
         ur2 = ur2 * 2.0 - 1.0

         s = ur1*ur1 + ur2*ur2  
         IF (s >= 1.0_dp) CYCLE

         nr1 = ur1 * SQRT(-2.0*LOG(s)/s)
         nd (n+1) =  nr1
         nr2 = ur2 * SQRT(-2.0*LOG(s)/s)
         nd (n+2) = nr2
         n = n + 2_ki
      END DO

      nd = sd*nd + mean
      IF (nm ==2) THEN
         arr%m = RESHAPE (nd,shape=[frm(1),frm(2)])
      ELSE
         arr%v = nd (1:osize)
      ENDIF

   END SUBROUTINE normaldist
   !==================================================================
   FUNCTION drop_mask (p,s)
      ! creates a drop-out mask with the keep rate of p, and with the size of s 
      REAL(kind=dp) , INTENT(in) :: p
      INTEGER , INTENT(in) :: s
      INTEGER , DIMENSION(:) , ALLOCATABLE :: drop_mask
      INTEGER, PARAMETER :: ki = selected_int_KIND(18)
      INTEGER (kind=ki) :: n
      INTEGER :: i , m
      REAL(kind=dp) :: u

      ALLOCATE (drop_mask(s))
      drop_mask = 1
      m = FLOOR ((1._dp-p)*s)
      i = 0
      DO WHILE (i <= m)
         CALL RANDOM_NUMBER (u)
         n = 1 + FLOOR (u*s)
         IF ( drop_mask(n) == 0 ) CYCLE
         drop_mask(n) = 0
         i = i + 1
      ENDDO
   END FUNCTION drop_mask
   !==================================================================
   SUBROUTINE seeder ()
      ! provides a seed for random_number function
      ! for benchmark purposes only.
      INTEGER :: j , seed_size
      INTEGER , DIMENSION(:) , ALLOCATABLE :: seed
      
      CALL random_SEED(size=seed_size)
      ALLOCATE (seed(seed_size))
      seed = [(j,j=1,seed_size)]
      CALL random_SEED(put=seed)
   END SUBROUTINE seeder
   !==================================================================
   FUNCTION randomizer(m)
      ! creates a random list from 1 to m
      INTEGER , INTENT(in) :: m
      INTEGER , DIMENSION(m) :: randomizer , l 
      INTEGER , DIMENSION(:) , ALLOCATABLE :: ln
      INTEGER :: i , j , n
      REAL :: u

      l = [(i,i=1,m)]
      ln = l
      n = m
      DO i = 1 , m
         CALL RANDOM_NUMBER (u)
         j = 1 + FLOOR (u*n)
         randomizer(i) = ln(j)
         ln(j) = 0
         ln = PACK(ln,ln/=0)
         n = n - 1
      ENDDO
   END FUNCTION randomizer
   !==================================================================
#ifdef CLI
   SUBROUTINE parser(mynn)
      ! CLI options
      USE m_cli
      ! we define a variable of class nn so that we can init it here.
      CLASS (nn) , INTENT(inout) :: mynn

      CHARACTER(len=:) , ALLOCATABLE     :: cmd
      CHARACTER(len=256)                 :: message
      INTEGER                            :: ios
      CHARACTER(len=:) , ALLOCATABLE     :: help_text(:), version_text(:)

      INTEGER, DIMENSION (:) , ALLOCATABLE :: mlayers
      REAL , DIMENSION(:) , ALLOCATABLE :: mdrates
      ! making auxiliary vars for holding layers and drates
      INTEGER , DIMENSION (100) :: layers = -100
      REAL , DIMENSION(100)    :: drates = -100.0
      CHARACTER(len=20) , DIMENSION(10) :: afuncs="" , lfuncs="" , optim=""
      REAL :: lrs , lrf , mu
      INTEGER :: epoch

      INTEGER :: i , nl , nd , naf
      TYPE(pointerproc1) , DIMENSION(:) , ALLOCATABLE :: afunc
      PROCEDURE (ndarray2) , POINTER :: lfunc
      PROCEDURE (ndarray0) , POINTER :: ofunc
      
      NAMELIST /args/ layers , drates , afuncs , lfuncs , &
           optim , lrs , lrf , mu , epoch
      ! we put the default values into mu, epoch, lrs, lrf and drates, because
      ! these are optional arguments in the init procedure. The rest would be read
      ! from the command lines and the values here do not count.

      help_text=[CHARACTER(len=80) :: &
           'The following options are available:                    ', &
           '                                                        ', &
           '     --layers                                           ', &
           '   Sets the the structure of neurons and the layers.    ', &
           '   Mandatory. For example:                              ', &
           '             --layers 2000,200,300,150,12               ', &
           '                                                        ', &
           '     --drates                                           ', &
           '   Determines the droprates for each of the layers.     ', &
           '   Optional (default = 0 for all).                      ', &
           '                                                        ', &
           '     --lrs                                              ', &
           '   The starting learning rate. Optional (defalut = 0.1).', &
           '                                                        ', &
           '     --lrf                                              ', &
           '   The final learning rate (for implementing decay).    ', &
           '   Optional (default = 0.1).                            ', &
           '                                                        ', &
           '     --afuncs                                           ', &
           '   The list of activation functions which will be used  ', &
           '   for each layer. Mandatory.                           ', &
           '   Available: "sigmoid" , "relu" , "lrelu" , "htan"     ', &
           '   "smax". For example for 3 layers:                    ', &
           '             --afuncs ''"null","htan","smax"''          ', &
           '                                                        ', &
           '     --lfuncs                                           ', &
           '   The loss function.                                   ', &
           '   Optional (mean squared error "mse" by default).      ', &
           '   Available: "mse" (mean squared err) , "sce" (softmax ', &
           '   cross entropy).                                      ', &
           '                                                        ', &
           '     --optim                                            ', &
           '   The optimizer procedure. Optional. Avilable: "sgd"   ', &
           '   (stochastic gradient descent, default).              ', &
           '                                                        ', &
           '     --mu                                               ', &
           '   The momentum. Optional (default = 0).                ', &
           '                                                        ', &
           '     --epoch                                            ', &
           '   The number of retrainings per set of data. Optional  ', &
           '   (default = 1)                                        ', &
           '' ]
      
      version_text=[CHARACTER(len=80) :: &
           '@(#)PROGRAM:     FortNN          >', &
           '@(#)DESCRIPTION: ANN Solver in Fortran  >', &
           '@(#)VERSION:     0.1     >', &
           '@(#)AUTHOR:      Kete Tefid       >', &
           '@(#)LICENSE:     GPL v3    >', &
           '' ]
      
      cmd = commandline('--layers 784,100,10 --drates -1.,-1.,-1. &
           &--afuncs "null","sigmoid","sigmoid" &
           & --lfuncs "mse" --optim "sgd" --lrs -0.1 --lrf -0.1 --mu 0. --epoch 1')
      
      READ(cmd,nml=args,iostat=ios,iomsg=message)
      CALL check_commandline(ios,message,help_text,version_text)

      ! the trick was to temporarily hold the values for layers and droprates in
      ! big-enough arrays, and then transfer to the main arrays.
      nl = 0
      DO i = 1 , 100
         IF ( layers(i) /= -100 ) THEN
            nl = nl + 1
         ELSE
            EXIT
         ENDIF
      ENDDO
      ALLOCATE (mlayers(nl))
      
      nd = 0
      DO i = 1 , 100
         IF ( drates(i) > 0. ) THEN
            nd = nd + 1
         ELSE
            EXIT
         ENDIF
      ENDDO
      ALLOCATE (mdrates(nd))

      mlayers = layers (1:nl)

      mdrates = drates (1:nd)

      WRITE (*,args)

      ! number of active functions supplied
      naf = 0
      DO WHILE (TRIM(afuncs(naf+1)) .NE. "")
         naf = naf + 1
      END DO

      ! activation functions like layers will start from zero.
      ALLOCATE (afunc(0:naf-1))

      DO i = 1 , naf 
         SELECT CASE(TRIM(afuncs(i)))
         CASE ("relu")
            !PRINT*,'RELU'
            afunc(i-1)%f => relu
         CASE ("lrelu")
            !PRINT*,'LRELU'
            afunc(i-1)%f => lrelu
         CASE ("htan")
            !PRINT*,'HTAN'
            afunc(i-1)%f => htan
         CASE ("sigmoid")
            !PRINT*,"SIGMOID"
            afunc(i-1)%f => sigmoid
         CASE ("smax")
            !PRINT*,"SOFTMAX"
            afunc(i-1)%f => smax
         CASE ("null")
            !PRINT*,"NULL"
            afunc(i-1)%f => NULL()
         CASE default
            PRINT*,"Your activation function is invalid."
            STOP
         END SELECT
      END DO

      i = 1
      DO WHILE (TRIM(lfuncs(i)) .NE. "")
         SELECT CASE(TRIM(lfuncs(i)))
         CASE ("mse")
            !PRINT*,'MSE'
            lfunc => mse
         CASE ("sce")
            !PRINT*,'SCE'
            lfunc => sce
         CASE default
            PRINT*,"Your loss function is invalid."
            STOP
         END SELECT
         i = i + 1
      END DO

      i = 1
      DO WHILE (TRIM(optim(i)) .NE. "")
         SELECT CASE(TRIM(optim(i)))
         CASE ("sgd")
            !PRINT*,'SGD'
            ofunc => sgd
         CASE default
            PRINT*,"Your optimizer function is invalid."
            STOP
         END SELECT
         i = i + 1
      END DO

      ! in case the user supplied one learning rate but not both
      IF (lrf > 0 .AND. lrs < 0) lrs = lrf
      IF (lrf < 0 .and. lrs > 0) lrf = lrs
      
      ! if the user did not supply the droprates, the default droprate would be -1
      ! and we would omit it from the init.
      IF (drates(1)<0.) THEN 
         CALL mynn%init(layers=mlayers , activ_func=afunc ,  &
              loss_func=lfunc , optimizer=ofunc , lrs=lrs , lrf=lrf , mu=mu , epoch=epoch)
      ELSE
         CALL mynn%init(layers=mlayers , droprates=mdrates , activ_func=afunc ,  &
              loss_func=lfunc , optimizer=ofunc , lrs=lrs , lrf=lrf , mu=mu , epoch=epoch)
      ENDIF
   END SUBROUTINE parser
#endif
   !==================================================================
END MODULE FortNN
!----------------------------------------------------------------------------------
