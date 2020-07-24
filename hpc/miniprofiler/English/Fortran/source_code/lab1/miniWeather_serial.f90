
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! miniWeather
!! Author: Matt Norman <normanmr@ornl.gov>  , Oak Ridge National Laboratory
!! This code simulates dry, stratified, compressible, non-hydrostatic fluid flows
!! For documentation, please see the attached documentation in the "documentation" folder
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

program miniweather
  use nvtx
  !implicit none

  !Declare the precision for the real constants (at least 15 digits of accuracy = double precision)
  integer , parameter :: rp = selected_real_kind(15)
  !Define some physical constants to use throughout the simulation
  real(rp), parameter :: pi        = 3.14159265358979323846264338327_rp   !Pi
  real(rp), parameter :: grav      = 9.8_rp                               !Gravitational acceleration (m / s^2)
  real(rp), parameter :: cp        = 1004._rp                             !Specific heat of dry air at constant pressure
  real(rp), parameter :: rd        = 287._rp                              !Dry air constant for equation of state (P=rho*rd*T)
  real(rp), parameter :: p0        = 1.e5_rp                              !Standard pressure at the surface in Pascals
  real(rp), parameter :: C0        = 27.5629410929725921310572974482_rp   !Constant to translate potential temperature into pressure (P=C0*(rho*theta)**gamma)
  real(rp), parameter :: gamma     = 1.40027894002789400278940027894_rp   !gamma=cp/Rd
  !Define domain and stability-related constants
  real(rp), parameter :: xlen      = 2.e4_rp    !Length of the domain in the x-direction (meters)
  real(rp), parameter :: zlen      = 1.e4_rp    !Length of the domain in the z-direction (meters)
  real(rp), parameter :: hv_beta   = 0.25_rp     !How strong to diffuse the solution: hv_beta \in [0:1]
  real(rp), parameter :: cfl       = 1.50_rp    !"Courant, Friedrichs, Lewy" number (for numerical stability)
  real(rp), parameter :: max_speed = 450        !Assumed maximum wave speed during the simulation (speed of sound + speed of wind) (meter / sec)
  integer , parameter :: hs        = 2          !"Halo" size: number of cells needed for a full "stencil" of information for reconstruction
  integer , parameter :: sten_size = 4          !Size of the stencil used for interpolation

  !Parameters for indexing and flags
  integer , parameter :: NUM_VARS = 4           !Number of fluid state variables
  integer , parameter :: ID_DENS  = 1           !index for density ("rho")
  integer , parameter :: ID_UMOM  = 2           !index for momentum in the x-direction ("rho * u")
  integer , parameter :: ID_WMOM  = 3           !index for momentum in the z-direction ("rho * w")
  integer , parameter :: ID_RHOT  = 4           !index for density * potential temperature ("rho * theta")
  integer , parameter :: DIR_X = 1              !Integer constant to express that this operation is in the x-direction
  integer , parameter :: DIR_Z = 2              !Integer constant to express that this operation is in the z-direction

  !Gauss-Legendre quadrature points and weights on the domain [0:1]
  integer , parameter :: nqpoints = 3
  real(rp) :: qpoints (nqpoints) = (/ 0.112701665379258311482073460022E0_rp , 0.500000000000000000000000000000E0_rp , 0.887298334620741688517926539980E0_rp /)
  real(rp) :: qweights(nqpoints) = (/ 0.277777777777777777777777777779E0_rp , 0.444444444444444444444444444444E0_rp , 0.277777777777777777777777777779E0_rp /)

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! Variables that are initialized but remain static over the course of the simulation
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  real(rp) :: sim_time                          !total simulation time in seconds
  real(rp) :: output_freq                       !frequency to perform output in seconds
  real(rp) :: dt                                !Model time step (seconds)
  integer  :: nx, nz                            !Number of local grid cells in the x- and z- dimensions 
  real(rp) :: dx, dz                            !Grid space length in x- and z-dimension (meters)
  integer  :: nx_glob, nz_glob                  !Number of total grid cells in the x- and z- dimensions
  integer  :: i_beg, k_beg                      !beginning index in the x- and z-directions 
  integer  :: nranks, myrank                    !my rank id
  integer  :: left_rank, right_rank             

  real(rp), allocatable :: hy_dens_cell      (:)      !hydrostatic density (vert cell avgs).   Dimensions: (1-hs:nz+hs)
  real(rp), allocatable :: hy_dens_theta_cell(:)      !hydrostatic rho*t (vert cell avgs).     Dimensions: (1-hs:nz+hs)
  real(rp), allocatable :: hy_dens_int       (:)      !hydrostatic density (vert cell interf). Dimensions: (1:nz+1)
  real(rp), allocatable :: hy_dens_theta_int (:)      !hydrostatic rho*t (vert cell interf).   Dimensions: (1:nz+1)
  real(rp), allocatable :: hy_pressure_int   (:)      !hydrostatic press (vert cell interf).   Dimensions: (1:nz+1)

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! Variables that are dynamics over the course of the simulation
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  real(rp) :: etime                             !Elapsed model time
  real(rp) :: output_counter                    !Helps determine when it's time to do output
  !Runtime variable arrays
  real(rp), allocatable :: state             (:,:,:)  !Fluid state.                            Dimensions: (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
  real(rp), allocatable :: state_tmp         (:,:,:)  !Fluid state.                            Dimensions: (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
  real(rp), allocatable :: flux              (:,:,:)  !Cell interface fluxes.                  Dimensions: (nx+1,nz+1,NUM_VARS)
  real(rp), allocatable :: tend              (:,:,:)  !Fluid state tendencies.                 Dimensions: (nx,nz,NUM_VARS)
  integer(8) :: t1, t2, rate                          !For CPU Timings

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! THE MAIN PROGRAM STARTS HERE
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! BEGIN USER-CONFIGURABLE PARAMETERS
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !The x-direction length is twice as long as the z-direction length
  !So, you'll want to have nx_glob be twice as large as nz_glob
  nx_glob = 400        !Number of total cells in the x-dirction
  nz_glob = 200        !Number of total cells in the z-dirction
  sim_time = 10 !How many seconds to run the simulation
  output_freq = 100 !How frequently to output data to file (in seconds)
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! END USER-CONFIGURABLE PARAMETERS
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !Allocate arrays, initialize the grid and the data
  call nvtxStartRange("Total")
  call init()

  !Output the initial state
  !call output(state,etime)

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! MAIN TIME STEP LOOP
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
call system_clock(t1)
  call nvtxStartRange("while")
  do while (etime < sim_time)
    !If the time step leads to exceeding the simulation time, shorten it for the last step
    if (etime + dt > sim_time) dt = sim_time - etime
    !Perform a single time step
    call nvtxStartRange("perform_timestep")
    call perform_timestep(state,state_tmp,flux,tend,dt)
    call nvtxEndRange
    !Inform the user
   write(*,*) 'Elapsed Time: ', etime , ' / ' , sim_time
    !Update the elapsed time and output counter
    etime = etime + dt
    output_counter = output_counter + dt
    !If it's time for output, reset the counter, and do output
    if (output_counter >= output_freq) then
      output_counter = output_counter - output_freq
     !call output(state,etime)
    endif
  enddo
  call nvtxEndRange

    call system_clock(t2,rate)
    write(*,*) "CPU Time: ",dble(t2-t1)/dble(rate)


  !Deallocate
  call finalize()
  call nvtxEndRange

contains


  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! HELPER SUBROUTINES AND FUNCTIONS GO HERE
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


  !Performs a single dimensionally split time step using a simple low-storate three-stage Runge-Kutta time integrator
  !The dimensional splitting is a second-order-accurate alternating Strang splitting in which the
  !order of directions is alternated each time step.
  !The Runge-Kutta method used here is defined as follows:
  ! q*     = q[n] + dt/3 * rhs(q[n])
  ! q**    = q[n] + dt/2 * rhs(q*  )
  ! q[n+1] = q[n] + dt/1 * rhs(q** )
  subroutine perform_timestep(state,state_tmp,flux,tend,dt)
    implicit none
    real(rp), intent(inout) :: state    (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
    real(rp), intent(  out) :: state_tmp(1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
    real(rp), intent(  out) :: flux     (nx+1,nz+1,NUM_VARS)
    real(rp), intent(  out) :: tend     (nx,nz,NUM_VARS)
    real(rp), intent(in   ) :: dt
    logical, save :: direction_switch = .true.
    if (direction_switch) then
      !x-direction first
      call semi_discrete_step( state , state     , state_tmp , dt / 3 , DIR_X , flux , tend )
      call semi_discrete_step( state , state_tmp , state_tmp , dt / 2 , DIR_X , flux , tend )
      call semi_discrete_step( state , state_tmp , state     , dt / 1 , DIR_X , flux , tend )
      !z-direction second
      call semi_discrete_step( state , state     , state_tmp , dt / 3 , DIR_Z , flux , tend )
      call semi_discrete_step( state , state_tmp , state_tmp , dt / 2 , DIR_Z , flux , tend )
      call semi_discrete_step( state , state_tmp , state     , dt / 1 , DIR_Z , flux , tend )
    else
      !z-direction second
      call semi_discrete_step( state , state     , state_tmp , dt / 3 , DIR_Z , flux , tend )
      call semi_discrete_step( state , state_tmp , state_tmp , dt / 2 , DIR_Z , flux , tend )
      call semi_discrete_step( state , state_tmp , state     , dt / 1 , DIR_Z , flux , tend )
      !x-direction first
      call semi_discrete_step( state , state     , state_tmp , dt / 3 , DIR_X , flux , tend )
      call semi_discrete_step( state , state_tmp , state_tmp , dt / 2 , DIR_X , flux , tend )
      call semi_discrete_step( state , state_tmp , state     , dt / 1 , DIR_X , flux , tend )
    endif
    direction_switch = .not. direction_switch
  end subroutine perform_timestep


  !Perform a single semi-discretized step in time with the form:
  !state_out = state_init + dt * rhs(state_forcing)
  !Meaning the step starts from state_init, computes the rhs using state_forcing, and stores the result in state_out
  subroutine semi_discrete_step( state_init , state_forcing , state_out , dt , dir , flux , tend )
    implicit none
    real(rp), intent(in   ) :: state_init   (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
    real(rp), intent(inout) :: state_forcing(1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
    real(rp), intent(  out) :: state_out    (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
    real(rp), intent(  out) :: flux         (nx+1,nz+1,NUM_VARS)
    real(rp), intent(  out) :: tend         (nx,nz,NUM_VARS)
    real(rp), intent(in   ) :: dt
    integer , intent(in   ) :: dir
    integer :: i,k,ll

    if     (dir == DIR_X) then
      !Set the halo values in the x-direction
      call set_halo_values_x(state_forcing)
      !Compute the time tendencies for the fluid state in the x-direction
      call compute_tendencies_x(state_forcing,flux,tend)
    elseif (dir == DIR_Z) then
      !Set the halo values in the z-direction
      call set_halo_values_z(state_forcing)
      !Compute the time tendencies for the fluid state in the z-direction
      call compute_tendencies_z(state_forcing,flux,tend)
    endif

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !! TODO: THREAD ME
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !Apply the tendencies to the fluid state
    do ll = 1 , NUM_VARS
      do k = 1 , nz
        do i = 1 , nx
          state_out(i,k,ll) = state_init(i,k,ll) + dt * tend(i,k,ll)
        enddo
      enddo
    enddo
  end subroutine semi_discrete_step


  !Compute the time tendencies of the fluid state using forcing in the x-direction

  !First, compute the flux vector at each cell interface in the x-direction (including hyperviscosity)
  !Then, compute the tendencies using those fluxes
  subroutine compute_tendencies_x(state,flux,tend)
    implicit none
    real(rp), intent(in   ) :: state(1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
    real(rp), intent(  out) :: flux (nx+1,nz+1,NUM_VARS)
    real(rp), intent(  out) :: tend (nx,nz,NUM_VARS)
    integer :: i,k,ll,s
    real(rp) :: r,u,w,t,p, stencil(4), d3_vals(NUM_VARS), vals(NUM_VARS), hv_coef
    !Compute the hyperviscosity coeficient
    hv_coef = -hv_beta * dx / (16*dt)
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !! TODO: THREAD ME
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !Compute fluxes in the x-direction for each cell
    do k = 1 , nz
      do i = 1 , nx+1
        !Use fourth-order interpolation from four cell averages to compute the value at the interface in question
        do ll = 1 , NUM_VARS
          do s = 1 , sten_size
            stencil(s) = state(i-hs-1+s,k,ll)
          enddo
          !Fourth-order-accurate interpolation of the state
          vals(ll) = -stencil(1)/12 + 7*stencil(2)/12 + 7*stencil(3)/12 - stencil(4)/12
          !First-order-accurate interpolation of the third spatial derivative of the state (for artificial viscosity)
          d3_vals(ll) = -stencil(1) + 3*stencil(2) - 3*stencil(3) + stencil(4)
        enddo

        !Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
        r = vals(ID_DENS) + hy_dens_cell(k)
        u = vals(ID_UMOM) / r
        w = vals(ID_WMOM) / r
        t = ( vals(ID_RHOT) + hy_dens_theta_cell(k) ) / r
        p = C0*(r*t)**gamma

        !Compute the flux vector
        flux(i,k,ID_DENS) = r*u     - hv_coef*d3_vals(ID_DENS)
        flux(i,k,ID_UMOM) = r*u*u+p - hv_coef*d3_vals(ID_UMOM)
        flux(i,k,ID_WMOM) = r*u*w   - hv_coef*d3_vals(ID_WMOM)
        flux(i,k,ID_RHOT) = r*u*t   - hv_coef*d3_vals(ID_RHOT)
      enddo
    enddo

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !! TODO: THREAD ME
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !Use the fluxes to compute tendencies for each cell
    do ll = 1 , NUM_VARS
      do k = 1 , nz
        do i = 1 , nx
          tend(i,k,ll) = -( flux(i+1,k,ll) - flux(i,k,ll) ) / dx
        enddo
      enddo
    enddo
  end subroutine compute_tendencies_x


  !Compute the time tendencies of the fluid state using forcing in the z-direction

  !First, compute the flux vector at each cell interface in the z-direction (including hyperviscosity)
  !Then, compute the tendencies using those fluxes
  subroutine compute_tendencies_z(state,flux,tend)
    implicit none
    real(rp), intent(in   ) :: state(1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
    real(rp), intent(  out) :: flux (nx+1,nz+1,NUM_VARS)
    real(rp), intent(  out) :: tend (nx,nz,NUM_VARS)
    integer :: i,k,ll,s
    real(rp) :: r,u,w,t,p, stencil(4), d3_vals(NUM_VARS), vals(NUM_VARS), hv_coef
    !Compute the hyperviscosity coeficient
    hv_coef = -hv_beta * dz / (16*dt)
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !! TODO: THREAD ME
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !Compute fluxes in the x-direction for each cell
    do k = 1 , nz+1
      do i = 1 , nx
        !Use fourth-order interpolation from four cell averages to compute the value at the interface in question
        do ll = 1 , NUM_VARS
          do s = 1 , sten_size
            stencil(s) = state(i,k-hs-1+s,ll)
          enddo
          !Fourth-order-accurate interpolation of the state
          vals(ll) = -stencil(1)/12 + 7*stencil(2)/12 + 7*stencil(3)/12 - stencil(4)/12
          !First-order-accurate interpolation of the third spatial derivative of the state
          d3_vals(ll) = -stencil(1) + 3*stencil(2) - 3*stencil(3) + stencil(4)
        enddo

        !Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
        r = vals(ID_DENS) + hy_dens_int(k)
        u = vals(ID_UMOM) / r
        w = vals(ID_WMOM) / r
        t = ( vals(ID_RHOT) + hy_dens_theta_int(k) ) / r
        p = C0*(r*t)**gamma - hy_pressure_int(k)
        !Enforce vertical boundary condition and exact mass conservation
        if (k == 1 .or. k == nz+1) then
          w                = 0
          d3_vals(ID_DENS) = 0
        endif

        !Compute the flux vector with hyperviscosity
        flux(i,k,ID_DENS) = r*w     - hv_coef*d3_vals(ID_DENS)
        flux(i,k,ID_UMOM) = r*w*u   - hv_coef*d3_vals(ID_UMOM)
        flux(i,k,ID_WMOM) = r*w*w+p - hv_coef*d3_vals(ID_WMOM)
        flux(i,k,ID_RHOT) = r*w*t   - hv_coef*d3_vals(ID_RHOT)
      enddo
    enddo

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !! TODO: THREAD ME
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !Use the fluxes to compute tendencies for each cell
    do ll = 1 , NUM_VARS
      do k = 1 , nz
        do i = 1 , nx
          tend(i,k,ll) = -( flux(i,k+1,ll) - flux(i,k,ll) ) / dz
          if (ll == ID_WMOM) then
            tend(i,k,ID_WMOM) = tend(i,k,ID_WMOM) - state(i,k,ID_DENS)*grav
          endif
        enddo
      enddo
    enddo
  end subroutine compute_tendencies_z


  !Set halo values in the x-direction. 
  subroutine set_halo_values_x(state)
    implicit none
    real(rp), intent(inout) :: state(1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
    integer :: k, ll
    real(rp) :: z

    do ll = 1 , NUM_VARS
      do k = 1 , nz
        state(-1  ,k,ll) = state(nx-1,k,ll)
        state(0   ,k,ll) = state(nx  ,k,ll)
        state(nx+1,k,ll) = state(1   ,k,ll)
        state(nx+2,k,ll) = state(2   ,k,ll)
      enddo
    enddo
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      if (myrank == 0) then
        do k = 1 , nz
          z = (k_beg-1 + k-0.5_rp)*dz
          if (abs(z-3*zlen/4) <= zlen/16) then
            state(-1:0,k,ID_UMOM) = (state(-1:0,k,ID_DENS)+hy_dens_cell(k)) * 50._rp
            state(-1:0,k,ID_RHOT) = (state(-1:0,k,ID_DENS)+hy_dens_cell(k)) * 298._rp - hy_dens_theta_cell(k)
          endif
        enddo
      endif
  end subroutine set_halo_values_x


  !Set halo values in the z-direction. 
  !decomposition in the vertical direction
  subroutine set_halo_values_z(state)
    implicit none
    real(rp), intent(inout) :: state(1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
    integer :: i, ll
    real(rp), parameter :: mnt_width = xlen/8
    real(rp) :: x, xloc, mnt_deriv
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !! TODO: THREAD ME
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    do ll = 1 , NUM_VARS
      do i = 1-hs,nx+hs
        if (ll == ID_WMOM) then
          state(i,-1  ,ll) = 0
          state(i,0   ,ll) = 0
          state(i,nz+1,ll) = 0
          state(i,nz+2,ll) = 0
        else
          state(i,-1  ,ll) = state(i,1 ,ll)
          state(i,0   ,ll) = state(i,1 ,ll)
          state(i,nz+1,ll) = state(i,nz,ll)
          state(i,nz+2,ll) = state(i,nz,ll)
        endif
      enddo
    enddo
  end subroutine set_halo_values_z


  !Initialize some grid settings, allocate data, and initialize the full grid and data
  subroutine init()
    implicit none
    integer :: i, k, ii, kk, ll, ierr
    real(rp) :: x, z, r, u, w, t, hr, ht

    !Set the cell grid size
    dx = xlen / nx_glob
    dz = zlen / nz_glob

    nranks = 1
    myrank = 0
    i_beg = 1
    nx = nx_glob
    left_rank = 0
    right_rank = 0


    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !! YOU DON'T NEED TO ALTER ANYTHING BELOW THIS POINT IN THE CODE
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   
    k_beg = 1
    nz = nz_glob
   

    !Allocate the model data
    allocate(state             (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS))
    allocate(state_tmp         (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS))
    allocate(flux              (nx+1,nz+1,NUM_VARS))
    allocate(tend              (nx,nz,NUM_VARS))
    allocate(hy_dens_cell      (1-hs:nz+hs))
    allocate(hy_dens_theta_cell(1-hs:nz+hs))
    allocate(hy_dens_int       (nz+1))
    allocate(hy_dens_theta_int (nz+1))
    allocate(hy_pressure_int   (nz+1))

    !Define the maximum stable time step based on an assumed maximum wind speed
    dt = min(dx,dz) / max_speed * cfl
    !Set initial elapsed model time and output_counter to zero
    etime = 0
    output_counter = 0

    !Display some grid information
 
      write(*,*) 'nx_glob, nz_glob:', nx_glob, nz_glob
      write(*,*) 'dx,dz: ',dx,dz
      write(*,*) 'dt: ',dt


    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !! Initialize the cell-averaged fluid state via Gauss-Legendre quadrature
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    state = 0
    do k = 1-hs , nz+hs
      do i = 1-hs , nx+hs
        !Initialize the state to zero
        !Use Gauss-Legendre quadrature to initialize a hydrostatic balance + temperature perturbation
        do kk = 1 , nqpoints
          do ii = 1 , nqpoints
            !Compute the x,z location within the global domain based on cell and quadrature index
            x = (i_beg-1 + i-0.5_rp) * dx + (qpoints(ii)-0.5_rp)*dx
            z = (k_beg-1 + k-0.5_rp) * dz + (qpoints(kk)-0.5_rp)*dz

            !Set the fluid state based on the user's specification
            call injection      (x,z,r,u,w,t,hr,ht)

            !Store into the fluid state array
            state(i,k,ID_DENS) = state(i,k,ID_DENS) + r                         * qweights(ii)*qweights(kk)
            state(i,k,ID_UMOM) = state(i,k,ID_UMOM) + (r+hr)*u                  * qweights(ii)*qweights(kk)
            state(i,k,ID_WMOM) = state(i,k,ID_WMOM) + (r+hr)*w                  * qweights(ii)*qweights(kk)
            state(i,k,ID_RHOT) = state(i,k,ID_RHOT) + ( (r+hr)*(t+ht) - hr*ht ) * qweights(ii)*qweights(kk)
          enddo
        enddo
        do ll = 1 , NUM_VARS
          state_tmp(i,k,ll) = state(i,k,ll)
        enddo
      enddo
    enddo
    !Compute the hydrostatic background state over vertical cell averages
    hy_dens_cell = 0
    hy_dens_theta_cell = 0
    do k = 1-hs , nz+hs
      do kk = 1 , nqpoints
        z = (k_beg-1 + k-0.5_rp) * dz + (qpoints(kk)-0.5_rp)*dz
        !Set the fluid state based on the user's specification
        call injection      (0._rp,z,r,u,w,t,hr,ht)
        hy_dens_cell(k)       = hy_dens_cell(k)       + hr    * qweights(kk)
        hy_dens_theta_cell(k) = hy_dens_theta_cell(k) + hr*ht * qweights(kk)
      enddo
    enddo
    !Compute the hydrostatic background state at vertical cell interfaces
    do k = 1 , nz+1
      z = (k_beg-1 + k-1) * dz
      call injection      (0._rp,z,r,u,w,t,hr,ht)
      hy_dens_int      (k) = hr
      hy_dens_theta_int(k) = hr*ht
      hy_pressure_int  (k) = C0*(hr*ht)**gamma
    enddo
  end subroutine init


  !This test case is initially balanced but injects fast, cold air from the left boundary near the model top
  subroutine injection(x,z,r,u,w,t,hr,ht)
    implicit none
    real(rp), intent(in   ) :: x, z        !x- and z- location of the point being sampled
    real(rp), intent(  out) :: r, u, w, t  !Density, uwind, wwind, and potential temperature
    real(rp), intent(  out) :: hr, ht      !Hydrostatic density and potential temperature
    call hydro_const_theta(z,hr,ht)
    r = 0
    t = 0
    u = 0
    w = 0
  end subroutine injection

  !Establish hydrstatic balance using constant potential temperature (thermally neutral atmosphere)
  subroutine hydro_const_theta(z,r,t)
    implicit none
    real(rp), intent(in   ) :: z  !x- and z- location of the point being sampled
    real(rp), intent(  out) :: r, t  !Density and potential temperature at this point
    real(rp), parameter :: theta0 = 300._rp  !Background potential temperature
    real(rp), parameter :: exner0 = 1._rp    !Surface-level Exner pressure
    real(rp) :: p,exner,rt
    !Establish hydrostatic balance first using Exner pressure
    t = theta0                                  !Potential Temperature at z
    exner = exner0 - grav * z / (cp * theta0)   !Exner pressure at z
    p = p0 * exner**(cp/rd)                     !Pressure at z
    rt = (p / c0)**(1._rp / gamma)              !rho*theta at z
    r = rt / t                                  !Density at z
  end subroutine hydro_const_theta

  !Deallocate and call Finalize
  subroutine finalize()
    implicit none
    integer :: ierr
    deallocate(state             )
    deallocate(state_tmp         )
    deallocate(flux              )
    deallocate(tend              )
    deallocate(hy_dens_cell      )
    deallocate(hy_dens_theta_cell)
    deallocate(hy_dens_int       )
    deallocate(hy_dens_theta_int )
    deallocate(hy_pressure_int   )
  end subroutine finalize


  !Output the fluid state (state) to a NetCDF file at a given elapsed model time (etime)
  !The file I/O uses parallel-netcdf, the only external library required for this mini-app.
  !If it's too cumbersome, you can comment the I/O out, but you'll miss out on some potentially cool graphics
  subroutine output(state,etime)
    implicit none

    real(rp), intent(in) :: state(1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
    real(rp), intent(in) :: etime
    integer :: ncid, t_dimid, x_dimid, z_dimid, dens_varid, uwnd_varid, wwnd_varid, theta_varid, t_varid
    integer :: i,k, __LINE__
    integer, save :: num_out = 0
    integer :: len, st1(1),ct1(1),st3(3),ct3(3)
    !Temporary arrays to hold density, u-wind, w-wind, and potential temperature (theta)
    real(rp), allocatable :: dens(:,:), uwnd(:,:), wwnd(:,:), theta(:,:)
    real(rp) :: etimearr(1)
    !Inform the user
    write(*,*) '*** OUTPUT ***'
    !Allocate some (big) temp arrays
    allocate(dens (nx,nz))
    allocate(uwnd (nx,nz))
    allocate(wwnd (nx,nz))
    allocate(theta(nx,nz))

    !If the elapsed time is zero, create the file. Otherwise, open the file
    if (etime == 0) then
      !Create the file
      call ncwrap( nf90_create('reference.nc' , nf90_clobber , ncid ) , __LINE__ )
      !Create the dimensions
      len=nf90_unlimited; call ncwrap( nf90_def_dim( ncid , 't' , len , t_dimid ) , __LINE__ )
      len=nx_glob       ; call ncwrap( nf90_def_dim( ncid , 'x' , len , x_dimid ) , __LINE__ )
      len=nz_glob       ; call ncwrap( nf90_def_dim( ncid , 'z' , len , z_dimid ) , __LINE__ )
      !Create the variables
      call ncwrap( nf_def_var( ncid , 't' , NF90_DOUBLE , 1 , (/ t_dimid /) , t_varid ) , __LINE__ )
      call ncwrap( nf_def_var( ncid , 'dens'  , nf90_double , 3 , (/ x_dimid , z_dimid , t_dimid /) ,  dens_varid ) , __LINE__ )
      call ncwrap( nf_def_var( ncid , 'uwnd'  , nf90_double , 3 , (/ x_dimid , z_dimid , t_dimid /) ,  uwnd_varid ) , __LINE__ )
      call ncwrap( nf_def_var( ncid , 'wwnd'  , nf90_double , 3 , (/ x_dimid , z_dimid , t_dimid /) ,  wwnd_varid ) , __LINE__ )
      call ncwrap( nf_def_var( ncid , 'theta' , nf90_double , 3 , (/ x_dimid , z_dimid , t_dimid /) , theta_varid ) , __LINE__ )
      !End "define" mode
      call ncwrap( nf90_enddef( ncid ) , __LINE__ )
    else
      !Open the file
      call ncwrap( nf90_open( 'reference.nc' , nf90_write , ncid ) , __LINE__ )
      !Get the variable IDs
      call ncwrap( nf90_inq_varid( ncid , 'dens'  ,  dens_varid ) , __LINE__ )
      call ncwrap( nf90_inq_varid( ncid , 'uwnd'  ,  uwnd_varid ) , __LINE__ )
      call ncwrap( nf90_inq_varid( ncid , 'wwnd'  ,  wwnd_varid ) , __LINE__ )
      call ncwrap( nf90_inq_varid( ncid , 'theta' , theta_varid ) , __LINE__ )
      call ncwrap( nf90_inq_varid( ncid , 't'     ,     t_varid ) , __LINE__ )
    endif

    !Store perturbed values in the temp arrays for output
    do k = 1 , nz
      do i = 1 , nx
        dens (i,k) = state(i,k,ID_DENS)
        uwnd (i,k) = state(i,k,ID_UMOM) / ( hy_dens_cell(k) + state(i,k,ID_DENS) )
        wwnd (i,k) = state(i,k,ID_WMOM) / ( hy_dens_cell(k) + state(i,k,ID_DENS) )
        theta(i,k) = ( state(i,k,ID_RHOT) + hy_dens_theta_cell(k) ) / ( hy_dens_cell(k) + state(i,k,ID_DENS) ) - hy_dens_theta_cell(k) / hy_dens_cell(k)
      enddo
    enddo

    !Write the grid data to file with all the processes writing collectively
    st3=(/i_beg,k_beg,num_out+1/); ct3=(/nx,nz,1/); call ncwrap( nf_put_vara_double( ncid ,  dens_varid , st3 , ct3 , dens  ) , __LINE__ )
    st3=(/i_beg,k_beg,num_out+1/); ct3=(/nx,nz,1/); call ncwrap( nf_put_vara_double( ncid ,  uwnd_varid , st3 , ct3 , uwnd  ) , __LINE__ )
    st3=(/i_beg,k_beg,num_out+1/); ct3=(/nx,nz,1/); call ncwrap( nf_put_vara_double( ncid ,  wwnd_varid , st3 , ct3 , wwnd  ) , __LINE__ )
    st3=(/i_beg,k_beg,num_out+1/); ct3=(/nx,nz,1/); call ncwrap( nf_put_vara_double( ncid , theta_varid , st3 , ct3 , theta ) , __LINE__ )

    !Only the master process needs to write the elapsed time
    !write elapsed time to file

      st1=(/num_out+1/); ct1=(/1/); etimearr(1) = etime; call ncwrap( nf_put_vara_double( ncid , t_varid , st1 , ct1 , etimearr ) , __LINE__ )



    !Close the file
    call ncwrap( nf90_close(ncid) , __LINE__ )

    !Increment the number of outputs
    num_out = num_out + 1

    !Deallocate the temp arrays
    deallocate(dens )
    deallocate(uwnd )
    deallocate(wwnd )
    deallocate(theta)
  end subroutine output

  !Error reporting routine for the NetCDF I/O
  subroutine ncwrap( ierr , line )
    implicit none
    integer, intent(in) :: ierr
    integer, intent(in) :: line
    if (ierr /= nf90_noerr) then
      write(*,*) 'NetCDF Error at line: ', line
      write(*,*) nf90_strerror(ierr)
      stop
    endif
  end subroutine ncwrap
end program miniweather
