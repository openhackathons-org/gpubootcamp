
module readdata
      contains
      
        subroutine readdcd(maxframes,maxatoms,x,y,z,xbox,ybox,zbox,natoms,nframes)
          use cudafor
          implicit none
          integer i,j
          integer maxframes,maxatoms

          double precision d(6),xbox,ybox,zbox
          real*4, managed, allocatable   :: x(:,:)
          real*4, managed, allocatable   :: y(:,:)
          real*4, managed, allocatable   :: z(:,:)

          real*4 dummyr
          integer*4 nset, natoms, dummyi,nframes,tframes
          character*4 dummyc
      
          open(10,file='../input/alk.traj.dcd',status='old',form='unformatted')
          read(10) dummyc, tframes,(dummyi,i=1,8),dummyr, (dummyi,i=1,9)
          read(10) dummyi, dummyr,dummyr
          read(10) natoms
          print*,"Total number of frames and atoms are",tframes,natoms

          allocate ( x(maxframes,natoms) )
          allocate ( y(maxframes,natoms) )
          allocate ( z(maxframes,natoms) )

          do i = 1,nframes
              read(10) (d(j),j=1, 6)
              
              read(10) (x(i,j),j=1,natoms)
              read(10) (y(i,j),j=1,natoms)
              read(10) (z(i,j),j=1,natoms)
          end do
          xbox=d(1)
          ybox=d(3)
          zbox=d(6)
          print*,"File reading is done: xbox,ybox,zbox",xbox,ybox,zbox
          return

        end subroutine readdcd

        attributes(global) subroutine pair_calculation( x,y,z,g,natoms,nframes,xbox,ybox,zbox,del,cut)
          use cudafor
          implicit none
          real*4     :: x(:,:)
          real*4     :: y(:,:)
          real*4    :: z(:,:)
          double precision,intent(inout)    :: g(:)
          integer, value :: nframes,natoms,ind
          double precision, value :: xbox,ybox,zbox,del,cut
          integer i,j,iconf
          double precision dx,dy,dz,r,oldvalue

          i = (blockIdx%x-1)*blockDim%x+threadIdx%x
          j = (blockIdx%y-1)*blockDim%y+threadIdx%y

          do iconf=1,nframes
           if(i<=natoms .and. j<=natoms) then

             dx=x(iconf,i)-x(iconf,j)
             dy=y(iconf,i)-y(iconf,j)
             dz=z(iconf,i)-z(iconf,j)

             dx=dx-nint(dx/xbox)*xbox
             dy=dy-nint(dy/ybox)*ybox
             dz=dz-nint(dz/zbox)*zbox

             r=dsqrt(dx**2+dy**2+dz**2)
             ind=int(r/del)+1
             if(r<cut)then
               oldvalue = atomicadd(g(ind),1.0d0)
             endif
           endif
          end do
          return
      end subroutine pair_calculation

end module readdata

program rdf
      use readdata
      use cudafor
      implicit none
      integer n,i,j,iconf,ind,istat
      integer natoms,nframes,nbin
      integer maxframes,maxatoms
      type(dim3) :: blocks, threads
      parameter (maxframes=10,maxatoms=60000,nbin=2000)
      real*4, managed, allocatable   :: x(:,:)
      real*4, managed, allocatable   :: y(:,:)
      real*4, managed, allocatable   :: z(:,:)
      double precision dx,dy,dz
      double precision xbox,ybox,zbox,cut
      double precision vol,r,del,s2,s2bond
      double precision, managed, allocatable   ::  g(:)
      double precision rho,gr,lngr,lngrbond,pi,const,nideal,rf
      double precision rlower,rupper
      character  atmnm*4
      real*4 start,finish
        
      open(23,file='RDF.dat',status='unknown')
      open(24,file='Pair_entropy.dat',status='unknown')

      nframes=10
         
      call cpu_time(start)

      print*,"Going to read coordinates"
      call readdcd(maxframes,maxatoms,x,y,z,xbox,ybox,zbox,natoms,nframes)
      allocate ( g(nbin) )
      do i=1,nbin 
        g(i) = 0.0d0
      end do
 
      pi=dacos(-1.0d0)
      vol=xbox*ybox*zbox
      rho=dble(natoms)/vol

      del=xbox/dble(2.0*nbin)
      write(*,*) "bin width is : ",del
      cut = dble(xbox * 0.5);
      threads = dim3(16,16,1)
      blocks = dim3(ceiling(real(natoms)/threads%x), ceiling(real(natoms)/threads%y),1)       
      print *, "natoms:, Grid Size:", threads, blocks 

       call pair_calculation<<<blocks,threads>>>(x,y,z,g,natoms,nframes,xbox,ybox,zbox,del,cut)

      istat = cudaDeviceSynchronize()
      if(istat /= 0) then
        print *, "Error"
      end if

       s2=0.01d0
       s2bond=0.01d0 
       const=(4.0d0/3.0d0)*pi*rho

       do i=1,nbin
          rlower=dble((i-1)*del)
          rupper=rlower+del
          nideal=const*(rupper**3-rlower**3)
          g(i)=g(i)/(dble(nframes)*dble(natoms)*nideal)
          r=dble(i)*del
          if (r.lt.2.0) then
            gr=0.0
          else
            gr=g(i)
          endif

          if (gr.lt.1e-5) then
            lngr=0.0
          else
            lngr=dlog(gr)
          endif
          if (g(i).lt.1e-6) then
            lngrbond=0.01
          else
            lngrbond=dlog(g(i))
          endif

          s2=s2-2*pi*rho*((gr*lngr)-gr+1)*del*r**2.0
          s2bond=s2bond-2*pi*rho*((g(i)*lngrbond)-g(i)+1)*del*r*r

          
          rf=dble(i-.5)*del
          write(23,*) rf,g(i)
       enddo
          write(24,*)"s2      : ",s2
          write(24,*)"s2bond  : ",s2bond
      call cpu_time(finish)
       print*,"starting at time",start,"and ending at",finish
      stop
      deallocate(x,y,z,g)
end
