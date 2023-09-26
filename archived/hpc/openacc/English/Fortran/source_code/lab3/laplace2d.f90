! Copyright (c) 2012, NVIDIA CORPORATION. All rights reserved.
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions
! are met:
!  * Redistributions of source code must retain the above copyright
!    notice, this list of conditions and the following disclaimer.
!  * Redistributions in binary form must reproduce the above copyright
!    notice, this list of conditions and the following disclaimer in the
!    documentation and/or other materials provided with the distribution.
!  * Neither the name of NVIDIA CORPORATION nor the names of its
!    contributors may be used to endorse or promote products derived
!    from this software without specific prior written permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
! EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
! PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
! CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
! EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
! PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
! PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
! OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
! (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
! OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

module laplace2d
  public :: initialize
  public :: calcNext
  public :: swap
  public :: dealloc
  contains
    subroutine initialize(A, Anew, m, n)
      integer, parameter :: fp_kind=kind(1.0d0)
      real(fp_kind),allocatable,intent(out)   :: A(:,:)
      real(fp_kind),allocatable,intent(out)   :: Anew(:,:)
      integer,intent(in)          :: m, n

      allocate ( A(0:n-1,0:m-1), Anew(0:n-1,0:m-1) )

      A    = 0.0_fp_kind
      Anew = 0.0_fp_kind

      A(0,:)    = 1.0_fp_kind
      Anew(0,:) = 1.0_fp_kind
    end subroutine initialize

    function calcNext(A, Anew, m, n)
      integer, parameter          :: fp_kind=kind(1.0d0)
      real(fp_kind),intent(inout) :: A(0:n-1,0:m-1)
      real(fp_kind),intent(inout) :: Anew(0:n-1,0:m-1)
      integer,intent(in)          :: m, n
      integer                     :: i, j
      real(fp_kind)               :: error

      error=0.0_fp_kind

      !$acc parallel loop reduction(max:error) copyin(A) copyout(Anew)
      do j=1,m-2
        do i=1,n-2
          Anew(i,j) = 0.25_fp_kind * ( A(i+1,j  ) + A(i-1,j  ) + &
                                       A(i  ,j-1) + A(i  ,j+1) )
          error = max( error, abs(Anew(i,j)-A(i,j)) )
        end do
      end do
      calcNext = error
    end function calcNext

    subroutine swap(A, Anew, m, n)
      integer, parameter        :: fp_kind=kind(1.0d0)
      real(fp_kind),intent(out) :: A(0:n-1,0:m-1)
      real(fp_kind),intent(in)  :: Anew(0:n-1,0:m-1)
      integer,intent(in)        :: m, n
      integer                   :: i, j

      !$acc parallel loop copyin(Anew) copyout(A)
      do j=1,m-2
        do i=1,n-2
          A(i,j) = Anew(i,j)
        end do
      end do
    end subroutine swap

    subroutine dealloc(A, Anew)
      integer, parameter :: fp_kind=kind(1.0d0)
      real(fp_kind),allocatable,intent(in) :: A
      real(fp_kind),allocatable,intent(in) :: Anew
	  
	  deallocate (A,Anew)
    end subroutine
end module laplace2d
