c=========================================================================
c User material definition based of ML flow rule
c
c Version 1.0.0 (2021-11-27)
c
c Authors: Alexander Hartmaier, Anderson Wallace Paiva do Nascimento
c Email: alexander.hartmaier@rub.de
c ICAMS / Ruhr University Bochum, Germany
c November 2021
c
c distributed under GNU General Public License (GPLv3)
c==========================================================================
      subroutine umat(stress, statev, ddsdde, sse, spd, scd, rpl, 
     &                ddsddt, drplde, drpldt, stran, dstran, 
     &                time, dtime, temp, dtemp, 
     &                predef, dpred, cmname, ndi, nshr, ntens, 
     &                nstatv, props, nprops,
     &                coords, drot, pnewdt, celent, 
     &                dfgrd0, dfgrd1, noel, npt,
     &                layer, kspt, kstep, kinc)
c !========================================================================
c ! variables available in UMAT: 
c !------------------------------
c ! stress, stran, statev, props
c ! dstran, drot, dfgrd0, dfgrd1, time, dtime, temp, dtemp, predef, dpred
c ! coords, celent
c ! number of element, integration point, current step and increment
c !--------------------------------
c ! variables that must be defined:
c !--------------------------------
c ! stress, statev, ddsdde
c !========================================================================
c ! Material properties (props):
c ! 1: nsv, number of support vectors
c ! 2: nsd, dimensionality of support vetcors
c ! 3: C11
c ! 4: C12
c ! 5: C44
c ! 6: rho (SVM parameter)
c ! 7: lambda (SVM parameter)
c ! 8: ep_c (critical value of plastic strain defining onset of yielding in material data)
c ! 9: scale_seq (scaling factor for equiv. stresses)
c ! 10: scale_wh (scaling factor for plastic strain  /w.h. parameter)
c ! 11: C22
c ! 12: C33
c ! 13: C13
c ! 14: C23
c ! 15: C55
c ! 16: C66
c ! 17: Nset (number of sets, if multiple textures are cpntained in SVM)
c ! 18..29: scale_text (scaling factors for texture parameters)
c ! 30..nsv+29: dual coefficients
c ! nsv+30..+5*nsv+29: support vectors (5-dimensional)
c !========================================================================
c ! State variables
c ! 1-6: plastic strain tensor (eplas, components 11, 22, 33, 12, 13, 23)
c ! 7  : equivalent plastic strain (PEEQ)
 
      implicit none
 
      character(len=80) :: cmname                                                 !User defined material name
      integer :: ndi                                                              !Number of direct stress components
      integer :: nshr                                                             !Number of engineering shear stress components
      integer :: ntens                                                            !Size of the stress/strain array (ndi + nshr)
      integer :: nstatv                                                           !Number state variables
      integer :: nprops                                                           !Number of user defined material constants
      integer :: layer                                                            !Layer number
      integer :: kspt                                                             !Section point number within the current layer
      integer :: kstep                                                            !Step number
      integer :: noel                                                             !Element number
      integer :: npt                                                              !Integration point number
      integer :: kinc                                                             !Increment number

      real(8) :: drpldt                                                           !Variation of rpl w.r.t. the temperature
      real(8) :: dtime                                                            !Time increment
      real(8) :: temp                                                             !Temperature at the start of the increment
      real(8) :: dtemp                                                            !Increment of temperature.
      real(8) :: celent                                                           !Characteristic element length
      real(8) :: sse                                                              !Specific elastic strain energy
      real(8) :: spd                                                              !Specific plastic dissipation
      real(8) :: scd                                                              !Specific creep dissipation
      real(8) :: rpl                                                              !volumetric heat generation per unit time
      real(8) :: pnewdt                                                           !Ratio dtime_next/dtime_now

      real(8) :: stress(ntens), dsig(ntens)                                       !Stress tensor at start of increment, stress increment
      real(8) :: ddsdde(ntens,ntens)                                              !Jacobian matrix of the constitutive model
      real(8) :: ddsddt(ntens)                                                    !Variation of the stress increment w.r.t. to the temperature
      real(8) :: drplde(ntens)                                                    !Variation of rpl w.r.t. the strain increment
      real(8) :: stran (ntens)                                                    !Total strain tensor at the beggining of the increment
      real(8) :: dstran(ntens)                                                    !Strain increment
      real(8) :: eplas(ntens)                                                     !plastic strain

      real(8) :: statev(nstatv)                                                   !Solution dependent state variables
      real(8) :: props(nprops)                                                    !User specified material constants 
      real(8) :: dfgrd0(3,3)                                                      !Deformation gradient at the beggining of the increment
      real(8) :: dfgrd1(3,3)                                                      !Deformation gradient at the end of the increment
      real(8) :: drot(3,3)                                                        !Rotation increment matrix
      real(8) :: coords(3)                                                        !Coordinates of the material point
      real(8) :: time(2)                                                          !1: Step time; 2:Total time; Both at the beggining of the inc
      real(8) :: predef(1)                                                        !Predefined field variables at the beggining of the increment
      real(8) :: dpred (1)                                                        !Increment of predefined field variables

      real(8) :: C11, C22, C33, C44, C55, C66, C12, C13, C23                      !anisotropic elastic constants
      real(8) :: epc, khard                                                       !offset in plastic strain (definition of yield point), WH parameter
      integer :: nsv, nsd                                                         !Number and dimensionality of support vectors
      integer :: Nset 
      real(8), dimension(:, :), allocatable :: temp_sv                            !Array of support vectors
      real(8), dimension(:), allocatable :: temp_dc                               !Array of dual coefficients
      real(8), dimension(ntens) :: sigma                                          !Stress tensor
      real(8), dimension(ntens) :: stress_fl                                      !flow stress on yield locus
      real(8) :: rho                                                              !Intercept of the decision rule function
      real(8) :: lambda                                                           !Regularization parameter
      real(8) :: scale_seq, scale_wh                                              !scaling factors for stress, work hardening rate
      real(8) :: fsvc                                                             !Decision function
      real(8) :: sq0, sq1, sq2                                                    !equiv. stresses
      real(8), dimension(ntens) :: dfds                                           !Derivative of the decision function w.r.t. princ. stress
      real(8), dimension(ntens) :: flow                                           !Flow vector

      real(8), parameter :: tol = 1.e-2                                           !Rel. tolerance for for yield function
      
      integer :: i, j, k, fileUnit, niter, ind_sv0, ind_dc0                       !Auxiliar indices
      real(8), dimension(ntens, ntens) :: Ct                                      !Consistent tanget stiffness matrix
      real(8), dimension(ntens) :: deps                                           !strain increment outside yield locus, if load step is splitted
      real(8) :: threshold, h1, h2, eeq, peeq, sc_elstep

      nsv = int(props(1))
      nsd = int(props(2))
      C11 = props(3)
      C12 = props(4)
      C44 = props(5)
      rho = props(6)
      lambda = props(7)
      epc = props(8)
      scale_seq = props(9)
      scale_wh  = props(10)
      C22 = props(11)
      C33 = props(12)
      C13 = props(13)
      C23 = props(14)
      C55 = props(15)
      C66 = props(16)
      Nset = int(props(17))
      ind_dc0 = 30
      ind_sv0 = 30+nsv

      threshold = tol*scale_seq   ! threshold value for yield function still accepted as elastic

      ! internally the standard convention for Voigt tensors is used
      h1 = stress(6)
      stress(6) = stress(4)
      stress(4) = h1 ! convert to standard convention
      h1 = dstran(6)
      dstran(6) = dstran(4)
      dstran(4) = h1
      
      ! get accumulated plastic strain
      eplas(1:ndi) = statev(1:ndi)
      eplas(4) = statev(6)
      eplas(5) = statev(5)
      eplas(6) = statev(4)   ! store in standard convention for Voigt strain
      call calcEqStrain(eplas, peeq) ! peeq: equivalent plastic strain
      khard = 0.  ! work hardening parameter, updated in calcGradFSVC


      ! Elastic stiffness matrix is defined
      ddsdde = 0.d0
      sc_elstep = 0.d0  
      ddsdde(1,1) = C11
      ddsdde(1,2) = C12
      ddsdde(2,1) = C12
      ddsdde(4,4) = C44
      if (C22<0.) then
        ! cubic symmetry in elasticity
        ddsdde(2,2) = C11
        ddsdde(3,3) = C11
        ddsdde(5,5) = C44
        ddsdde(6,6) = C44
        ddsdde(1,3) = C12
        ddsdde(3,1) = C12
        ddsdde(2,3) = C12
        ddsdde(3,2) = C12
      else
        ! full orthotropic elasticity
        ddsdde(2,2) = C22
        ddsdde(3,3) = C33
        ddsdde(5,5) = C55
        ddsdde(6,6) = C66
        ddsdde(1,3) = C13
        ddsdde(3,1) = C13
        ddsdde(2,3) = C23
        ddsdde(3,2) = C23
      end if          
      Ct = ddsdde

      ! elastic predictor for stress is computed
      deps = dstran
      dsig = 0.d0
      do i=1, ntens
        do j=1, ntens
            dsig(i)=dsig(i)+ddsdde(i,j)*dstran(j)
        end do
      end do

      ! calculate yield function by evaluating support vector classification
      sigma = stress+dsig
      call calcFSVC(sigma, fsvc)
      niter = 0
      do while ((fsvc.ge.threshold).and.(niter<10))
         ! stress lies outside yield locus
         ! 1. calculate proper flow stress on yield locus
         call findRoot(sigma, stress_fl)
         if (niter.eq.0) then
            ! in first iteration: test if part of load step is elastic
            call calcFSVC(stress, h1) ! yield function at start of increment
            if (h1.lt.-tol) then
                !load step started in elastic regime and has to be splited
                !for load reversals, negative values must be treated separately
                !perform elastic substep
                call calcEqStress(stress, sq0) ! equiv. stress at start of load step
                call calcEqStress(sigma, sq2)  ! equiv stress at end of load step
                call calcEqStress(stress_fl, sq1) ! eqiv. stress on yield locus
                sc_elstep = (sq1-sq0)/(sq2-sq0) ! split load step in elastic regime
                deps = dstran*sc_elstep ! elastic strain increment   
                stran = stran + deps    ! add to accomplished strain
                deps = dstran - deps    ! deduct from remaining strain
                stress = stress_fl      ! stress at start of remaining load increment (on yield locus)
            end if !h1<-tol
         end if ! niter==0
         ! 2. calculate gradient 'dfds' of yield locus for given stress tensor
         ! also updates khard based on current strain hardening rate
         call calcGradFSVC(stress_fl, dfds)
         ! 3. calculate plastic strain increment 'flow'
         call calcFlow(dfds, deps, Ct, eplas, flow)
         ! 4. calculate consistent tangent stiffness tensor 'Ct'
         call calcTangstiff(ddsdde, dfds, Ct)
         ! 5. calculate consistent stress increment 'dsig'
         dsig = 0.
         do i=1, ntens
             do j=1, ntens
                 dsig(i)=dsig(i)+Ct(i,j)*deps(j)
             end do
         end do
         ! update stress for next iteration and
         ! calculate yield function
         sigma = stress + dsig
         call calcFSVC(sigma, fsvc)
         niter = niter + 1 
      end do
      if (niter.ge.10) then
        print*,'***Warning: plasticity algorithm did not converge after',
     &         niter,'iterations'
      end if
  
      !update stresses and strain
      stran = stran + deps
      h1 = stran(6)
      stran(6) = stran(4)
      stran(4) = h1  ! convert to Abaqus convention
      h1 = dstran(6)
      dstran(6) = dstran(4)
      dstran(4) = h1
      stress = stress + dsig
      h1 = stress(6)
      stress(6) = stress(4)
      stress(4) = h1  ! convert to Abaqus convention
      eplas = eplas + flow 
      !update internal variables
      statev(1:3) = eplas(1:3)
      statev(4) = eplas(6)   ! store in Abaqus convention
      statev(5) = eplas(5)
      statev(6) = eplas(4)
      call calcEqStrain(eplas, peeq)
      statev(7) = peeq
      
      !update material Jacobian
      ddsdde = ddsdde*sc_elstep + Ct*(1.d0-sc_elstep)
      ! exchange column 6 and 4 and row 6 and 4 in stiffness tensor to meet Abaqus convention
      dfds = ddsdde(:,6)
      ddsdde(:,6) = ddsdde(:,4)
      ddsdde(:,4) = dfds
      dfds = ddsdde(6,:)
      ddsdde(6,:) = ddsdde(4,:)
      ddsdde(4,:) = dfds
      ! END main 

      return
      
      contains
 
      subroutine index(i,j,k)
        ! Calculate index of support vector in prop array
        implicit none
        integer :: i,j,k
  
        k = ind_sv0 + (i-1)*nsd + j-1
      end subroutine index

      subroutine calcHydStress(stress, sigmaHyd)
        !Caculate the hydrostatic stress component
        implicit none
        real(8), dimension(ntens) :: stress
        real(8) :: sigmaHyd

        sigmaHyd = (stress(1)+stress(2)+stress(3))/3.
      end subroutine calcHydStress

      subroutine calcDevStress(stress, sigmaDev)
        !Calculate the deviatoric stress tensor
        implicit none
        real(8), dimension(ntens) :: stress, sigmaDev
        real(8) :: sigmaHyd
        integer :: i

        call calcHydStress(stress, sigmaHyd)
        do i=1, ndi
            sigmaDev(i) = stress(i) - sigmaHyd
        end do
        do i=ndi+1, ntens
            sigmaDev(i) = stress(i)
        end do
      end subroutine calcDevStress

      subroutine calcEqStress(sig, seq)
        !Calculate the equivalent J2 stress
        real(8), dimension(ntens) :: sig
        real(8) :: seq
        real(8), dimension(ntens) :: sd
  
        call calcDevStress(sig, sd)
        seq = sqrt(0.5*((sd(1)-sd(2))**2 + (sd(2)-sd(3))**2 + 
     &   (sd(3)-sd(1))**2 + 6.*(sd(4)**2 + sd(5)**2 + sd(6)**2)))
      end subroutine calcEqStress

      subroutine calcEqStrain(eps, eeq)
        !Calculate the equivalent strain
        real(8), dimension(ntens) :: eps
        real(8) :: eeq
C       real(8), dimension(ntens) :: ed
C       real(8) :: ev, hdi, hsh
C       integer :: i
C 
C       ev = 0.d0
C       do i=1,ndi
C           ev = ev + eps(i)
C       end do
C       ed = eps
C       hdi = 0.d0
C       hsh = 0.d0
C       do i=1,ndi
C           hdi = hdi + (eps(i)-ev)**2
C           hsh = hsh + eps(ndi+i)**2
C       end do
C       eeq = sqrt((2.d0*hdi+hsh)/3.)
        eeq =  dsqrt(2.0*(eplas(1)**2 + eplas(2)**2 + eplas(3)**2
     &             + 2.0*(eplas(4)**2 + eplas(5)**2 + eplas(6)**2))
     &             / 3.0)
      end subroutine calcEqStrain

      subroutine calcKernelFunction(x, i_sv, kernelFunc)
      !Calculate the Radial Basis Function (RBF) kernel of the SVC
        implicit none
        real(8), dimension(nsd) :: x   ! scaled component of SVC feature vector
        real(8) :: kernelFunc, hh, hs, sv
        integer :: i, i_sv, k
  
        hh = 0.
        do i=1,nsd
            call index(i_sv, i, k)
            hs = x(i) - props(k)
            hh = hh + hs*hs
        end do
        kernelFunc = exp(-lambda*hh)
      end subroutine calcKernelFunction

      subroutine calcFSVC(sigma, fsvc)
      !Evaluate  decision function 'fsvc' at stress sigma
      !based on the trained Support Vector Classification (SVC)
        implicit none
        integer :: i
        real(8), dimension(ntens) :: sigma, sig_dev, gj2
        real(8), dimension(nsd) :: hs
        real(8) :: fsvc, kernelFunc

        fsvc = 0.
        do i=1, nsv
            call calcDevStress(sigma, sig_dev)
            hs(1:6) = sig_dev(1:6)/scale_seq
            if (nsd>6) then
                hs(7) = peeq/scale_wh - 1.
            end if
            call calcKernelFunction(hs, i, kernelFunc)
            fsvc = fsvc + props(ind_dc0+i-1)*kernelFunc
        end do
        fsvc = fsvc + rho
      end subroutine calcFSVC

      subroutine calcDK_DX(x, i_sv, dk_dx)
        !Calculate the derivative of the kernel basis function
        !with respect to the SVM feature vector 
        implicit none
        real(8), dimension(nsd) :: x, dk_dx
        real(8) :: kernelFunc
        integer :: i_sv, i, k

        call calcKernelFunction(x, i_sv, kernelFunc)
        do i=1,nsd
            call index(i_sv, i, k)
            dk_dx(i) = -2.*lambda*kernelFunc*(x(i) - props(k))
        end do
      end subroutine calcDK_DX

      subroutine calcGradFSVC(sigma, dfds)
        !Calculate the gradient of the decision function w.r.t. the stress
        !strain hardening rate khard is also updated based on gradient ov SVC w.r.t peeq
        implicit none
        integer :: i
        real(8), dimension(ntens) :: sigma, sig_dev, dfds, gj2
        real(8), dimension(nsd) :: hs, dk_dx, hg
        real(8) :: h0,h1,h2,h3,h4,h5,seq

        call calcDevStress(sigma, sig_dev)
        hs(1:6) = sig_dev(1:6)/scale_seq
        if (nsd>6) then
            hs(7) = peeq/scale_wh - 1.
        end if
        hg = 0.
        do i=1, nsv
            call calcDK_DX(hs, i, dk_dx)
            hg = hg + props(ind_dc0+i-1)*dk_dx
        end do
        dfds(1:6) = hg(1:6) / scale_seq
        if (nsd>6) then
            khard = -hg(7)*scale_seq/scale_wh
        end if
      end subroutine calcGradFSVC

      subroutine calcFlow(dfsvc, deps, Ct, eplas, flow)
        !Calculate the consistent plastic flow tensor
        implicit none
        real(8), dimension(ntens) :: dfsvc
        real(8), dimension(ntens) :: deps, eplas, flow
        real(8), dimension(ntens, ntens) :: Ct
        real(8) :: hh, l_dot
        integer :: i,j

        hh = 0.
        l_dot = 0.
        do i=1,ntens
            do j=1,ntens
                hh = hh + dfsvc(i)*Ct(i,j)*dfsvc(j)
            end do
        end do
        hh = hh + 4.*khard
        do i=1,ntens
            do j=1,ntens
                l_dot = l_dot + dfsvc(i) * Ct(i,j) * deps(j) / hh  ! deps must not contain elastic strain components
            end do
        end do
        flow = l_dot * dfsvc 
      end subroutine calcFlow

      subroutine calcTangStiff(Cel, dfds, Ct)
        !Calculate the elasto-plastic tangent stiffness matrix
        implicit none
        real(8), dimension(ntens, ntens) :: Cel, Ct
        real(8), dimension(ntens) :: dfds
        real(8), dimension(ntens) :: ca
        real(8) :: hh
        integer :: i, j

        hh = 0.d0
        ca = 0.d0
        Ct = 0.d0
        do i=1,ntens
            do j=1,ntens
                hh =  hh + dfds(i) * Cel(i,j) * dfds(j)
                ca(i) = ca(i) + Cel(i,j) * dfds(j)
            end do
        end do
        hh = hh + 4.*khard
        do i=1,ntens
            do j=1,ntens
                Ct(i,j) = Cel(i,j) - ca(i)*ca(j)/hh
            end do
        end do
      end subroutine calcTangStiff

      subroutine findRoot(sigma, s_fl)
        !Perform bisection method to find the root of the
        !yield function by proportional variations of the given stress sigma
        implicit none
        real(8), dimension(ntens) :: sigma, s_fl
  
        integer :: i, j
        real(8) :: fsvc, error
        real(8) :: lowerBound, upperBound, increment
        real(8) :: a, b, fsvca, fsvcb, root, fsvcAtroot, seq0
        real(8), dimension(ntens) :: sunit
        integer, parameter :: split = 10
        integer, parameter :: nmax = 100

        call calcFSVC(sigma, fsvca)
        if (fsvca.le.threshold) then
            return
        else
            !An initial broad interval is split into subintervals and a change
            !in the sign of fsvc is sought. The first subinterval meeting this 
            !criterion will be used for the bisection root finding procedure
            !It is assumed that the fSVC value at sigma is positive turning negative at 
            !smaller stresses
            call calcEqStress(sigma, seq0)
            sunit = sigma/seq0
            upperBound = seq0
            a = upperBound
            lowerBound = 0.9*seq0
            b = lowerBound
            increment = lowerBound/split
      
            s_fl = sunit*b
            call calcFSVC(s_fl, fsvcb)
            j = 1
            do while ((fsvca*fsvcb.gt.0.).and.(j.le.split))
                b = lowerBound - j*increment
                s_fl = sunit*b
                call calcFSVC(s_fl, fsvcb)
                j = j + 1
            end do 
            ! b is now the factor for the largest stress with negative yield function
            ! now look for smallest upper bracket a
            increment = (a-b)/split
            j = 1
            do while ((fsvca*fsvcb .lt. 0.).and.(j.lt.split))
                a = upperBound - j*increment
                s_fl = sunit*a
                call calcFSVC(s_fl, fsvca)
                j = j + 1
            end do 
            a = a + increment

            i = 1
            error = 2.*threshold
            do while ((i.lt.nmax).and.(error.ge.threshold))
                !Calculating fsvc at the bounds of the interval
                s_fl = sunit*a
                call calcFSVC(s_fl, fsvca)
                s_fl = sunit*b
                call calcFSVC(s_fl, fsvcb)
                !Checking if the root is bracketed within the interval
                if (fsvca*fsvcb .lt. 0.) then
                    root = (a+b)/2.
                    s_fl = sunit*root
                    call calcFSVC(s_fl, fsvcAtroot)
                    if (fsvca*fsvcAtroot.lt.0.) then
                        b = root
                    else
                        a = root
                    end if
                else
                    print*, "Root not bracketed properly"
                    print*, a, fsvca
                    print*, b, fsvcb
                    stop
                end if
                i = i + 1
                error = abs(fsvcAtroot)
            end do
            if (abs(fsvca).lt.error) then
                s_fl = sunit*a
            end if
            if (abs(fsvcb).lt.error) then
                s_fl = sunit*b
            end if
        end if 
      end subroutine findRoot

      end subroutine umat
    
    
    
