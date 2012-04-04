#!/usr/bin/env python
"""
Reduced Gravity Shallow Water Model
based Matlab code by: Francois Primeau UC Irvine 2011

Kelsey Jordahl
kjordahl@enthought.com
Time-stamp: <Wed Apr  4 07:36:54 EDT 2012>
"""

import time
import numpy as np
from numpy import pi, sin, cos, sqrt, exp
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt
from IPython.frontend.terminal.embed import InteractiveShellEmbed

# Parameters
nx = 101; # number of grid points in the x-direction
ny = 101; # number of grid points in the y-direction
Lx = 2000e3; #(m) East-West domain size
Ly = 2000e3; #(m) North-South domain size
Omega = 2*pi/(24*60**2); # (1/s) rotational frequency of the Earth
phi0 = pi*30/180     # (rad) reference latitude
f0 = 2*Omega*sin(phi0); # (1/s) Coriolis parameter
a = 6400e3;             # (m) Earth's radius
beta = 0*(2*Omega/a)*cos(phi0); # (1/(ms))
Rd = 100e3;  # (m)  Rossby Radius 
H = 600;     # (m)  reference thickness
gp = (f0*Rd)**2/H; # (m/s^2) reduced gravity
Ah = 1e4;    # (m^2/s) viscosity
cg = sqrt(gp*H);

def d0(M):
    m = M.flatten()
    n = len(m);
    return sparse.spdiags(m,0,n,n);

def main():
    shell = InteractiveShellEmbed()
    # Mesh
    dx = Lx/nx
    dy = Ly/ny
    # mesh for the h-points
    xh = np.arange(dx/2, Lx, dx)
    yh = np.arange(dy/2, Ly, dy)
    Xh, Yh = np.meshgrid(xh,yh)
    # mesh for the u-points
    xu = np.arange(dx, Lx + dx, dx)
    yu = yh
    Xu, Yu = np.meshgrid(xu,yu)
    # mesh for the v-points
    xv = xh
    yv = np.arange(dy, Ly + dy, dy)
    Xv, Yv = np.meshgrid(xv,yv);

    # Land-sea mask defined on the h-points
    # 1 = ocean point
    # 0 = land point
    msk = np.ones((ny, nx))
    msk[:,-1] = 0
    msk[-1,:] = 0

    # Differential operators
    n = nx*ny
    I = sparse.eye(n, n).tocsc()
    ii = np.arange(n).reshape(nx, ny, order='F')
    ie = np.roll(ii, -1, 1)
    iw = np.roll(ii, 1, 1)
    iin = np.roll(ii, -1, 0)            # "in" is a reserved word
    iis = np.roll(ii, 1, 0)             # so is "is"
    IE = I[ie.flatten('F'),:10201]
    IW = I[iw.flatten('F'),:10201]
    IN = I[iin.flatten('F'),:10201]
    IS = I[iis.flatten('F'),:10201]

    DX = (1/dx)*(IE-I)
    DY = (1/dy)*(IN-I)
    GRAD = sparse.hstack([DX, DY])

    DIV = (1/(dx*dy))*sparse.hstack([I*dy-IW*dy, I*dx-IS*dx])
    hDIV = (1/(dx*dy))*sparse.hstack([dy*(I-IW)*d0(msk)*d0(IE*msk.flatten()),dx*(I-IS)*d0(msk)*d0(IN*msk.flatten())])
    # GRAD for the case of no slip boundary conditions
    # DEL2 for the v points
    # GRAD that assumes that v is zero on the boundary
    DX0 = d0(msk)*d0(IE*msk.flatten())*DX+d0(msk)*d0(1-IE*msk.flatten())*((1/dx)*(-2*I))+d0(1-msk)*d0(IE*msk.flatten())*((1/dx)*(2*IE))
    DY0 = DY
    GRADv = sparse.vstack([DX0, DY0])
    DEL2v = DIV*GRADv
    # DEL2 for the u ponts
    # GRAD that assumes that u is zero on the boundary
    DX0 = DX
    DY0 = d0(msk)*d0(IN*msk.flatten())*DY+d0(msk)*d0(1-IN*msk.flatten())*((1/dy)*(-2*I))+d0(1-msk)*d0(IN*msk.flatten())*((1/dy)*(2*IN))
    GRADu = sparse.vstack([DX0, DY0])
    DEL2u = DIV*GRADu
    # Averging operators that zero out the velocities through the boundaries
    uAv = 0.25*(I+IE+IS+IS*IE)*d0(msk)*d0(IN*msk.flatten())
    vAu = 0.25*(I+IN+IW+IN*IW)*d0(msk)*d0(IE*msk.flatten())

    # State vector
    # s = [u(:);v(:);h(:)];
    fu = f0+beta*Yu
    fv = f0+beta*Yv

    # Linear swm operator
    # L = np.array([-Ah*DEL2u, -d0(fu)*uAv,      gp*DX],
    #              [ d0(fv)*vAu, -Ah*DEL2v,      gp*DY],
    #              [       H*hDIV      , zeros(n,n)])
    L = sparse.vstack([sparse.hstack([-Ah*DEL2u, -d0(fu)*uAv, gp*DX]),
                       sparse.hstack([d0(fv)*vAu, -Ah*DEL2v, gp*DY]),
                       sparse.hstack([H*hDIV, sparse.csc_matrix((n,n))])]).tocsc()

    # Geostrophic adjustment problem
    # initial condition
    h0 = 10*exp(-((Xh-Lx/2)**2+(Yh-Ly/2)**2)/(Rd)**2)
    u0 = 0*Xv
    v0 = 0*Yv
    s = np.hstack([u0.flatten(), v0.flatten(), h0.flatten()])

    #ukeep = msk
    ukeep = msk.flatten()*IE*msk.flatten() # keep only pnts where u not 0 
        #vkeep = msk;
    vkeep = msk.flatten()*IN*msk.flatten() # keep only pnts where v not 0
    hkeep = msk.flatten()
    keep = np.hstack([ukeep, vkeep, hkeep])
    ikeep = np.nonzero(keep)[0]

    sbig = s
    s = s[np.nonzero(keep)]
    # indices of ocean points in the 2-d fields
    ih = np.nonzero(hkeep)
    iu = np.nonzero(ukeep)
    iv = np.nonzero(vkeep)
    # indices of variables inside the big s vector
    iubig = np.nonzero(np.hstack([ukeep, np.zeros(vkeep.shape), np.zeros(hkeep.shape)]))
    ivbig = np.nonzero(np.hstack([np.zeros(ukeep.shape), vkeep, np.zeros(hkeep.shape)]))
    ihbig = np.nonzero(np.hstack([np.zeros(ukeep.shape), np.zeros(vkeep.shape), hkeep]))

    # Crank-Nicholson time-stepping scheme
    # s(n+1)-s(n) + (dt/2)*L*(s(n)+s(n+1)) = 0
    # [I+(dt/2)*L]*s(n+1) = [I-(dt/2)*L]*s(n)
    # A*s(n+1) = B*s(n);
    # s(n+1) = A\(B*s(n));
    dt = 0.5*dx/cg
    I = sparse.eye(3*n, 3*n).tocsc()
    A = I + (dt/2)*L
    B = I - (dt/2)*L
    A = A[ikeep,:]
    A = A[:,ikeep]
    B = B[ikeep,:]
    B = B[:,ikeep]
    print 'Factoring the big matrix...',

    tic = time.time()
    solve = linalg.factorized(A)
    print 'Elapsed time: ', time.time() - tic

    h = np.zeros(msk.shape).flatten()
    u = np.zeros(msk.shape).flatten()
    v = np.zeros(msk.shape).flatten()

    for k in xrange(10000):
        #tic = time.time()
        s = solve(B*s)
        #print 'step time', time.time() - tic
        print '.',
        if k % 2 == 0: # make plot
            sbig[ikeep] = s
            u[iu] = sbig[iubig]
            v[iv] = sbig[ivbig]
            h[ih] = sbig[ihbig]
            V = v.reshape(msk.shape)
            U = u.reshape(msk.shape)
            Z = h.reshape(msk.shape)
            if k == 0:
                plt.subplot(211)
                p1 = plt.imshow(h.reshape(msk.shape))
                plt.subplot(212)
                p2 = plt.plot(xu,V[51,:]*200,'r')[0]
                p3 = plt.plot(xu,Z[51,:],'g')[0]
                plt.ylim(-10, 10)
                plt.show(block=False)
            else:
                #shell()
                p1.set_data(h.reshape(msk.shape))
                p2.set_data(xu, V[51,:]*200)
                p3.set_data(xu, Z[51,:])
            plt.pause(0.01)


        # original matlab plot commands follow
        #figure(1)
        #hold on
        #contour(Xh,Yh,h,[0 0],'k'); colorbar
        #caxis([-3,3]);
        #shading flat;
        #axis image;
        #drawnow
        #hold off
        #figure(2)
        #plotyy(Xh(51,:),h(51,:),Xu(51,:),v(51,:));
        #drawnow

if __name__ == '__main__':
    main()
