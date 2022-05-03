#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 16:39:35 2022

@author: amtsdg
"""

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as pt, numpy as np

def phi(X):
    
    # shape function for slope, going from 0 to 1
    # as non-dim slope coordinate goes from 0 to 1
    
    #temp=X
    temp=np.sin(X*np.pi/2)**2
    
    return temp

def hss(x,pars):
    
    # depth of shelf/slope/abyss 
    
    h1=pars.h1
    h2=pars.h2
    s=pars.s
    
    temp=h1*(x<0)+(x>=0)*(x<=s)*(h1+(h2-h1)*phi(x/pars.s))+h2*(x>s)
    
    return temp

def hc(x,y,pars):
    
    delta=pars.delta
    h1=pars.h1
    h2=pars.h2
    L=pars.L
    W=pars.W
    s=pars.s

    # d: cross-canyon lengthscale used in formula!   
    d=W*np.sqrt( 1+h1*(1+delta*s/L)/( (h2-h1)*phi(delta) ) )
    
    temp=hv(x,pars)*(1-(y/d)**2)
    
    return temp

def hv(x,pars):
    
    # depth of valley (linear in x), at centre of canyon
    
    h1=pars.h1
    h2=pars.h2
    L=pars.L
    s=pars.s
    delta=pars.delta
    
    temp=h1+(h2-h1)*phi(delta)*(x+L)/(delta*s+L)
    
    return temp

class pars:
    
    s=1        # slope width
    L=.15       # canyon length ON SHELF
    W=0.25      # (half) canyon width AT SHELF BREAK
    delta=0.2 # 0<delta<1: canyon occupies this proportion of slope
    h1=1       # shelf depth
    h2=4       # open-ocean depth
 
nx=400
x=np.linspace(-pars.L-1,pars.s+1,nx+1)

pt.figure(1,figsize=[5,4])
pt.clf()
pt.plot(x,-hss(x,pars),'k')
pt.plot(x,-hv(x,pars),'r')
pt.ylim([-1.1*pars.h2,0])
pt.grid()

ny=300
y=np.linspace(-.5,.5,ny+1)

[xa,ya]=np.meshgrid(x,y)
hssa=hss(xa,pars)
hca=hc(xa,ya,pars)
# take maximum of hss (shelf/slope geometry) and hc (canyon geometry), 
# provided we are on the shelf or slope:
h=(xa>pars.s)*hssa+(xa<=pars.s)*( (hssa<hca)*hca+(hssa>=hca)*hssa )

fig=pt.figure(2,figsize=[8,5])
ax=fig.add_subplot(111,projection='3d')
ax.plot_surface(xa,ya,-h)
ax.view_init(40,-30)
pt.show()

pt.savefig('canyon.jpg',dpi=300)