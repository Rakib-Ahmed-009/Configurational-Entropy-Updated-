import autograd.numpy as np
from autograd import grad, elementwise_grad
import autograd.numpy.random as npr
from autograd.misc.optimizers import adam
import pandas as pd

a=0.4
def init_random_params(scale, layer_sizes, rs=npr.RandomState(42)):
    """Build a list of (weights, biases) tuples, one for each layer."""
    return [(rs.randn(insize, outsize) * scale,   # weight matrix
             rs.randn(outsize) * scale)           # bias vector
            for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]

def swish(x):
    "see https://arxiv.org/pdf/1710.05941.pdf"
    return (x+1)/(1.0+np.exp(-x-1))

def psi(nnparams, inputs):
    "Neural network wavefunction"
    for W, b in nnparams:
        outputs = np.dot(inputs, W) + b
        inputs = swish(outputs)    
    return outputs

psip = elementwise_grad(psi, 1) # dpsi/dx 
psipp = elementwise_grad(psip, 1) # d^2psi/dx^2
nnparams = init_random_params(0.1, layer_sizes=[1, 8, 1])

params = {'nn': nnparams, 'factor': 0.0}

x = np.linspace(0, 20, 500)[:, None]

def objective(params, step):
    nnparams = params['nn']
    E = params['factor']        
    # This is Profile Function
    zeq = ((x)**2+a)*psipp(nnparams,x)+ 2*(x+a)*psip(nnparams,x) - np.sin(2*psi(nnparams,x)) 
    bc0 = psi(nnparams, 0.0)-3.14 # This approximates the first boundary condition at the initial boundary point
    bc1 = psi(nnparams, 20.0) -0.000182 # This approximates the second boundary condition at the edge boundary point
    y2 = psi(nnparams, x)**2
    # This is a numerical trapezoid integration
    prob = np.sum((y2[1:] + y2[0:-1]) / 2 * (x[1:] - x[0:-1]))
    return np.mean(zeq**2) + bc0**2 + bc1**2

# This gives us feedback from the optimizer
def callback(params, step, g):
    if step % 1000 == 0:
        print("Iteration {0:3d} objective {1}".format(step,
                                                      objective(params, step)))
params = adam(grad(objective), params,
              step_size=0.095, num_iters=30000, callback=callback)
import matplotlib.pyplot as plt

x = np.linspace(0, 20,500)[:, None]
y = psi(params['nn'], x)
z = psip(params['nn'], x)
fpi=93
e=1
a=0.4
ed=[0 for element in range(500)]
ed1=[0 for element in range(500)]
ed2=[0 for element in range(500)]
ed3=[0 for element in range(500)]
ed4=[0 for element in range(500)]
for i in range(500):
    ed[i]=fpi**2*fpi**2*e**2*0.5*(2*np.sin(y[i])**2/(x[i]**2+a)+z[i]**2)
    ed1[i]=fpi**2*fpi**2*2**2*0.5*(2*np.sin(y[i])**2/(x[i]**2+a)+z[i]**2)
    ed2[i]=fpi**2*fpi**2*3**2*0.5*(2*np.sin(y[i])**2/(x[i]**2+a)+z[i]**2)
    ed3[i]=fpi**2*fpi**2*4**2*0.5*(2*np.sin(y[i])**2/(x[i]**2+a)+z[i]**2)
    ed4[i]=fpi**2*fpi**2*5**2*0.5*(2*np.sin(y[i])**2/(x[i]**2+a)+z[i]**2)



y_solve = 4*np.arctan(np.exp(-x/2))
y_der = -2*np.exp(-x/2)/(1+np.exp(-x))

ed5=[0 for element in range(500)]
for i in range(500):
    ed5[i]=fpi/e*0.5*((2*np.sin(y_solve[i])**2+x[i]**2*y_der[i]**2)+np.sin(y_solve[i])**2*(np.sin(y_solve[i])**2/(x[i]**2+a)+2*y_der[i]**2))

summation = [complex(0,0) for i in range(500)]
y_k = [complex(0,0) for i in range(500)]
for i in range(500):
   for k in range(500):
       summation[i]+= 4*np.pi*(k**2)*np.exp(-2*np.pi*1j*i*k/500)*ed[k]
       y_k[i]=summation[i]
       
y_mod = np.absolute(y_k)
ymods = y_mod**2
ymodsum = np.sum(ymods)
modfrac = ymods/ymodsum
modftud = modfrac/np.max(modfrac)
Df = -modftud*np.log(modftud)
CE = np.sum(Df)
print(CE)
col1="F(r)"

plt.plot(x,ed,color='r',label='e=1')
plt.plot(x,ed1,color='g',label='e=2')
plt.plot(x,ed2,color='b',label='e=3')
plt.plot(x,ed3,color='k',label='e=4')
plt.plot(x,ed4,color='c',label='e=5')

plt.xlabel('Dimensionless Coordinate, r')
plt.ylabel('Normalized Modal Fraction, $\\tilde{f}(k)$')
plt.legend()
plt.show()

