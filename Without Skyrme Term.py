import autograd.numpy as np
from autograd import grad, elementwise_grad
import autograd.numpy.random as npr
from autograd.misc.optimizers import adam
from matplotlib import pyplot as plt

m=10 ##### The parameter e in the theory of the Skyrme Model #####
####### Creating the Basic layers of the network, defining weight and bias matrices as a function of shapes #######
def init_random_params(scale, layer_sizes, rs=npr.RandomState(42)):
    """Build a list of (weights, biases) tuples, one for each layer."""
    return [(rs.randn(insize, outsize) * scale,   # weight matrix
             rs.randn(outsize) * scale)           # bias vector
            for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]

def swish(x):
    return (x)/(1.0+np.exp(-x))   ##### Defining the activation function #####

def psi(nnparams, inputs): ##### Defining the profile function as a parameter of the radial distance and neural network parameters #####
    for W, b in nnparams:
        outputs = np.dot(inputs, W) + b  ##### Combining weights and biases with inputs and hence connecting all the neurons as a linear combination #####
        inputs = swish(outputs)    
    return outputs

psip = elementwise_grad(psi, 1)  ##### dpsi/dx = 1st order derivative of the profile function ##### 
psipp = elementwise_grad(psip, 1)  ##### d^2psi/dx^2 = 2nd order derivative of the profile function #####
nnparams = init_random_params(0.1, layer_sizes=[1, 8, 1])   ##### Layer sizes #####

params = {'nn': nnparams, 'factor': 1}

x = np.linspace(0, 20, 500)[:, None] ##### Discretizing the radial distance #####

def objective(params, step):
    nnparams = params['nn']
    a = params['factor']        
    ##### zeq is the ODE for the profile function #####
    zeq = ((x)**2+a)*psipp(nnparams,x)+ 2*(x+a)*psip(nnparams,x) - np.sin(2*psi(nnparams,x)) 
    bc0 = psi(nnparams, 0.0)-3.14 ##### This approximates the first boundary condition at the initial boundary point #####
    bc1 = psi(nnparams, 20.0) -0.00 ##### This approximates the second boundary condition at the final boundary point #####
    return np.mean(zeq**2) + bc0**2 + bc1**2
	
##### This gives us feedback from the optimizer, the values of the cost function #####
def callback(params, step, g):
    if step % 1000 == 0:
        print("Iteration {0:3d} objective {1}".format(step,
                                                      objective(params, step))) ##### Adam optimizer is used to minimize the cost #####
params = adam(grad(objective), params,
              step_size=0.095, num_iters=29000, callback=callback)

##### neural network solution ends here #####
import matplotlib.pyplot as plt

x = np.linspace(0, 20,500)[:, None]
y = psi(params['nn'], x) ##### profile function #####
z = psip(params['nn'], x) ##### first order derivative #####
fpi=129 ##### pion decay constant #####

ed=[0 for elements in range(500)] ##### initial array for energy density #####

for i in range(498):
    ed[i+1]=(2*(np.sin(y[i+1]))**2+x[i+1]**2*z[i+1]**2)   ##### energy density without the Skyrme Term, except boundary points #####

def e(k,x):
	return np.exp(-k*1j*x/(m*fpi))  ##### the exponential factor for fourier transform #####

yk=[complex(0,0) for element in range(3000)] ##### Array initialization for storing fourier transform results #####
ymod=[0 for element in range(3000)] ##### array for storing mod square of the fourier transform  #####
ysum=0 ##### sum of all the ymods #####

for j in range(3000):
	yk[j]=(e(k[j],0)/2)*x[0]**2*z[0]**2+(e(k[j],20)/2)*x[499]**2*z[499]**2  ##### initializing the trapezoidal rule for boundary points, the first and last summand gets divided by two, so it is added separately here #####

for j in range(3000):
	for i in range(498):
		yk[j]+=e(k[j],x[i+1])*ed[i+1]  ##### implementing trapezoidal rule for the inner points, which are simply summed altogether #####
		
##### Fourier transform ends here #####
		
for j in range(3000):
	ymod[j] =np.absolute(yk[j])**2  ##### mod square of all the fourier transform values #####

ysum=ymod[0]/2+ymod[2999]/2 ##### initializing the trapezoidal rule for integrating all the mod squares, starting with the 1st and last point #####

for j in range(2998):
       ysum+=ymod[j+1] ##### mod square integration for inner points, which are simple sums according to the trapezoidal rule #####
       
modfrac = ymod/ysum ##### modal fraction, defined by dividing the modulus squared with the sum of all mod squared values #####
modftud = modfrac/np.max(modfrac) ##### normalized modal fraction, defined by dividing with the highest value of the modal fraction #####
Df = -modftud*np.log(modftud) ##### Differential configurational entropy #####
CE = (Df[0]/2)+(Df[2999]/2) ##### initializing trapezoidal rule for integrating DCE, for the initial and final points #####
for j in range(2998):
	CE+=Df[j+1] ##### DCE trapezoidal rule for all the inner points #####

print(CE) ##### Configurational entropy value #####
