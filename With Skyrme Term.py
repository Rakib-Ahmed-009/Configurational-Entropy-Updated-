import autograd.numpy as np
from autograd import grad, elementwise_grad
import autograd.numpy.random as npr
from autograd.misc.optimizers import adam

#######neural network implementation starts here
def init_random_params(scale, layer_sizes, rs=npr.RandomState(42)):
    """Build a list of (weights, biases) tuples, one for each layer."""
    return [(rs.randn(insize, outsize) * scale,   # weight matrix
             rs.randn(outsize) * scale)           # bias vector
            for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]

def swish(x):
    return (x+1) / (1.0 + np.exp(-x-1))

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
    zeq = psipp(nnparams,x)+((psip(nnparams,x)**2)*(np.sin(2*psi(nnparams,x))) + 2*(x)*psip(nnparams,x) - np.sin(2*psi(nnparams,x)) - (1/(x**2+1))*((np.sin(psi(nnparams,x)))**2)*(np.sin(2*psi(nnparams,x))))/((x**2+1)+2*((np.sin(psi(nnparams,x)))**2)) 
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
              step_size=0.095, num_iters=29000, callback=callback)
import matplotlib.pyplot as plt

#neural network solution ends here
x = np.linspace(0, 20, 500)[:, None]
y = psi(params['nn'], x) #profile function
z = psip(params['nn'], x) #first order derivative
fpi=129 #pion decay constant
m=10 #Skyrme parameter
k=np.linspace(0,1450,3000) #discretization of momentum space for a close approximation to continuous fourier transform 


ed=[0 for elements in range(500)]  #initial array for energy density

for i in range(498):
    ed[i+1]=(2*(np.sin(y[i+1]))**2+x[i+1]**2*z[i+1]**2)+ (np.sin(y[i+1]))**2*((np.sin(y[i+1])**2)/(x[i+1]**2)+2*z[i+1]**2)  #energy density for the skyrme model excluding the boundary points

def e(k,x):
	return np.exp(-k*1j*x/(m*fpi)) #the exponential factor for fourier transform

yk=[complex(0,0) for element in range(3000)] #Array initialization for storing fourier transform results
ymod=[0 for element in range(3000)] #mod square of the fourier transform
ysum=0 #sum of all the ymods

#Fourier transform starts here
for j in range(3000):
	yk[j]=(e(k[j],0)/2)*x[0]**2*z[0]**2+(e(k[j],20)/2)*x[499]**2*z[499]**2 #initializing the trapezoidal rule for boundary points

for j in range(3000):
	for i in range(498):
		yk[j]+=e(k[j],x[i+1])*ed[i+1] #implementing trapezoidal rule for the inner points
#Fourier transform ends
for j in range(3000):
	ymod[j] =np.absolute(yk[j])**2  #mod square of all the fourier transform values

ysum=ymod[0]/2+ymod[2999]/2  #initializing the trapezoidal rule for integrating all the mod squares

for j in range(2998):
       ysum+=ymod[j+1] #mod square integration for inner points
       
modfrac = ymod/ysum #modal fraction
modftud = modfrac/np.max(modfrac) #normalized modal fraction
Df = -modftud*np.log(modftud) #Differential configurational entropy
CE = (Df[0]/2)+(Df[2999]/2) #initializing trapezoidal rule for integrating DCE
for j in range(2998):
	CE+=Df[j+1] #DCE trapezoidal rule for all the inner points

print(CE)
