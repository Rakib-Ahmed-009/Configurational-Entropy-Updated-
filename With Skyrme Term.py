import autograd.numpy as np
from autograd import grad, elementwise_grad
import autograd.numpy.random as npr
from autograd.misc.optimizers import adam


def init_random_params(scale, layer_sizes, rs=npr.RandomState(42)):
    """Build a list of (weights, biases) tuples, one for each layer."""
    return [(rs.randn(insize, outsize) * scale,   # weight matrix
             rs.randn(outsize) * scale)           # bias vector
            for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]

def swish(x):
    "see https://arxiv.org/pdf/1710.05941.pdf"
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

x = np.linspace(0, 20, 500)[:, None]
y = psi(params['nn'], x)
z = psip(params['nn'], x)
fpi=93
m=10


ed=[0 for elements in range(500)]

for i in range(498):
    ed[i+1]=(2*(np.sin(y[i+1]))**2+x[i+1]**2*z[i+1]**2)+ (np.sin(y[i+1]))**2*((np.sin(y[i+1])**2)/(x[i+1]**2)+2*z[i+1]**2)

def e(k,x):
	return np.exp(-k*1j*x/(m*93))

yk=[complex(0,0) for element in range(500)]
ymod=[0 for element in range(500)]
ysum=0

for k in range(500):
	yk[k]=(e(k,0)/2)*x[0]**2*z[0]**2+(e(k,20)/2)*x[499]**2*z[499]**2

for k in range(500):
	for i in range(498):
		yk[k]+=e(k,x[i+1])*ed[i+1]
		
for k in range(500):
	ymod[k] =np.absolute(yk[k])**2

ysum=ymod[0]/2+ymod[499]/2

for k in range(498):
       ysum+=ymod[k+1]
       
modfrac = ymod/ysum
modftud = modfrac/np.max(modfrac)
Df = -modftud*np.log(modftud)
CE = (Df[0]/2)+(Df[499]/2)
for k in range(498):
	CE+=Df[k+1]

print(CE)


plt.plot(x, y, label='NN')
plt.legend()
plt.show()
