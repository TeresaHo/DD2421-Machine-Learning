import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from generateData import getDataset

class SupportVectorMachines:
    def __init__(self, size=20, C=10, p=4, data = None, sigma = 2):
        if data is None:
            # self.inputs, self.targets, self.classA, self.classB = getDataset(size)
            return
        else:
            self.inputs, self.targets, self.classA, self.classB = data
            
            
        self.N = self.inputs.shape[0]

        self.start = numpy.zeros(self.N) # initial alpha value
        self.a = [] # alpha value after minimizing
        self.C = C # coeffient for slack
        self.B = [(0, C) for b in range(self.N)] #bounds
        self.p = p # parameter for polyKernel
        self.sigma = sigma # parameter for Radial Basis kernel       
        self.nonz_alpha = []
        self.nonz_inputs = []
        self.nonz_targets = []
        self.b = None
        self.kernel = None
        
    def linearKernel(self,x,y):
        return numpy.dot(numpy.transpose(x), y)

    def polyKernel(self,x,y):
        return (numpy.dot(numpy.transpose(x), y) + 1)**self.p

    def RBFKernel(self, x, y):
        return math.exp(-((math.dist(x, y)**2)/(2*self.sigma*self.sigma)))

    def objective(self,alpha):
        s1 = 0
        s2 = 0
        for i in range(self.N):
            s2 = s2 + alpha[i]
            for j in range(self.N):
                s1 = s1 + alpha[i]*alpha[j]*self.targets[i]*self.targets[j]*self.kernel(self.inputs[i],self.inputs[j])
        return 0.5*s1 - s2

    def zerofun(self,alpha):
        return numpy.dot(alpha, self.targets)

    def runMinimize(self,kernel):
        self.name = kernel
        if kernel == "linear":
            self.kernel = self.linearKernel
        elif kernel == "polynomial":
            self.kernel = self.polyKernel
        else:
            self.kernel = self.RBFKernel
        XC = {'type':'eq', 'fun': self.zerofun} #contraints  
        ret = minimize(self.objective, self.start, bounds=self.B, constraints=XC)
        self.a = ret.x
        
        print(ret.success)
        if ret.success:
            print ("Optimal solution found")
        else:
            print ("No optimal solution could be found")
        return

    def extract_nonzeroes(self):
        self.nonz_alpha = [self.a[i] for i in range(self.N) if abs(self.a[i]) > 10e-5]
        self.nonz_inputs = [self.inputs[i] for i in range(self.N) if abs(self.a[i]) > 10e-5]
        self.nonz_targets = [self.targets[i] for i in range(self.N) if abs(self.a[i]) > 10e-5]       

    def calculateB(self):
        total = 0
        sv = self.nonz_inputs[0]
        sv_target = self.nonz_targets[0]
        for i in range(len(self.nonz_alpha)):
            total += self.nonz_alpha[i] * self.nonz_targets[i] * self.kernel(sv, self.nonz_inputs[i])
        self.b = total - sv_target
            
    def indicator(self, x, y):
        self.calculateB()
        total = 0
        for i in range(len(self.nonz_alpha)):
            total += self.nonz_alpha[i] * self.nonz_targets[i] * self.kernel([x, y], self.nonz_inputs[i])
        return total - self.b

    def plot(self):
        # plot datapoints
        
        plt.title(f'Decision boundary with Kernel: {self.name}')
        
        plt.plot([p[0] for p in self.classA], [p[1] for p in self.classA], 'b.')
        plt.plot([p[0] for p in self.classB], [p[1] for p in self.classB], 'r.')

        # plot hyperplane
        xgrid = numpy.linspace(-5, 5)
        ygrid = numpy.linspace(-4, 4)
        grid = numpy.array([[self.indicator(x, y) for x in xgrid] for y in ygrid])
        plt.contour(xgrid, ygrid, grid, (-1.0,0.0,1.0), colors=('red','black','blue'), linewidths=(1,3,1))
        plt.show()

def main():
    svm = SupportVectorMachines(20)
    svm.runMinimize("linear")  
    svm.extract_nonzeroes()   
    svm.plot()

if __name__ == "__main__":
    main()