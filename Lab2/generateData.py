from typing import Tuple
import numpy, random

def getDataset(size: int):
    random.seed(10)


    # classA = numpy.concatenate(
    #     (numpy.random.randn(size//4, 2) * 0.2 + [1.5, 0.5],
    #     numpy.random.randn(size//4, 2) * 0.2 + [-1.5, 0.5]))
    # classB = numpy.random.randn(size//2, 2) * 0.2 + [0.0, -0.5]


    classA = numpy.concatenate((
    numpy.random.randn(10, 2) * 0.3 + [1.5, 0.5],
    numpy.random.randn(5, 2) * 0.3 + [-1.5, 0.5],
    numpy.random.randn(5, 2) * 0.7 + [0.0, -1.0],
    ))
    classB=numpy.random.randn(20, 2) * 0.3 + [0.0, -0.5]

    inputs = numpy.concatenate((classA, classB))
    targets = numpy.concatenate(
        (numpy.ones(classA.shape[0]),
        -numpy.ones(classB.shape[0])))

    N = inputs.shape[0]

    permute = list(range(N))
    random.shuffle(permute)
    
    inputs = inputs[permute, :]
    targets = targets[permute]

    return inputs, targets, classA, classB