import dtree
import monkdata as m
import matplotlib.pyplot as plt



# assignment 4
def assignment4():
    datasets = [
        (m.monk1, 'monk1', m.attributes[0]),
        (m.monk1, 'monk1', m.attributes[1]),
        (m.monk1, 'monk1', m.attributes[2]),
        (m.monk1, 'monk1', m.attributes[3]),
        (m.monk1, 'monk1 max', m.attributes[4]),
        ]

    for data, name, attribute in datasets:
        summ = 0
        for value in attribute.values:
            subset = dtree.select(data, attribute, value)

            print(f'Entropy of S{value} for {name}:\t{dtree.entropy(subset)}')

            summ += len(subset)/len(data) * dtree.entropy(subset)
        
        print(dtree.entropy(data) - summ)
        print()


import random

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]


def getTree(tree, validation):
    bestTree, bestScore = tree, dtree.check(tree, validation)
    stack = [(tree, bestScore)]

    while stack:
        currTree, currScore = stack.pop()

        if currScore > bestScore:
            bestTree, bestScore = currTree, currScore

        for prunedTree in dtree.allPruned(currTree):
            pruneScore = dtree.check(prunedTree, validation)
            if pruneScore > currScore:
                stack.append((prunedTree, pruneScore))
    
    return bestTree, bestScore

from statistics import variance, mean

def assignment7():
    datasets = [m.monk1, m.monk3]
    test = [m.monk1test, m.monk3test]
    name = ['Monk1', 'Monk2']
    fractions = [i*.1 for i in range(3, 9)]

    runs = 50

    scores = []
    scores_numbers =[]

    for dataset, testset, name in zip(datasets, test, name):
        datasetScore = []
        for fraction in fractions:
            results = []
            for _ in range(runs):
                monktrain, monkval = partition(dataset, fraction)
                tree = dtree.buildTree(monktrain, m.attributes)
                tree, score = getTree(tree, monkval)

                results.append(1- dtree.check(tree, testset))
            

            datasetScore.append((mean(results), variance(results)))
        
        scores_numbers.append(datasetScore)

        # scores.append(f'Fraction: {fraction}\nMean: {mean(results)}\nVariance: {variance(results)}')

        
    return scores_numbers
    

def plot(scores):
    monk1, monk3 = scores[0], scores[1]

    fractions = [.3, .4, .5, .6, .7, .8]

    monk1_mean = [x[0] for x in monk1]
    monk1_variance = [x[1] for x in monk1]

    monk3_mean = [x[0] for x in monk3]
    monk3_variance = [x[1] for x in monk3]

    # plt.title('Mean error rates over 50 runs')
    # plt.ylabel('Error rate')
    # # plt.xticks(labels=fractions)
    # plt.xlabel('Fraction size')

    # plt.plot(fractions, monk1_mean, label = 'monk1', marker ='o')
    # plt.plot(fractions, monk3_mean, label = 'monk3', marker = 'x')
    # plt.legend()

    # plt.show()


    # plt.ion()
    plt.title('Variance of error rates')
    plt.ylabel('Variance in error')
    plt.xlabel('Fraction size')
    plt.plot(fractions, monk1_variance, label = 'monk1', marker = 'o')
    plt.plot(fractions, monk3_variance, label = 'monk3', marker = 'x')
    plt.legend()

    plt.show()




if __name__ == '__main__':
    plot(assignment7())