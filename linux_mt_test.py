from torch.multiprocessing import Pool

def doSomething(x):
    return x*x

if __name__ == "__main__":
    threads = 8
    pool = Pool(threads)
    iterations = 101
    for _ in range(iterations):
        inputs = []
        for i in range(threads):
            inputs.append(i)
        results = pool.map(doSomething, inputs)
    print(results)