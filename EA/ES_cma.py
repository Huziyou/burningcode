import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass

np.random.seed(42)

budget = 5000
dimension = 50
# 设置种群大小、维度和预算
pop_size = 100  # 种群大小 μ
num_offspring = 200  # 后代数量 λ

# def uniform_crossover(parent1, parent2):
#     mask = np.random.randint(0, 2, size=parent1.shape)
#     offspring = np.where(mask, parent1, parent2)
#     return offspring


# To make your results reproducible (not required by the assignment), you could set the random seed by
# `np.random.seed(some integer, e.g., 42)`

def studentnumber1_studentnumber2_ES(problem):
    # CMA-ES参数初始化
    sigma = 0.3  # 步长
    # pop_size = 4 + int(3 * np.log(dimension))  # 种群大小
    mu = pop_size // 2  # 父代个体数
    weights = np.array([np.log(mu + 0.5) - np.log(i + 1) for i in range(mu)])  # 重组权重
    weights /= np.sum(weights)  # 归一化权重
    mueff = np.sum(weights)**2 / np.sum(weights**2)  # 方差的有效性
    # 初始化协方差矩阵参数
    cc = 4 / (dimension + 4)
    cs = (mueff + 2) / (dimension + mueff + 3)
    c1 = 2 / ((dimension + 1.3)**2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dimension + 2)**2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dimension + 1)) - 1) + cs
    pc = np.zeros(dimension)
    ps = np.zeros(dimension)
    B = np.eye(dimension)
    D = np.ones(dimension)
    C = np.eye(dimension)
    invsqrtC = np.eye(dimension)
    eigeneval = 0  # 特征值的计算次数
    chiN = dimension**0.5 * (1 - 1 / (4 * dimension) + 1 / (21 * dimension**2))

    # 初始化种群的平均值
    xmean = np.random.rand(dimension)  # 种群的平均值
    arfitness = np.zeros(pop_size)  # 初始化适应度数组
    evaluations = 0  # 评估次数计数器

    while evaluations < budget:
        # 生成新的候选解
        Z = np.random.randn(pop_size, dimension)
        X = np.array([xmean + sigma * (B @ (D * Z[i])) for i in range(pop_size)])
        # 将解四舍五入为二进制，并评估适应度
        for i in range(pop_size):
            X_bin = np.where(X[i] > 0.5, 1, 0)
            arfitness[i] = problem(X_bin)
            evaluations += 1

        # 更新种群平均值
        sorted_indices = np.argsort(arfitness)
        xold = xmean.copy()
        xmean = np.dot(weights, X[sorted_indices[:mu]])

        # 更新进化路径
        Zmean = np.dot(weights, Z[sorted_indices[:mu]])
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * B @ Zmean
        hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * evaluations / pop_size)) / chiN < 1.4 + 2 / (dimension + 1)
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * B @ (D * Zmean)

        # 更新协方差矩阵 C
        C = (1 - c1 - cmu * sum(weights)) * C  # 衰减旧的协方差矩阵
        C += c1 * (pc @ pc.T + (1 - hsig) * cc * (2 - cc) * C)  # 排名一的更新
        for i in range(mu):  # 排名其余的更新
            C += cmu * weights[i] * (B @ (D * Z[sorted_indices[i]])) @ (B @ (D * Z[sorted_indices[i]])).T

        # 更新步长 sigma
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

        # 更新B和D，用于表示C的特征向量和特征值
        if evaluations - eigeneval > pop_size / (c1 + cmu) / dimension / 10:  # 条件控制更新频率
            eigeneval = evaluations
            C = np.triu(C) + np.triu(C, 1).T  # 强制对称
            D, B = np.linalg.eigh(C)  # 特征分解
            D = np.sqrt(D)  # 方差
            invsqrtC = B @ np.diag(1 / D) @ B.T  # C的平方根倒数

        # 打印进度
        if evaluations % 1000 == 0 or evaluations == budget:
            print(f"Evaluations: {evaluations}/{budget}, Best fitness: {arfitness[sorted_indices[0]]}")



def create_problem(fid: int):
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data_cma",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="evolution_strategies",  # name of your algorithm
        algorithm_info="Practical assignment of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    F18, _logger = create_problem(18)
    for run in range(20): 
        studentnumber1_studentnumber2_ES(F18)
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    F19, _logger = create_problem(19)
    for run in range(20): 
        studentnumber1_studentnumber2_ES(F19)
        F19.reset()
    _logger.close()


