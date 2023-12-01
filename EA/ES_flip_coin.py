import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass

np.random.seed(42)

budget = 5000
dimension = 50
# 设置种群大小、维度和预算
pop_size = 30  # 种群大小 μ
num_offspring = 200  # 后代数量 λ
mutation_rate = 0.01  # 变异概率
# dimension = 50  # 问题维度
# budget = 5000  # 评估预算
sigma = 0.1  # 步长

def bit_flip_mutation(individual, mutation_rate=0.01):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual

def uniform_crossover(parent1, parent2):
    mask = np.random.randint(0, 2, size=len(parent1))
    offspring = np.where(mask, parent1, parent2)
    return offspring


# To make your results reproducible (not required by the assignment), you could set the random seed by
# `np.random.seed(some integer, e.g., 42)`

def studentnumber1_studentnumber2_ES(problem):
    # hint: F18 and F19 are Boolean problems. Consider how to present bitstrings as real-valued vectors in ES
    # initial_pop = ... make sure you randomly create the first population

    # 初始化种群
    population = np.random.randint(0, 2, (pop_size, dimension))

    # `problem.state.evaluations` counts the number of function evaluation automatically,
    # which is incremented by 1 whenever you call `problem(x)`.
    # You could also maintain a counter of function evaluations if you prefer.
    while problem.state.evaluations < budget:
        # please implement the mutation, crossover, selection here
        # .....
        # this is how you evaluate one solution `x`
        # f = problem(x)
        if problem.state.evaluations + num_offspring > budget:
            break  # 如果超出预算，结束循环

        # # 重组步骤
        # offspring = np.array([uniform_crossover(population[np.random.randint(pop_size)],
        #                                         population[np.random.randint(pop_size)]) 
        #                       for _ in range(num_offspring)])

        # 变异步骤
        # 生成后代
        offspring = np.empty((num_offspring, dimension), dtype=int)
        for i in range(num_offspring):
            parents = population[np.random.choice(pop_size, 2, replace=False)]
            offspring[i] = uniform_crossover(parents[0], parents[1])
            offspring[i] = bit_flip_mutation(offspring[i], mutation_rate)
        
        # 评估后代适应度
        fitness = np.array([problem(ind) for ind in offspring])

        # (μ, λ) 选择：选择适应度最好的μ个个体作为下一代
        indices = np.argsort(fitness)[-pop_size:]
        population = offspring[indices]

        
    # no return value needed 


def create_problem(fid: int):
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data_bit_flip",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
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


