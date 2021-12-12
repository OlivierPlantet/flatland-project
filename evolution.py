from evaluate import get_env, get_state_action_size, evaluate
from policy import NeuroevoPolicy
from argparse import ArgumentParser
import numpy as np
import logging
import cma
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA

# ---- Algorithms ----
def oneplus_lambda(x, fitness, gens, lam, std=0.01, rng=np.random.default_rng()):
    x_best = x
    f_best = -np.Inf
    n_evals = 0
    for g in range(gens):
        N = rng.normal(size=(lam, len(x))) * std
        for i in range(lam):
            ind = x + N[i, :]
            f = fitness(ind)
            if f > f_best:
                f_best = f
                x_best = ind
        x = x_best
        n_evals += lam
        logging.info('\t%d\t%d', n_evals, f_best)

    print(x_best.shape)
    return x_best

def docma(x, fitness, gens, pop):
    es = cma.CMAEvolutionStrategy(x, np.std(x), {'popsize': pop})
    for g in range(gens):
        res = es.ask()
        es.tell(res, [fitness(x) for x in res])
    return res

def mues(x, fitness, gens, lam, alpha=0.1):
    x_best = x
    f_best = fitness(x)
    e = 0
    n_evals = 0
    for g in range(gens):
        N = np.random.normal(size=(lam,len(x)))
        F = np.zeros(lam)
        for i in range(lam):
            ind = x + N[i,:]
            F[i] = fitness(ind)
            if F[i] > f_best:
                f_best = F[i]
                x_best = ind
            e += 1
        mu_f = np.mean(F)
        std_f = np.std(F)
        A = F
        if std_f > 0:
            A = (F - mu_f) / std_f
        x = x - alpha * np.dot(A, N) / lam
        
        #Logging
        n_evals += lam
        logging.info('\t%d\t%d', n_evals, f_best)
    return x_best   

def ga(x, fitness, pop):
    class MyProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=len(x),
                            n_obj=1,
                            n_constr=0,
                            xl = -1000*np.ones(len(x)),
                            xu = 1000*np.ones(len(x)))

        def _evaluate(self, x, out, *args, **kwargs):
            out["F"] = -fitness(x)

    problem = MyProblem()

    N = np.random.normal(size=(100,len(x)))
    first = x + N

    algorithm = GA(
        pop_size=pop,
        sampling=first,
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True)

    res = minimize(problem,
            algorithm,
            seed=1,
            verbose=False)
    return res.X

def nsga(x, fitness, gens, pop) :
    class MyProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=len(x),
                            n_obj=1,
                            n_constr=0,
                            xl = -1000*np.ones(len(x)),
                            xu = 1000*np.ones(len(x)))

        def _evaluate(self, x, out, *args, **kwargs):
            out["F"] = -fitness(x)

    problem = MyProblem()

    N = np.random.normal(size=(100,len(x)))
    first = x + N

    algorithm = NSGA2(
        sampling=first,
        pop_size=pop,
        n_offsprings=100,
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True
    )

    res = minimize(problem,
        algorithm,
        ('n_gen', gens),
        seed=1,
        save_history=True,
        verbose=False)

    return res.X[0]

# ---------------------------------------------------------------   

def fitness(x, s, a, env, params):
    policy = NeuroevoPolicy(s, a)
    policy.set_params(x)
    return evaluate(env, params, policy)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default='oneplus_lambda', type=str)
    parser.add_argument('--alpha', default=10, type=float)
    parser.add_argument('-e', '--env', help='environment', default='small', type=str)
    parser.add_argument('-g', '--gens', help='number of generations', default=100, type=int)
    parser.add_argument('-p', '--pop', help='population size', default=10, type=int)
    parser.add_argument('-s', '--seed', help='seed for evolution', default=0, type=int)
    parser.add_argument('--log', help='log file', default='evolution.log', type=str)
    parser.add_argument('--weights', help='filename to save policy weights', default='weights', type=str)
    args = parser.parse_args()
    logging.basicConfig(filename=args.log, level=logging.DEBUG,
                        format='%(asctime)s %(message)s')

    # starting point
    env, params = get_env(args.env)
    s, a = get_state_action_size(env)
    policy = NeuroevoPolicy(s, a)

    # evolution
    rng = np.random.default_rng(args.seed)
    start = rng.normal(size=(len(policy.get_params(),)))

    def fit(x):
        return fitness(x, s, a, env, params)
    
    if args.model == 'oneplus_lambda' :
        x_best = oneplus_lambda(start, fit, args.gens, args.pop, rng=rng)
    
    elif args.model == 'cma' :
        x_best = docma(start, fit, args.gens, args.pop)

    elif args.model == 'mues' :
        x_best = mues(start, fit, args.gens, args.pop, args.alpha)

    elif args.model == 'nsga' :
        x_best = nsga(start, fit, args.gens, args.pop)

    elif args.model == 'ga' :
        x_best = ga(start, fit, args.pop)

    # Evaluation
    policy.set_params(x_best)
    policy.save(args.weights)
    best_eval = evaluate(env, params, policy)
    print('Best individual: ', x_best[:5])
    print('Fitness: ', best_eval)