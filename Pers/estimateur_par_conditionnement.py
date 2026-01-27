import numpy as np
import matplotlib.pyplot as plt

#on veut mettre en place un estimateur par conditionnement
#on estimera la probabilité de tomber dans le disque unité en dimension 2 et on comparera avec MC naif

#on considère deux uniformes indépendentes sur [0,1], U et V. On estimera phi(X) = 1U^2+V^2}<=1
#puis on definit chi(V) comme l'ésperance conditionnelle de phi(X) sachant V
#on a que P(phi(X)) = E[1-v^2], on prend comme estimateur 1/n sum(sqrt(1-v_i^2))


def conditionnement(n):
    v = np.random.uniform(0,1,n)
    estimate = sum(np.sqrt(1-v**2))/n
    variance = np.var(np.sqrt(1-v**2))/n
    return estimate, variance

def simulate_monte_carlo(n):
    samples = np.random.uniform(0,1,(n,2))
    indicators = np.linalg.norm(samples,axis=1)<=1
    estimate = np.mean(indicators)
    variance = np.var(indicators)/n
    return estimate, variance

def plot_convergence():
    N = np.logspace(1,5,num=10,base=10,dtype=int)
    MC_naif_var = []
    Cond_var = []
    for i in N:
        cond_estimates = []
        mc_estimates = []
        for m in range(1000):
            e_cn = conditionnement(i)[0]
            e_naif = simulate_monte_carlo(i)[0]
            cond_estimates.append(e_cn)
            mc_estimates.append(e_naif)
        Cond_var.append(np.var(cond_estimates))
        MC_naif_var.append(np.var(mc_estimates))
            

    fig, ax = plt.subplots()
    fig.suptitle("Convergence de la variance")
    ax.plot(N, MC_naif_var, N, Cond_var)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Taille de l'échantillon (n)")
    ax.set_ylabel("Variance de l'estimateur")
    plt.legend(["MC naif", "Conditionnement"])
    plt.show()


def main():
    n = 10000
    e_naif, var_naif = simulate_monte_carlo(n)
    e_cn, var_cn = conditionnement(n)

    print(f"Estimation avec MC naif: {e_naif}")
    print(f"Estimation avec conditionnement: {e_cn}")
    print(f"Variance avec MC naif: {var_naif}")
    print(f"Variance avec conditionnement: {var_cn}")

    plot_convergence()

main()


