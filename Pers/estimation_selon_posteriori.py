import numpy as np
import matplotlib.pyplot as plt

#on veut utiliser la methode de l'estimation selon a posteriori
#supposons theta est une variable aleatoire suivant une loi de cauchy et X|theta suit 
# une normale de moyenne theta et variance 1.

#On sait simuler la loi a priori, aussi pour tout theta on sait calculer p_theta(X) et on connait l'EMV de X|theta
#notre but est estimer E[phi(theta)|X] pour une fonction phi donnée (la moyenne a posteriori si phi est l'identité)

#on sait que la moyenne a posteriori es donné par la intégrale de phi(theta)Pi(theta|X)dPi(theta|X) et par
#la formule de bayes on a que Pi(theta|X) est Pi(theta)p_theta(X)/intégrale sur le domaine de theta de Pi(t)p_t(X)dt
#on construit un estimateur pour ces intégrales en utilisant un échantillon de thetas simulés selon la loi a priori
#et en prenant la somme pour chaque theta_i de phi(theta_i)p_theta_i(X) divisé par la somme de p_theta_i(X)

#On suppose qu'il existe un theta_0 et on genère un échantillon X de taille m=50

def phi_theta(theta):
    #on definit la fonction test pour theta
    #si phi est l'identité on calcule la moyenne a posteriori
    return theta

def generer_echantillon_X(theta_0, m):
    X = np.random.normal(loc=theta_0, scale=1, size=m)
    return X

def vraisemblance(theta, X, m):
    return 1/((2*np.pi)**(m/2)) * np.exp(-1/2 * np.sum((X-theta)**2))

#Pour simuler l'échantillon de thetas on utilise le méthode d'inversion
def simulation_a_posteriori(n, m, X):
    U = np.random.uniform(0,1,n)
    Th = phi_theta(np.tan(np.pi*(U-1/2)))
    #un a un échantillon de theta_i qui suivent une loi de cauchy. En considerant un échantillon X fixé, 
    #qui suit conditionné à theta une loi normale de moyenne theta et variance 1, on peut calculer la vraisemblance
    #de p_theta_i(X)
    V = []
    for t in Th:
        V.append(vraisemblance(t, X, m))
    In = np.dot(Th, V)/np.sum(V)
    return In

def influence_a_priori(N, n, theta_0, m):
    E = []
    for _ in range(N):
        X = generer_echantillon_X(theta_0, m)
        Emp = np.mean(X)
        E_Post = simulation_a_posteriori(n,m,X)
        E.append(E_Post - Emp)
    return E



def main():
    theta_0 = 2.0
    m = 10
    #on voit que comme m augmente, l'estimateur converge vers l'EMV, (si phi est l'identité)
    # qui est logique car la loi a posteriori devient de plus en plus concentrée autour de theta_0
    X = generer_echantillon_X(theta_0, m)

    #on étudie la convergence de l'estimateur en fonction de n
    repétitions = 20
    estimations = []
    deviations = []
    ns = np.logspace(1, 3, 15, dtype=int)
    for ni in ns:
        ests = []
        for _ in range(repétitions):
            est = simulation_a_posteriori(ni, m, X)
            ests.append(est)
        estimations.append(np.mean(ests))
        deviations.append(np.std(ests))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.errorbar(ns, estimations, deviations, fmt='.', linewidth=1, capsize=2)
    ax1.set_xscale('log')
    ax1.set_xlabel("Taille de l'échantillon n")
    ax1.set_ylabel("Estimation de E[phi(theta)|X]")
    ax1.set_title("Convergence de l'estimateur selon a posteriori")
    ax1.axhline(y=np.mean(X), color='g', linestyle='--', label='EMV')
    ax1.legend()

    #on répresente aussi l'influence de la loi a priori
    N = 100
    n = 1000
    E = influence_a_priori(N, n, theta_0, m)
    ax2.hist(E, bins=15, density=True, alpha=0.7, edgecolor='black')
    ax2.set_title("Influence de la loi a priori Cauchy")
    ax2.set_xlabel('Différence: Estimation Bayesienne - EMV')
    ax2.set_ylabel('Fréquence')
    ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5, label="Différence nulle")
    ax2.axvline(x=np.mean(E), color='b', linestyle='-', alpha=0.7, label="Moyenne des différences")
    ax2.legend()
    plt.show()

main()


    



 


