import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

#avc monte_carlo_naif, on estime phi(x) avec 1/n sum phi(X_i), mais si phi(X) prend grandes 
# valeurs où X tombe rarement, la variance de l'estimateur peut être grande par raport a E[phi(X)]

#avec l'échantillonage préférentiel, on prend Y de loi g qu'on sait simuler, telle que f*mu << g*mu sur l'ensemble
# E_0 des x où phi(x) est positif et pour tout y, on peut calculer w(y) = f(y)/g(y)

#alors I = E[phi(x)] = int_{E}(phi(x)f(x)dx) = int_{E_0}(phi(x)f(x)dx) = int_{E_0}(phi(x)f(x)/g(x) * g(x)dx) = int_{E_0}(phi(x)w(x)g(x)dx)
# = int_{E}(phi(x)w(x)g(x)dx) = E[phi(Y)w(Y)]
# donc on etime I avec 1/n sum(phi(Y_i)w(Y_i)) avec Y_i iid de loi g


#supposons que X suit une loi normale N(0,1) et phi(X) = P(X > 3). On va faire la comparaison entre MC naif et IS
#par le IS on prendra Y qui suit la loi N(3,1)
def simulate_monte_carlo_naif(n):
    samples = np.random.standard_normal(n) #on prend un echantillon de taille n de N(0,1)
    indicators = samples>3 #on calcule combien de fois X_i > 3
    estimate = np.mean(indicators) #on calcule la moyenne empirique
    variance = (1/n * sum(indicators**2) - estimate**2)
    return estimate, samples, variance

def simulate_importance_sampling(n):
    samples = np.random.normal(3,1,n)
    weights = np.exp(-0.5*((samples)**2 - (samples - 3)**2))
    indicators = samples>3
    estimate = np.mean(indicators*weights)
    variance = ((1/n)*sum((indicators*weights)**2) - estimate**2)
    return estimate, samples, variance

#on veut comparer les deux méthodes et leur precision
#on veut aussi montrer la distribution des points simulés, en different couleur selon s'ils son naif ou IS
def main():
    n = 10000
    e_naif, s_naif, variance_naif = simulate_monte_carlo_naif(n)
    e_is,  s_is, variance_is = simulate_importance_sampling(n)

    print(f"Estimation avec MC naif:{e_naif}")
    print(f"Estimation avec importance sampling:{e_is}")
    print(f"Variance avec MC naif:{variance_naif}")
    print(f"Variance avec importance sampling:{variance_is}")
    print(f"Longueur de l'intervalle de confiance a 95% avec MC naif: {2 * 1.96 * np.sqrt(variance_naif)}")
    print(f"Longueur de l'intervalle de confiance a 95% avec importance sampling: {2 * 1.96 * np.sqrt(variance_is)}")

    #on dessine les points simulés montrant la différence entre les deux méthodes, avec la grande cantité
    #de points de IS autour de 2 et les peux nombreux de MC naif.
    plt.figure(figsize=(12,6))
    plt.hist(s_naif, bins=100, density=True, alpha=0.5, label='MC Naif', color='blue')
    plt.hist(s_is, bins=100, density=True, alpha=0.5, label='Importance Sampling', color='orange')
    x = np.linspace(-4, 6, 1000) 
    #draw a line at x=3 showing the intereseting threshold of points that we want to estimate
    plt.axvline(x=3, color='red', linestyle='--', label='Threshold x=3')
    plt.title('Comparison of MC Naif and Importance Sampling')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.show()


main()

