import numpy as np
import matplotlib.pyplot as plt

#on essaye de simuler une suite iid de loi X de densité f(x)
#on essayera d'utiliser la méthode de Monte Carlo naïf, donc on doit savoir simuler X et phi(X)
#on considerera l'estimateur I_n comme la moyenne empirique des phi(X_i)

#on considere le quart de disque unité dans le rectangle [0,1]x[0,1]
#on veut estimer l'aire de ce quart de disque unité, qui vaut pi/4
#on prendra la suite p_n = 1/n sum_{i=1}^n(indicatrice(norm(X_i)<=1)) 

#comme la probabilité que la norme d'X_i soit inférieure à 1 es pi/4, 
# alors la variance de phi(X) est pi/4(1-pi/4) car phi(X) suit une loi de Bernoulli de paramètre pi/4

#we want to see the precision of the method showing too how the points are distributed

def simulate_monte_carlo_naif(n):
    samples = np.random.uniform(0, 1, (n, 2))
    indicators = np.linalg.norm(samples, axis=1) <= 1
    estimate = np.mean(indicators)
    return estimate, samples

def main():
    n = 1000
    #on va essayer d'estimer la valeur de pi
    estimate, samples = simulate_monte_carlo_naif(n)
    #finalment on calcule le nombre de cifres significatifs qu'on a pour pi
    significant_digits = np.abs(4*estimate - np.pi)/np.abs(np.pi)
    significant_digits =  np.floor(-np.log10(2*significant_digits))
    
    print(f"Valeur de la surface du disque unité approximé: {estimate}")
    print(f"Valeur de pi approximé: {4*estimate}")
    print(f"Nombre de chiffres significatifs pour pi: {significant_digits}")



    # Plot quarter circle with the points inside blue and the points outside green
    inside_points = samples[np.linalg.norm(samples, axis=1) <= 1] 
    outside_points = samples[np.linalg.norm(samples, axis=1) > 1]
    plt.figure(figsize=(6,6))
    plt.scatter(inside_points[:,0], inside_points[:,1], color='blue', s=1, label='Inside Quarter Circle')
    plt.scatter(outside_points[:,0], outside_points[:,1], color='green', s=1, label='Outside Quarter Circle')
    circle = plt.Circle((0, 0), 1, color='red', fill=False, label='Quarter Circle Boundary')
    plt.gca().add_artist(circle)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Monte Carlo Naïf Simulation')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.show()

main()