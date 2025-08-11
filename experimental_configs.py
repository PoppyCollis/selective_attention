import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def plot_config(means, covariances, n_samples, exp, config):
    samples = []
    labels = []
    for i, (mean, cov) in enumerate(zip(means, covariances)):
        points = np.random.multivariate_normal(mean, cov, size=n_samples)
        samples.append(points)
        labels.append(np.full(n_samples, i))

    samples = np.vstack(samples)
    labels = np.concatenate(labels)

    # Plotting
    plt.figure(figsize=(6, 6))
    plt.scatter(samples[:, 0], samples[:, 1], c=labels, cmap='viridis', s=10)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(-300,300)
    plt.ylim(-300,300)
    plt.title(f'Experiment {exp}, config {config}')

    # Add legend with mean and covariance
    legend_elements = []
    for i, (mean, cov) in enumerate(zip(means, covariances)):
        label = f"μ={mean.tolist()}, σ={np.sqrt(cov[0][0])}"
        color = plt.cm.viridis(i / 2)  # 0, 0.5, 1 for three components
        legend_elements.append(Patch(facecolor=color, edgecolor='black', label=label))

    plt.legend(handles=legend_elements, loc='upper right', fontsize=8)
    plt.show()
    
def main():

    k = 3
    n_samples = 375
    std = 64
    sigma = std*std
        
    exp = 2 # 1,2,3 
    config = 1 # 1,2,3,4

    mu_xs =  [[-96,0,96], 
            [-128,0,128], 
            [-96,-59,96], 
            [-96,59,96],
            [-64,0,64],
            [-51,0,51],
            [-64,-64,64],
            [-64,64,64]]

    mu_ys = [[0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [37,-74,37],
            [30,-59,30],
            [37,0,37],
            [37,0,37]]

    if exp == 1 or exp == 3:
        config = config - 1
    elif exp == 2:
        config = config + 3
    else:
        raise ValueError("Experiment number (exp) must be either 1, 2 or 3")
        
    x_left = mu_xs[(config)][0] # top left distribution mean
    x_bottom = mu_xs[(config)][1] # bottom distribution mean
    x_right = mu_xs[(config)][2] # top right distribution mean

    y_left = mu_ys[(config)][0] # top left distribution mean
    y_bottom = mu_ys[(config)][1] # bottom distribution mean
    y_right = mu_ys[(config)][2] # top right distribution mean

    means = np.array([[x_left, y_left], [x_bottom, y_bottom], [x_right, y_right]])

    covariances = np.array([[[sigma, 0], [0, sigma]],        
                            [[sigma, 0], [0, sigma]],      
                            [[sigma, 0], [0, sigma]]])


    print(means)
            
    plot_config(means, covariances, n_samples, exp, config)
    
if __name__ == "__main__":
    means, covariances = main()