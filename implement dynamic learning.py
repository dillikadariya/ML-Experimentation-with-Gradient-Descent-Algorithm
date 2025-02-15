import numpy as np
import matplotlib.pyplot as plt


def generate_observations(num_observations):
    X = np.random.randint(1, num_observations, size=num_observations)
    Y = 3 * X + 2 + np.random.normal(0, 1.5, size=num_observations)
    return X, Y


def plot_data(X, Y, w=None):
    plt.scatter(X, Y, label="Data points")
    plt.ylabel("Target variable, $y$")
    plt.xlabel("Input variable, $x$")

    if w is not None:
        y_hat = get_predictions(X, w)
        mse = get_mean_squared_error(Y, y_hat)
        plt.plot(X, y_hat, color="green", label=f"Fitted Line\nw={w}, MSE={mse:.4f}")
        plt.legend()

    plt.show()


def get_predictions(X, w):
    #Computes predictions based on a linear model
    return w[0] + w[1] * X


def get_mean_squared_error(y, y_hat):
    return np.mean((y - y_hat) ** 2)


def plot_training_loss(loss_values):
    #Plots the training loss over iterations
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_values) + 1), loss_values, marker="o", linestyle="-", color="blue")
    plt.xlabel("Training Iterations")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Training Loss Over Iterations (Dynamic Learning Rate)")
    plt.grid(True)
    plt.show()


def train_model_dynamic_lr(X, y, w, alpha=0.01, num_iter=100, decay=0.001, decay_step=10):
    # implements Batch Gradient Descent with a Dynamic Learning Rate
    m = len(X)
    X = np.array(X)
    y = np.array(y)

    loss_values = []

    for it in range(num_iter):
        y_hat = get_predictions(X, w)

        # Compute gradients
        error = y - y_hat
        grad_w0 = -2 * np.mean(error)  # Mean gradient for bias
        grad_w1 = -2 * np.mean(error * X)  # Mean gradient for slope

        # Decay the learning rate every `decay_step` iterations
        if it % decay_step == 0:
            alpha = alpha / (1 + decay * it)

        # Update weights using standard gradient descent
        w[0] -= alpha * grad_w0
        w[1] -= alpha * grad_w1

        # Calculate and store MSE
        mse = get_mean_squared_error(y, get_predictions(X, w))
        loss_values.append(mse)
        print(f"Iteration {it + 1}: MSE = {mse:.4f}, Learning Rate = {alpha:.6f}, w = {w}")

    return w, loss_values


if __name__ == "__main__":
    np.random.seed(42)  # Ensure reproducibility
    X, y = generate_observations(10)

    # Initialize weights
    w = [-10.0, 0.0]
    w_new, loss_values = train_model_dynamic_lr(X, y, w)

    print("Final Weights:", w_new)
    plot_data(X, y, w_new)
    plot_training_loss(loss_values)  # Visualize training loss over iterations
