import numpy as np
import matplotlib.pyplot as plt

def generate_parametric_curve(N, num_points=100):
    t_values = np.linspace(0, 1, num_points)

    # Generate random coefficients for a polynomial of degree N-1
    coefficients_x = np.random.rand(N)
    coefficients_y = np.random.rand(N)

    # Generate N points through which the curve will pass
    points = np.random.rand(N, 2)

    def parametric_curve(t):
        t = t[:, np.newaxis]
        powers = np.array([t ** i for i in range(N-1, -1, -1)])
        
        # Constraints to ensure the curve passes through N points
        for i in range(N):
            powers_i = t_values ** i
            coefficients_x[i] = np.dot(powers_i, points[:, 0])
            coefficients_y[i] = np.dot(powers_i, points[:, 1])

        curve_x = np.dot(powers.T, coefficients_x)
        curve_y = np.dot(powers.T, coefficients_y)

        return np.column_stack([(1 - t) * points[0, 0] + t * points[-1, 0] + t * (1 - t) * curve_x,
                                (1 - t) * points[0, 1] + t * points[-1, 1] + t * (1 - t) * curve_y])

    return parametric_curve, points

# Example usage
N = 4  # You can set N to the desired number of points
parametric_curve, points = generate_parametric_curve(N)

# Plot the generated curve and points
t_values = np.linspace(0, 1, 100)
curve_points = parametric_curve(t_values)
plt.plot(curve_points[:, 0], curve_points[:, 1], label='Generated Curve')
plt.scatter(points[:, 0], points[:, 1], color='red', label='Given Points')
plt.legend()
plt.show()
