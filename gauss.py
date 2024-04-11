def gauss_seidel(A, b, x0, tol=1e-6, max_iter=1000):
    n = len(b)
    x = x0.copy()
    for iter_count in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            summation = sum(A[i][j] * x_new[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - summation) / A[i][i]
        if all(abs(x_new[i] - x[i]) < tol for i in range(n)):
            return x_new, iter_count + 1
        x = x_new
    raise ValueError("Gauss-Seidel method did not converge within the maximum number of iterations.")

# Example usage:
A = [[4, -1, 0], [-1, 4, -1], [0, -1, 3]]
b = [5, 10, 15]
x0 = [0, 0, 0]

solution, iterations = gauss_seidel(A, b, x0)
print("Solution:", solution)
print("Number of iterations:", iterations)
