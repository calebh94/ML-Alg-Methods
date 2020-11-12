import numpy as np


def rmse(u, v, mat, mask):
    res = np.sum(((u.dot(v.T) - mat) * mask) ** 2) / float(np.sum(mask))
    return np.sqrt(res)


def my_recommender(rate_mat, lr, with_reg):
    # Select Parameters
    if with_reg:
        reg_coef = 0.0015
        epsilon = 1e-5
        learning_rate = 0.00045
    else:
        learning_rate = 0.00045
        reg_coef = 0.00
        epsilon = 1e-5
    max_iter = 300
    n_user, n_item = rate_mat.shape[0], rate_mat.shape[1]

    # Initialize U and V
    U = np.random.rand(n_user, lr) / lr
    V = np.random.rand(n_item, lr) / lr

    U_new = U.copy()
    V_new = V.copy()

    mask = rate_mat > 0

    error = 1000
    cnt = 0
    while cnt < max_iter:

        Vt = np.transpose(V)
        Ut = np.transpose(U)
        U_new = U + 2 * learning_rate * ( rate_mat - np.multiply(U @ Vt, mask)) @ V - 2 * learning_rate * reg_coef * U
        V_new = V + 2 * learning_rate * np.transpose(rate_mat - np.multiply(U @ Vt, mask)) @ U - 2 * learning_rate * reg_coef * V

        U = U_new.copy()
        V = V_new.copy()

        cnt = cnt + 1
        # Calculcate RMSE and convergence check
        last_error = error
        error = rmse(U, V, rate_mat, mask)
        chg = abs( last_error - error)
        if chg <= epsilon:
            break
        elif chg <= 1e-4:
            learning_rate = learning_rate / 2
    return U, V


def isNaN(num):
    return num != num