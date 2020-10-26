import numpy as np

#TODO: REMOVE ALL PRINTS!
def rmse(u, v, mat):
    mask = mat > 0
    res = np.sum(((u.dot(v.T) - mat) * mask) ** 2) / float(np.sum(mask))
    return np.sqrt(res)


def my_recommender(rate_mat, lr, with_reg):
    """

    :param rate_mat:
    :param lr:
    :param with_reg:
        boolean flag, set true for using regularization and false otherwise
    :return:
    """

    learning_rate = 0.0005
    max_iter = 200
    epsilon = 1e-5
    reg_coef = 0.0001
    n_user, n_item = rate_mat.shape[0], rate_mat.shape[1]

    # Initialize U and V
    U = np.random.rand(n_user, lr) / lr
    V = np.random.rand(n_item, lr) / lr

    U_new = U.copy()
    V_new = V.copy()

    error = 1000
    cnt = 0
    while cnt < max_iter:

        for u in range(0,n_user):
            for k in range(0,lr):
                mask_i = rate_mat[u, :] > 0
                calc_i = ( (rate_mat[u, :] - np.sum(U[u,:]*V[:,:], axis=1) ) * V[:,k] * mask_i).sum()
                if with_reg:
                    reg_i = -2 * reg_coef * U[u,k]
                else:
                    reg_i = 0
                U_new[u,k] = U[u,k] + 2 * learning_rate * calc_i + reg_i

        for i in range(0, n_item):
            for k in range(0,lr):
                mask_u = rate_mat[:,i] > 0
                calc_u = ( (rate_mat[:,i] - np.sum( U[:,:]*V[i,:], axis=1) ) * U[:,k] * mask_u).sum()
                if with_reg:
                    reg_u = -2 * reg_coef * V[i,k]
                else:
                    reg_u = 0
                V_new[i,k] = V[i,k] + 2 * learning_rate * calc_u + reg_u

        U = U_new.copy()
        V = V_new.copy()

        cnt = cnt + 1
        # Calculcate RMSE and convergence check
        last_error = error
        error = rmse(U, V, rate_mat)
        print("Iteration #{}: RMSE Error is {}".format(cnt, error))
        chg = abs( last_error - error)
        if chg <= epsilon:
            break
    return U, V
