import numpy as np

#TODO: REMOVE ALL PRINTS!
def rmse(u, v, mat):
    mask = mat > 0
    res = np.sum(((u.dot(v.T) - mat) * mask) ** 2) / float(np.sum(mask))
    return np.sqrt(res)



#TODO: my_recommender_SGD()

def my_recommender(rate_mat, lr, with_reg):
    """

    :param rate_mat:
    :param lr:
    :param with_reg:
        boolean flag, set true for using regularization and false otherwise
    :return:
    """

    # learning_rate = 0.0005
    # max_iter = 200
    # epsilon = 1e-5
    # reg_coef = 0.0001
    if with_reg:
        reg_coef = 0.0065
        epsilon = 1e-4
        learning_rate = 0.00035
    else:
        learning_rate = 0.00035
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


        # for u in range(0,n_user):
        #     for k in range(0,lr):
        #         mask_i = rate_mat[u, :] > 0
        #         calc_i = ( (rate_mat[u, :] - np.sum(U[u,:]*V[:,:], axis=1) ) * V[:,k] * mask_i).sum()
        #         if with_reg:
        #             reg_i = -2 * learning_rate * reg_coef * U[u,k]
        #         else:
        #             reg_i = 0
        #         U_new[u,k] = U[u,k] + 2 * learning_rate * calc_i + reg_i
        #
        # for i in range(0, n_item):
        #     for k in range(0,lr):
        #         mask_u = rate_mat[:,i] > 0
        #         calc_u = ( (rate_mat[:,i] - np.sum( U[:,:]*V[i,:], axis=1) ) * U[:,k] * mask_u).sum()
        #         if with_reg:
        #             reg_u = -2 * learning_rate * reg_coef * V[i,k]
        #         else:
        #             reg_u = 0
        #         V_new[i,k] = V[i,k] + 2 * learning_rate * calc_u + reg_u

        U = U_new.copy()
        V = V_new.copy()

        cnt = cnt + 1
        # Calculcate RMSE and convergence check
        last_error = error
        error = rmse(U, V, rate_mat)
        # print("Iteration #{}: RMSE Error is {}".format(cnt, error))
        chg = abs( last_error - error)
        if chg <= epsilon:
            break
    return U, V


# def my_recommender_paramsearch(rate_mat, test_mat, lr, with_reg):
#     """
#
#     :param rate_mat:
#     :param lr:
#     :param with_reg:
#         boolean flag, set true for using regularization and false otherwise
#     :return:
#     """
#
#     learning_rate_arr = np.array([0.0002, 0.0003, 0.0004, 0.0005])
#     # max_iter_arr = np.array([200, 300, 500])
#     # epsilon_arr = np.array([1e-5, 1e-6, 5e-6])
#     reg_coef_arr = np.array([0.0005, 0.0006, 0.00065, 0.0007, 0.00085, 0.0009])
#     n_user, n_item = rate_mat.shape[0], rate_mat.shape[1]
#     reg_coef = 0.005
#     max_iter = 500
#     epsilon = 1e-6
#     learning_rate = 0.00048
#
#     results = []
#     test_error_best = 1000
#     params_best = []
#     while (reg_coef_arr.size > 0):
#
#         # Sample params
#         # if learning_rate_arr.size  > 0:
#         #     learning_rate = learning_rate_arr[0]
#         #     learning_rate_arr=np.delete(learning_rate_arr, learning_rate)
#         # # if max_iter_arr.size > 0:
#         #     max_iter = np.random.choice(max_iter_arr, 1, replace=False)
#         #     max_iter_arr=np.delete(max_iter_arr, max_iter)
#         # if epsilon_arr.size > 0:
#         #     epsilon = np.random.choice(epsilon_arr, 1, replace=False)
#         #     epsilon_arr=np.delete(epsilon_arr, epsilon)
#         if reg_coef_arr.size > 0:
#             reg_coef = reg_coef_arr[0]
#             reg_coef_arr=np.delete(reg_coef_arr, reg_coef)
#
#         # Initialize U and V
#         U = np.random.rand(n_user, lr) / lr
#         V = np.random.rand(n_item, lr) / lr
#
#         U_new = U.copy()
#         V_new = V.copy()
#
#         error = 1000
#         cnt = 0
#         while cnt < max_iter:
#
#             for u in range(0, n_user):
#                 for k in range(0, lr):
#                     mask_i = rate_mat[u, :] > 0
#                     calc_i = ((rate_mat[u, :] - np.sum(U[u, :] * V[:, :], axis=1)) * V[:, k] * mask_i).sum()
#                     if with_reg:
#                         reg_i = -2 * learning_rate * reg_coef * U[u, k]
#                     else:
#                         reg_i = 0
#                     U_new[u, k] = U[u, k] + 2 * learning_rate * calc_i + reg_i
#
#             for i in range(0, n_item):
#                 for k in range(0, lr):
#                     mask_u = rate_mat[:, i] > 0
#                     calc_u = ((rate_mat[:, i] - np.sum(U[:, :] * V[i, :], axis=1)) * U[:, k] * mask_u).sum()
#                     if with_reg:
#                         reg_u = -2 * learning_rate * reg_coef * V[i, k]
#                     else:
#                         reg_u = 0
#                     V_new[i, k] = V[i, k] + 2 * learning_rate * calc_u + reg_u
#
#             U = U_new.copy()
#             V = V_new.copy()
#
#             cnt = cnt + 1
#             # Calculcate RMSE and convergence check
#             last_error = error
#             error = rmse(U, V, rate_mat)
#             # print("Iteration #{}: RMSE Error is {}".format(cnt, error))
#             chg = abs( last_error - error)
#             if chg <= epsilon:
#                 break
#             if isNaN(error):
#                 break
#
#         test_error = rmse(U, V, test_mat)
#         if test_error < test_error_best:
#             test_error_best = test_error
#             print("New best test error! RMSE: {}".format(test_error_best))
#             U_best = U.copy()
#             V_best = V.copy()
#             params_best = [learning_rate, max_iter, epsilon, reg_coef]
#         else:
#             print("Better params exist, moving on...")
#
#     print("Best params! {} with test error of {}".format(params_best, test_error_best))
#     return U_best, V_best
#
#
#
# def my_recommender_SGD(rate_mat, lr, with_reg):
#     """
#
#     :param rate_mat:
#     :param lr:
#     :param with_reg:
#         boolean flag, set true for using regularization and false otherwise
#     :return:
#     """
#
#     # learning_rate = 0.0005
#     # max_iter = 200
#     # epsilon = 1e-5
#     # reg_coef = 0.0001
#     if with_reg:
#         reg_coef = 0.00065
#         epsilon = 1e-5
#         learning_rate = 0.5
#     else:
#         learning_rate = 0.5
#         reg_coef = 0.00
#         epsilon = 1e-6
#     max_iter = 500
#     n_user, n_item = rate_mat.shape[0], rate_mat.shape[1]
#
#     # Initialize U and V
#     U = np.random.rand(n_user, lr) / lr
#     V = np.random.rand(n_item, lr) / lr
#
#     # mask = rate_mat > 0
#     # lists = np.where(np.any(rate_mat > 0, axis=1))
#     lists = np.transpose((rate_mat > 0).nonzero())
#
#
#     # u_rand = np.random.uniform(0, n_user)
#     # v_rand = np.random.uniform(0, n_item)
#
#     U_new = U.copy()
#     V_new = V.copy()
#
#     error = 1000
#     cnt = 0
#     n_samples = 10
#     while cnt < max_iter:
#
#         # subset of samples instead of just one?
#         ind = np.random.random_integers(0, lists.shape[0])
#         uv = lists[ind]
#         u_rand = uv[0]
#         v_rand = uv[1]
#
#         rate_mat_rand = np.zeros((n_user, n_item))
#         rate_mat_rand[u_rand,v_rand] = rate_mat[u_rand, v_rand]
#
#         if with_reg:
#             reg_u = 2 * learning_rate * reg_coef * U[u_rand,:]
#             reg_v = 2 * learning_rate * reg_coef * V[v_rand, :]
#         else:
#             reg_u = 0
#             reg_v = 0
#
#         U_new[u_rand, :] = U[u_rand, :] + 2 * learning_rate * (rate_mat_rand[u_rand, v_rand] - U[u_rand, :] * np.transpose(V[v_rand, :])) * V[v_rand, :] - reg_u
#         V_new[v_rand, :] = V[v_rand, :] + 2 * learning_rate * (np.transpose(rate_mat_rand[u_rand, v_rand]) - V[v_rand, :] * np.transpose(U[u_rand, :])) * U[u_rand, :] - reg_v
#         # U_new[u_rand, :] = U[u_rand, :] + 2 * learning_rate * (rate_mat_rand[u_rand, v_rand]) * V[v_rand, :] - reg_u
#         # V_new[v_rand, :] = V[v_rand, :] + 2 * learning_rate * (np.transpose(rate_mat_rand[u_rand, v_rand])) * U[u_rand, :] - reg_v
#
#
#         # for u in range(0,n_user):
#         #     for k in range(0,lr):
#         #         # mask_i = rate_mat[u, :] > 0
#         #         calc_i = ( (rate_mat[u_rand, v_rand] - np.sum(U[u_rand,:]*V[v_rand,:], axis=1)) * V[v_rand, k]).sum()
#         #         # calc_i = ( (rate_mat[u, :] - np.sum(U[u,:]*V[:,:], axis=1) ) * V[:,k] * mask_i).sum()
#         #         if with_reg:
#         #             reg_i = -2 * learning_rate * reg_coef * U[u,k]
#         #         else:
#         #             reg_i = 0
#         #         U_new[u,k] = U[u,k] + 2 * learning_rate * calc_i + reg_i
#         #
#         # for i in range(0, n_item):
#         #     for k in range(0,lr):
#         #         mask_u = rate_mat[:,i] > 0
#         #         calc_u = ( (rate_mat[:,i] - np.sum( U[:,:]*V[i,:], axis=1) ) * U[:,k] * mask_u).sum()
#         #         if with_reg:
#         #             reg_u = -2 * learning_rate * reg_coef * V[i,k]
#         #         else:
#         #             reg_u = 0
#         #         V_new[i,k] = V[i,k] + 2 * learning_rate * calc_u + reg_u
#
#         U = U_new.copy()
#         V = V_new.copy()
#
#         cnt = cnt + 1
#         # Calculcate RMSE and convergence check
#         last_error = error
#         error = rmse(U, V, rate_mat)
#         print("Iteration #{}: RMSE Error is {}".format(cnt, error))
#         chg = abs( last_error - error)
#         if chg <= epsilon:
#             break
#     return U, V


def isNaN(num):
    return num != num