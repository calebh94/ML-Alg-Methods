import numpy as np

#TODO: REMOVE ALL PRINTS!
def rmse(u, v, mat):
    mask = mat > 0
    res = np.sum(((u.dot(v.T) - mat) * mask) ** 2) / float(np.sum(mask))
    return np.sqrt(res)



#TODO: my_recommender_SGD()

def my_recommender(rate_mat, lr, with_reg):
    # Select Parameters
    if with_reg:
        reg_coef = 0.0015
        epsilon = 1e-4
        learning_rate = 0.00045
    else:
        learning_rate = 0.00045
        reg_coef = 0.00
        epsilon = 1e-4
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
#     # learning_rate_arr = np.array([0.0005, 0.0004, 0.00045, 0.0003, 0.0002, 0.00015])
#     # max_iter_arr = np.array([200, 300, 500])
#     # epsilon_arr = np.array([1e-5, 1e-6, 5e-6])
#     reg_coef_arr = np.array([0.0015, 0.002, 0.0025])
#     n_user, n_item = rate_mat.shape[0], rate_mat.shape[1]
#     reg_coef = 0.005
#     max_iter = 300
#     epsilon = 1e-4
#     learning_rate = 0.00045
#
#     results = []
#     test_error_best = 1000
#     params_best = []
#     while (reg_coef_arr.size > 0):
#
#         # # Sample params
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
#         mask = rate_mat > 0
#
#         error = 1000
#         cnt = 0
#         while cnt < max_iter:
#
#             Vt = np.transpose(V)
#             Ut = np.transpose(U)
#             U_new = U + 2 * learning_rate * (
#                         rate_mat - np.multiply(U @ Vt, mask)) @ V - 2 * reg_coef * U
#             V_new = V + 2 * learning_rate * np.transpose(
#                 rate_mat - np.multiply(U @ Vt, mask)) @ U - 2 * reg_coef * V
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


def isNaN(num):
    return num != num