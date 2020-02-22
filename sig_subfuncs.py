import numpy as np

def mat_squeeze(mat, idx, axis):
	r,c = mat.shape
	zc_cnt = 0
	if axis == 1:
		tmp = r
		r = c 
		c = tmp
	
	for k in range(0,r):
		if idx[k] == 0:
			for l in range(k+1, r):
				if idx[l] != 0:
					mat[k,:] = mat[l,:]
			zc_cnt = zc_cnt + 1
	nzc_num = r - zc_cnt
	return mat, nzc_num


def get_mix_weight_sgd(Y_feat, X_est, D_est, iter, learning_rate):
	w_x = 1.0
	w_d = 1.0
	for i in range(iter):
		w_x, w_d = step_gradient(w_x, w_d, Y_feat, X_est, D_est, learning_rate)

	return w_x, w_d

def step_gradient(w_x, w_d, Y_feat, X_est, D_est, learning_rate):
	N = len(Y_feat)
	w_x_grad = 2/N * np.sum(w_x * X_est * X_est + \
							w_d * X_est * D_est - \
							Y_feat * X_est)
	w_d_grad = 2/N * np.sum(w_d * D_est * D_est + \
							w_x * X_est * D_est - \
							Y_feat * D_est)            

	new_w_x = w_x - learning_rate * w_x_grad
	new_w_d = w_d - learning_rate * w_d_grad
	
	return [new_w_x, new_w_d]

def ReLU(x):
    return abs(x) * (x > 0.000001)

#----spectro-temporal sparsity measurement
def st_sparsity(X_hat, D_hat, r_blk, theta_r, K, K_q, I_q, gamma):

	r = np.maximum(X_hat / D_hat, theta_r)
	#r = 10*(np.log10(X_hat + 0.00001) - np.log10(D_hat + 0.00001))
	#r += np.abs(np.min(r))
	r /= np.max(r)
	r_blk[0:I_q-1, :] = r_blk[1:I_q, :]
	r_blk[I_q-1, :] = np.reshape(r, (K,))
		
	K_q2 = int(K_q * 0.5)
	N = K_q * I_q
	q = np.zeros((K,1))
	for k in range(K_q2, K - K_q2):
		b =np.reshape(r_blk[:, k-K_q2:k+K_q2], (N,1))
		l1 = np.sum(b)
		l2 = np.sqrt(np.sum(b**2))
		s = 1 / (np.sqrt(N) - 1) * (np.sqrt(N) - (l1 / l2))
		q[k] = 1 / (1 + np.exp(-1 * (s - gamma)))
		#print(s)

	return (q, r_blk)