# Hawkes_pytorch
Pytorch Implementation of Hawkes Process
# Finished 
* Multivariate Hawkel process with single kernel for all events
* \lbda_i(t) = \mu_i + \sum_{t_k < t} \alpha_{i y_{t_k}} \sum_{j=1}^D \omega_{j} \exp{-\omega_j (t -t_k)} 

# Todo 
* Implement event forecasting (what type and what time)
* EM algorithm
* Multi-Kernel