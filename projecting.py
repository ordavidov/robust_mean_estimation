import numpy as np

def capped_simplex_projection(w: np.array, k: float=1., t: float=1.):
    r'''
    Find
    
    .. math::
        \min_{x \in \mathbb{R}^n} \frac{1}{2} \Vert x-w \Vert^2 \\
        \mbox{s.t.} x^T \cdot \mathbf{1} = k, \quad \mathbf{0} \leq x \leq t \cdot \mathbf{1}
    
    Algorithm by Weiran Wang and Canyi Lu in https://home.ttic.edu/~wwang5/.
           
            Parameters:
                    w (np.array): 1-dimensional array of size n. 
                    k (float): the sum constraint. Default is 1.
                    t (float): the upper bound constraint. Default is 1.                    
                    
            Returns:
                    The :math:'x' as a 1-dimensional np.array that minimizes the above.

    >>> capped_simplex_projection(np.array([1,1]))
    array([0.5, 0.5])
    >>> capped_simplex_projection(np.array([1,1/4,0]))
    array([0.875, 0.125, 0.   ])
    '''        
    
    n = w.flatten().size
    
    assert (k<n) and (k>0), 'The sum constraint (k) should be strictly positive and less than dim of y.'
    assert (t>0), 'The w upper bound (t) should be strictly positive.'
    
    x_min = np.zeros(n)
    w_input = w.astype(float).flatten()/t
    k_input = k/t
    
    idx = np.argsort(w_input)
    y = w_input[idx]
    s = np.cumsum(y)
    s = np.pad(s,(1,0))
    y = np.pad(y,1,constant_values=(-np.inf,np.inf))
    
    if k_input == round(k_input):
        b = n-int(k_input)
        if y[b+1]-y[b] >= 1:
            x_min[idx[b:]] = 1
            return x_min*t   
    
    for b in range(1,n+1):
        gamma = (k_input+b-n-s[b]) / b
        if ((y[1]+gamma)>0) and ((y[b]+gamma)<1) and ((y[b+1]+gamma)>=1):
            x_tmp=np.concatenate([y[1:b+1]+gamma, np.ones(n-b)])
            x_min[idx] = x_tmp
            return x_min*t

    for a in range(1,n+1):
        for b in range(a+1,n+1):
            gamma = (k_input+b-n+s[a]-s[b])/(b-a)
            if ((y[a]+gamma)<=0) and ((y[a+1]+gamma)>0) and ((y[b]+gamma)<1) and ((y[b+1]+gamma)>=1):
                x_tmp=np.concatenate([np.zeros(a), y[a+1:b+1]+gamma, np.ones(n-b)])
                x_min[idx]=x_tmp
                return x_min*t

def kullbeck_leibler_projection(w: np.array, alpha: float):
    r'''
    Find
    
    .. math::
        \min_{p \in \Delta_n} KL(p \Vert q) \\
        \mbox{s.t. } \mathbf{0} \leq p \leq \frac{\alpha}{n}
    
    Algorithm by Herbster and Warmuth in Tracking the Best Linear Predictor.
    https://dl.acm.org/doi/10.1162/153244301753683726
           
            Parameters:
                    w (np.array): 1-dimensional array of weights of size n. Must be non-negative and sum up to 1. 
                    alpha (float): the factor for the upper bound alpha/n. Must be >= 1.
                    
            Returns:
                    The :math:'x' as a 1-dimensional np.array that minimizes the above.

    >>> kullbeck_leibler_projection(np.array([0.56495992, 0.31297521, 0.12206487]),1.5)
    array([0.5       , 0.35970848, 0.14029152])
    '''            
    
    assert alpha>=1, 'alpha must be >= 1, otherwise, problem is unfeasible.'

    w_init = w.flatten()
    epsilon = 1e-8
    
    assert (np.abs(1-np.sum(w_init))<= epsilon), 'w does not sum up to 1. w must be a vector of weights.'
    assert np.all(w_init<=1) and np.all(w_init>=0), 'w must be in the hypercube [0,1]^n. w must be a vector of weights.'
    
    n = w_init.size
    
    if np.all(w_init<=alpha/n):
        
        return w_init
    
    W = np.array([*range(n)])
    c_size = 0
    c_sum = 0

    while W.size>0:
        
        w_median = np.median(w_init[W])

        L = np.intersect1d(np.where(w_init>w_median+epsilon)[0],W)
        M = np.intersect1d(np.where(np.abs(w_init-w_median)<=epsilon)[0],W)
        H = np.intersect1d(np.where(w_init<w_median-epsilon)[0],W)

        L_size = L.size
        L_sum = w_init[L].sum() if L_size>0 else 0
        M_size = M.size
        M_sum = w_init[M].sum() if M_size>0 else 0

        m_0 = (1 - (c_size+L_size)*alpha/n)/(1 - c_sum - L_sum)

        if m_0*w_median > alpha/n:
            c_size += L_size + M_size
            c_sum += L_sum + M_sum
            w_median = np.max(w_init[w_init<w_median-epsilon]) if H.size == 0 else w_median
            W = H
        else:
            W = L

    m_0 = (1 - (c_size)*alpha/n)/(1 - c_sum)
    v = np.where(w_init>w_median,alpha/n,w_init*m_0)

    assert np.abs(1-np.sum(v))<= epsilon, 'something went wrong... solution is unfeasible.'
    assert np.all(v<=alpha/n), 'something went wrong... solution is unfeasible.'

    return v

if __name__ == '__main__':
    
    p = capped_simplex_projection(np.array([1,1]))
    print('Projecting p = {} under L2 results in: {}'.format(np.array([1,1]),p))
    print('We expect the result to be: {}'.format(np.array([0.5,0.5])))

    p = capped_simplex_projection(np.array([1,1/4,0]))
    print('Projecting p = {} under L2 results in: {}'.format(np.array([1,1/4,0]),p))
    print('We expect the result to be: {}'.format(np.array([0.875,0.125,0.])))

    p = kullbeck_leibler_projection(np.array([0.56495992, 0.31297521, 0.12206487]),1.5)
    print('Projecting p = {} under KL divergence with alpha={} results in: {}'.format(np.array([0.56495992, 0.31297521, 0.12206487]),1.5,p))
    print('We expect the result to be: {}'.format(np.array([0.5,0.35970848,0.14029152])))