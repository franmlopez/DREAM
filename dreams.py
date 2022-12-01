import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import erfi
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize, space
from math import pi
import argparse
import os
from datetime import datetime
from utils import *

class DREAMs():
    def __init__(self, congr=1, mu_a=0.7, mu_c=0.4, gamma=0.0009,
                 sigma=2, b=75, alpha=2, dt=1, tmax=1000, mu_r=300, sigma_r=30):
        self.congr=congr
        self.mu_a=mu_a
        self.mu_c=mu_c
        self.gamma=gamma
        self.sigma=sigma
        self.b=b
        self.alpha=alpha
        self.dt=dt
        self.tmax=tmax
        self.mu_r=mu_r
        self.sigma_r=sigma_r

    def expected(self):
        n_tsteps = int(self.tmax/self.dt)
        t = np.linspace(self.dt, self.tmax, n_tsteps)
        X_c = self.mu_c * t
        X_a = self.congr * self.mu_a * np.sqrt(pi/(2*self.mu_c*self.gamma)) * np.exp(-self.mu_c*self.gamma*t**2/2) * erfi(np.sqrt(self.mu_c*self.gamma/2)*t)
        X_s = X_c + X_a
        X_s = np.clip(X_s, a_min=-self.b, a_max=self.b)
        return t, X_c, X_a, X_s

    def trial(self):
        n_tsteps = int(self.tmax/self.dt)
        t = np.linspace(self.dt, self.tmax, n_tsteps)
        X_0 = np.random.beta(self.alpha, self.alpha, size=1)*2*self.b - self.b
        X_c = self.mu_c * t
        X_a = self.congr * self.mu_a * np.sqrt(pi/(2*self.mu_c*self.gamma)) * np.exp(-self.mu_c*self.gamma*t**2/2) * erfi(np.sqrt(self.mu_c*self.gamma/2)*t)
        X_s = X_0 + X_c + X_a + np.cumsum( self.sigma * np.sqrt(self.dt) * np.random.normal(size=len(t)) )
        idx = np.argmax(np.abs(X_s)>=self.b)
        if (idx>0) and (idx<n_tsteps-1):
            X_s[idx+1:] = None
        return X_c, X_a, X_s

    def multi_trial(self, N=10):
        X_s = np.zeros([N, int(self.tmax/self.dt)])
        for idx in range(N):
            _, _, X_s[idx,:] = self.trial()
        return X_s

    def trial_response(self):
        while True:
            _,_,X = self.trial()
            idx = np.argmax(np.abs(X)>=self.b)
            if idx>0:
                # Add residual duration
                tr = np.random.normal(loc=self.mu_r, scale=self.sigma_r)
                return idx*self.dt + tr, np.sign(X[idx])/2 + 0.5

    def multi_response(self, N=1000):
        times = np.zeros(N)
        responses = np.zeros(N)
        for idx in range(N):
            time, response = self.trial_response()
            times[idx] = time
            responses[idx] = response
        return times, responses


def dreams_to_fit(x, exp_caf_congr, exp_caf_incongr, exp_cdf_congr, exp_cdf_incongr,
                  dt=1, tmax=1000, n_sims=100000, plots=False, save_name=None):
    dreams_congr = DREAMs(
        mu_a=x[0],
        mu_c=x[1],
        gamma=x[2],
        sigma=x[3],
        b=x[4],
        alpha=x[5],
        mu_r=x[6],
        sigma_r=x[7],
        congr=1,
        dt=dt,
        tmax=tmax,
    )
    times, responses = dreams_congr.multi_response(N=n_sims)
    dreams_data_congr = np.stack((times, responses), axis=1)
    dreams_caf_congr = caf(dreams_data_congr)
    dreams_cdf_congr = cdf(dreams_data_congr)

    dreams_incongr = DREAMs(
        mu_a = x[0],
        mu_c=x[1],
        gamma=x[2],
        sigma=x[3],
        b=x[4],
        alpha=x[5],
        mu_r=x[6],
        sigma_r=x[7],
        congr=-1,
        dt=dt,
        tmax=tmax,
    )
    times, responses = dreams_incongr.multi_response(N=n_sims)
    dreams_data_incongr = np.stack((times, responses), axis=1)
    dreams_caf_incongr = caf(dreams_data_incongr)
    dreams_cdf_incongr = cdf(dreams_data_incongr)

    rmse_caf = np.sqrt(mean_squared_error(dreams_caf_congr, exp_caf_congr) + mean_squared_error(dreams_caf_incongr, exp_caf_incongr))
    rmse_cdf = np.sqrt(mean_squared_error(dreams_cdf_congr, exp_cdf_congr) + mean_squared_error(dreams_cdf_incongr, exp_cdf_incongr))
    
    weight_caf = 1 / (max(np.max(exp_caf_congr),np.max(exp_caf_incongr)) - min(np.min(exp_caf_congr),np.min(exp_caf_incongr)))
    weight_cdf = 2 / (max(np.max(exp_cdf_congr),np.max(exp_cdf_incongr)) - min(np.min(exp_cdf_congr),np.min(exp_cdf_incongr)))

    if plots==True:
        plot_all_fits(exp_caf_congr, exp_caf_incongr, dreams_caf_congr, dreams_caf_incongr,
                      exp_cdf_congr, exp_cdf_incongr, dreams_cdf_congr, dreams_cdf_incongr,
                      save_name=save_name)

    return rmse_caf*weight_caf + rmse_cdf*weight_cdf


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--fit', action='store_true')
    parser.add_argument('--manual', action='store_true')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--save_name', type=str, default='')
    parser.add_argument('--data_file', type=str, default=None)
    parser.add_argument('--params', nargs='+', type=float, default=None)
    parser.add_argument('--params_noise', type=float, default=0)
    parser.add_argument('--n_sims', type=int, default=100000)
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--n_repeats', type=int, default=10)
    parser.add_argument('--n_examples', type=int, default=5)
    parser.add_argument('--tmax', type=int, default=1000)
    parser.add_argument('--dt', type=float, default=1.0)
    parser.add_argument('--tol', type=float, default=1e-4)
    args = parser.parse_args()
    save_name = args.save_name
    data_file = args.data_file
    params = np.array(args.params)
    params_noise = args.params_noise
    n_sims = args.n_sims
    n_iter = args.n_iter
    n_repeats = args.n_repeats
    n_examples = args.n_examples
    tmax = args.tmax
    dt = args.dt
    tol = args.tol

    if not os.path.exists('results/'+save_name):
        os.makedirs('results/'+save_name)

    if args.run:

        mu_a = params[0]
        mu_c=params[1]
        gamma=params[2]
        sigma=params[3]
        b=params[4]
        alpha=params[5]
        mu_r=params[6]
        sigma_r=params[7]

        dreams = DREAMs(congr=1, mu_a=mu_a, mu_c=mu_c, gamma=gamma, sigma=sigma, 
                        b=b, alpha=alpha, mu_r=mu_r, sigma_r=sigma_r)
        t, expected_X_c_congr, expected_X_a_congr, expected_X_s_congr = dreams.expected()
        multi_X_s_congr = dreams.multi_trial(N=n_examples)

        times,responses = dreams.multi_response(N=n_sims)
        dreams_data_congr = np.stack((times, responses), axis=1)
        cdf_congr = cdf(dreams_data_congr, bins=10)
        caf_congr = caf(dreams_data_congr, bins=10)

        dreams = DREAMs(congr=-1, mu_a=mu_a, mu_c=mu_c, gamma=gamma, sigma=sigma, 
                        b=b, alpha=alpha, mu_r=mu_r, sigma_r=sigma_r)
        t, expected_X_c_incongr, expected_X_a_incongr, expected_X_s_incongr = dreams.expected()
        multi_X_s_incongr = dreams.multi_trial(N=n_examples)

        times,responses = dreams.multi_response(N=n_sims)
        dreams_data_incongr = np.stack((times, responses), axis=1)
        cdf_incongr = cdf(dreams_data_incongr, bins=10)
        caf_incongr = caf(dreams_data_incongr, bins=10)

        plot_activations(t, expected_X_c_congr, expected_X_a_congr, expected_X_s_congr,
                        expected_X_a_incongr, expected_X_s_incongr,
                        multi_X_s_congr, multi_X_s_incongr,
                        save_name=save_name+'/run_activations')

        plot_all_sim(caf_congr, caf_incongr, cdf_congr, cdf_incongr,
                     save_name=save_name+'/run_statistics')
        

    if args.manual:

        # Load and analyse experimental data
        exp_data = np.genfromtxt(data_file, delimiter=',', skip_header=1)

        exp_data_congr = exp_data[exp_data[:,2]==1, :]
        exp_rt_congr = exp_data_congr[:,3]
        exp_rs_congr = exp_data_congr[:,4]
        exp_data_congr = np.stack((exp_rt_congr,exp_rs_congr), axis=1)
        exp_caf_congr = caf(exp_data_congr)
        exp_cdf_congr = cdf(exp_data_congr)

        exp_data_incongr = exp_data[exp_data[:,2]==2, :]
        exp_rt_incongr = exp_data_incongr[:,3]
        exp_rs_incongr = exp_data_incongr[:,4]
        exp_data_incongr = np.stack((exp_rt_incongr,exp_rs_incongr), axis=1)
        exp_caf_incongr = caf(exp_data_incongr)
        exp_cdf_incongr = cdf(exp_data_incongr)

        rmse = dreams_to_fit(params, exp_caf_congr, exp_caf_incongr, exp_cdf_congr, exp_cdf_incongr,
                     dt=dt, tmax=tmax, n_sims=n_sims, plots=True, save_name=save_name+'/manual_search')

        print('Manual parameter search with RMSE value ', rmse)
    
    if args.auto:

        # Load and analyse experimental data
        exp_data = np.genfromtxt(data_file, delimiter=',', skip_header=1)

        exp_data_congr = exp_data[exp_data[:,2]==1, :]
        exp_rt_congr = exp_data_congr[:,3]
        exp_rs_congr = exp_data_congr[:,4]
        exp_data_congr = np.stack((exp_rt_congr,exp_rs_congr), axis=1)
        exp_caf_congr = caf(exp_data_congr)
        exp_cdf_congr = cdf(exp_data_congr)

        exp_data_incongr = exp_data[exp_data[:,2]==2, :]
        exp_rt_incongr = exp_data_incongr[:,3]
        exp_rs_incongr = exp_data_incongr[:,4]
        exp_data_incongr = np.stack((exp_rt_incongr,exp_rs_incongr), axis=1)
        exp_caf_incongr = caf(exp_data_incongr)
        exp_cdf_incongr = cdf(exp_data_incongr)

        # Auxiliary objective function
        def dreams_objective(x):
            return dreams_to_fit(x, exp_caf_congr, exp_caf_incongr, exp_cdf_congr, exp_cdf_incongr,
                                dt=dt, tmax=tmax, n_sims=n_sims, plots=False)

        print('Searching parameter estimates for DREAMs')

        bounds_space = [space.Real(params[i]*0.5, params[i]*2) for i in range(len(params))]
        dreams_res = gp_minimize(dreams_objective, bounds_space, n_calls=n_iter, verbose=False)
        
        print('Parameter search finished with lowest RMSE value ', dreams_res.fun)
        print('Parameters: ', dreams_res.x)

        with open('results/'+save_name+'/auto_search.txt', 'a') as f:
            print('\nParameter optimization performed at', datetime.now().strftime('%Y-%m-%d_%H:%M:%S'), file=f)
            print('rmse\tparams', file=f)
            print(str(dreams_res.fun)+'\t',dreams_res.x, file=f)

        dreams_to_fit(dreams_res.x, exp_caf_congr, exp_caf_incongr, exp_cdf_congr, exp_cdf_incongr,
                     dt=dt, tmax=tmax, n_sims=n_sims, plots=True, save_name=save_name+'/auto_search')


    if args.fit:

        # Load and analyse experimental data
        exp_data = np.genfromtxt(data_file, delimiter=',', skip_header=1)

        exp_data_congr = exp_data[exp_data[:,2]==1, :]
        exp_rt_congr = exp_data_congr[:,3]
        exp_rs_congr = exp_data_congr[:,4]
        exp_data_congr = np.stack((exp_rt_congr,exp_rs_congr), axis=1)
        exp_caf_congr = caf(exp_data_congr)
        exp_cdf_congr = cdf(exp_data_congr)

        exp_data_incongr = exp_data[exp_data[:,2]==2, :]
        exp_rt_incongr = exp_data_incongr[:,3]
        exp_rs_incongr = exp_data_incongr[:,4]
        exp_data_incongr = np.stack((exp_rt_incongr,exp_rs_incongr), axis=1)
        exp_caf_incongr = caf(exp_data_incongr)
        exp_cdf_incongr = cdf(exp_data_incongr)

        print('Fitting DREAMs to experimental data')

        with open('results/'+save_name+'/fit.txt', 'a') as f:
            print('\nParameter optimization starting at', datetime.now().strftime('%Y-%m-%d_%H:%M:%S'), file=f)
            print('iter\trmse\tparams', file=f)

        best_rmse = None
        best_params = None

        for idx in range(n_repeats):
            print('\nIteration '+ str(idx) + '...')

            x0 = params + params * np.random.normal(scale=params_noise, size=len(params))
            bounds = ((0,None),(0,None),(0,None),(0,None),(0,None),(1,None),(0,None),(0,None))

            dreams_res = minimize(
                        dreams_to_fit, x0, tol=tol, bounds=bounds, 
                        args=(exp_caf_congr, exp_caf_incongr, exp_cdf_congr, exp_cdf_incongr, dt, tmax, n_sims, False), 
                        method='Nelder-Mead', options={'maxiter':n_iter, 'disp':True, 'adaptive':True})

            if best_rmse == None or dreams_res.fun<best_rmse:
                best_rmse = dreams_res.fun
                best_params = dreams_res.x

            print('Lowest RMSE found: ', dreams_res.fun)
            print('Parameters: ', dreams_res.x)

            with open('results/'+save_name+'/fit.txt', 'a') as f:
                print('\n'+str(idx)+'\t'+str(dreams_res.fun)+'\t',dreams_res.x, file=f)

        dreams_to_fit(best_params, exp_caf_congr, exp_caf_incongr, exp_cdf_congr, exp_cdf_incongr,
                     dt=dt, tmax=tmax, n_sims=n_sims, plots=True, save_name=save_name+'/fit')