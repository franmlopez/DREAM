import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize, space
from math import pi
import argparse
import os
from datetime import datetime
from utils import *


class DREAM():
    def __init__(self, congr=1, mu_a=0.5, mu_c=0.48, gamma=0.0005,
                 sigma_c=2, sigma_a=2, b=100, alpha_c=3, alpha_a=3,
                 mu_r=300, sigma_r=30, dt=1, tmax=1000):
        self.congr=congr
        self.mu_a=mu_a
        self.mu_c=mu_c
        self.gamma=gamma
        self.sigma_c=sigma_c
        self.sigma_a=sigma_a
        self.b=b
        self.alpha_c=alpha_c
        self.alpha_a=alpha_a
        self.mu_r=mu_r
        self.sigma_r=sigma_r
        self.dt=dt
        self.tmax=tmax

    def expected(self):
        n_tsteps = int(self.tmax/self.dt)
        t = np.linspace(self.dt, self.tmax, n_tsteps)

        # controlled process
        X_c = self.mu_c * t
        #X_c = np.clip(X_c, a_min=-2*self.b, a_max=2*self.b)
        # automatic process
        X_a = np.zeros([n_tsteps])
        # superimpossed process
        X_s = np.zeros([n_tsteps])
        X_a[0] = 0
        X_s[0] = 0
        
        for idx in range(1, n_tsteps):
            X_a[idx] = X_a[idx-1] + self.congr*self.mu_a*self.dt - self.gamma * np.abs(X_c[idx-1]) * X_a[idx-1] * self.dt
            X_s[idx] = X_c[idx] + X_a[idx]
        X_s = np.clip(X_s, a_min=-self.b, a_max=self.b)
        return t, X_c, X_a, X_s

    def trial(self):
        n_tsteps = int(self.tmax/self.dt)
        t = np.linspace(self.dt, self.tmax, n_tsteps)

        # controlled process
        X_c_0 = np.random.beta(self.alpha_c, self.alpha_c, size=1)*2*self.b - self.b if self.alpha_c>0 else 0
        X_c = X_c_0 + self.mu_c * t + np.cumsum( self.sigma_c * np.sqrt(self.dt) * np.random.normal(size=len(t)) )
        # automatic process
        X_a = np.zeros([n_tsteps])
        # superimpossed process
        X_s = np.zeros([n_tsteps])
        X_a[0] = np.random.beta(self.alpha_a, self.alpha_a, size=1)*2*self.b - self.b if self.alpha_c>0 else 0
        X_s[0] = X_a[0] + X_c[0]
        
        for idx in range(1, n_tsteps):
            X_a[idx] = X_a[idx-1] + self.congr*self.mu_a*self.dt - self.gamma*np.abs(X_c[idx-1])*X_a[idx-1]*self.dt + self.sigma_a*np.sqrt(self.dt)*np.random.normal()
            X_s[idx] = X_c[idx] + X_a[idx]
            if ((X_s[idx] >= self.b) or (X_s[idx] <= -self.b)) and (idx < n_tsteps-1):
                X_c[idx+1:] = X_c[idx]
                X_a[idx+1:] = X_a[idx]
                X_s[idx+1:] = X_s[idx]
                break
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


def dream_to_fit(x, exp_caf_congr, exp_caf_incongr, exp_cdf_congr, exp_cdf_incongr,
                  dt=1, tmax=1000, n_sims=100000, plots=False, save_name=None):
    dream_congr = DREAM(
        mu_a = x[0],
        mu_c=x[1],
        gamma=x[2],
        sigma_a=x[3],
        sigma_c=x[4],
        b=x[5],
        alpha_a=x[6],
        alpha_c=x[7],
        mu_r=x[8],
        sigma_r=x[9],
        congr=1,
        dt=dt,
        tmax=tmax,
    )
    times, responses = dream_congr.multi_response(N=n_sims)
    dream_data_congr = np.stack((times, responses), axis=1)
    dream_caf_congr = caf(dream_data_congr)
    dream_cdf_congr = cdf(dream_data_congr)

    dream_incongr = DREAM(
        mu_a = x[0],
        mu_c=x[1],
        gamma=x[2],
        sigma_a=x[3],
        sigma_c=x[4],
        b=x[5],
        alpha_a=x[6],
        alpha_c=x[7],
        mu_r=x[8],
        sigma_r=x[9],
        congr=-1,
        dt=dt,
        tmax=tmax,
    )
    times, responses = dream_incongr.multi_response(N=n_sims)
    dream_data_incongr = np.stack((times, responses), axis=1)
    dream_caf_incongr = caf(dream_data_incongr)
    dream_cdf_incongr = cdf(dream_data_incongr)

    rmse_caf = np.sqrt(mean_squared_error(dream_caf_congr, exp_caf_congr) + mean_squared_error(dream_caf_incongr, exp_caf_incongr))
    rmse_cdf = np.sqrt(mean_squared_error(dream_cdf_congr, exp_cdf_congr) + mean_squared_error(dream_cdf_incongr, exp_cdf_incongr))
    
    weight_caf = 1 / (max(np.max(exp_caf_congr),np.max(exp_caf_incongr)) - min(np.min(exp_caf_congr),np.min(exp_caf_incongr)))
    weight_cdf = 2 / (max(np.max(exp_cdf_congr),np.max(exp_cdf_incongr)) - min(np.min(exp_cdf_congr),np.min(exp_cdf_incongr)))

    if plots==True:
        plot_all_fits(exp_caf_congr, exp_caf_incongr, dream_caf_congr, dream_caf_incongr,
                      exp_cdf_congr, exp_cdf_incongr, dream_cdf_congr, dream_cdf_incongr,
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
        sigma_a=params[3]
        sigma_c=params[4]
        b=params[5]
        alpha_a=params[6]
        alpha_c=params[7]
        mu_r=params[8]
        sigma_r=params[9]

        dream = DREAM(congr=1, mu_a=mu_a, mu_c=mu_c, gamma=gamma, sigma_c=sigma_c, sigma_a=sigma_a, 
              b=b, alpha_c=alpha_c, alpha_a=alpha_a, mu_r=mu_r, sigma_r=sigma_r)
        t, expected_X_c_congr, expected_X_a_congr, expected_X_s_congr = dream.expected()
        multi_X_s_congr = dream.multi_trial(N=n_examples)

        times,responses = dream.multi_response(N=n_sims)
        dream_data_congr = np.stack((times, responses), axis=1)
        cdf_congr = cdf(dream_data_congr, bins=10)
        caf_congr = caf(dream_data_congr, bins=10)

        dream = DREAM(congr=-1, mu_a=mu_a, mu_c=mu_c, gamma=gamma, sigma_c=sigma_c, sigma_a=sigma_a, 
                    b=b, alpha_c=alpha_c, alpha_a=alpha_a, mu_r=mu_r, sigma_r=sigma_r)
        t, expected_X_c_incongr, expected_X_a_incongr, expected_X_s_incongr = dream.expected()
        multi_X_s_incongr = dream.multi_trial(N=n_examples)

        times,responses = dream.multi_response(N=n_sims)
        dream_data_incongr = np.stack((times, responses), axis=1)
        cdf_incongr = cdf(dream_data_incongr, bins=10)
        caf_incongr = caf(dream_data_incongr, bins=10)

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

        rmse = dream_to_fit(params, exp_caf_congr, exp_caf_incongr, exp_cdf_congr, exp_cdf_incongr,
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
        def dream_objective(x):
            return dream_to_fit(x, exp_caf_congr, exp_caf_incongr, exp_cdf_congr, exp_cdf_incongr,
                                dt, tmax, n_sims, plots=False)

        print('Searching for parameter estimates for DREAM')

        bounds_space = [space.Real(params[i]*0.5, params[i]*2) for i in range(len(params))]
        dream_res = gp_minimize(dream_objective, bounds_space, n_calls=n_iter, verbose=False)
        
        print('Parameter search finished with lowest RMSE value ', dream_res.fun)
        print('Parameters: ', dream_res.x)

        with open('results/'+save_name+'/auto_search.txt', 'a') as f:
            print('\nParameter optimization performed at', datetime.now().strftime('%Y-%m-%d_%H:%M:%S'), file=f)
            print('rmse\tparams', file=f)
            print(str(dream_res.fun)+'\t', dream_res.x, file=f)

        dream_to_fit(dream_res.x, exp_caf_congr, exp_caf_incongr, exp_cdf_congr, exp_cdf_incongr,
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

            print('Fitting DREAM to experimental data')

            with open('results/'+save_name+'/fit.txt', 'a') as f:
                print('\nParameter optimization starting at', datetime.now().strftime('%Y-%m-%d_%H:%M:%S'), file=f)
                print('iter\trmse\tparams', file=f)

            best_rmse = None
            best_params = None

            for idx in range(n_repeats):
                print('\nIteration '+ str(idx) + '...')

                x0 = params + params * np.random.normal(scale=params_noise, size=len(params))
                bounds = ((0,None),(0,None),(0,None),(0,None),(0,None),(0,None),(1,None),(1,None),(0,None),(0,None))

                dream_res = minimize(
                            dream_to_fit, x0, tol=tol, bounds=bounds, 
                            args=(exp_caf_congr, exp_caf_incongr, exp_cdf_congr, exp_cdf_incongr, dt, tmax, n_sims, False), 
                            method='Nelder-Mead', options={'maxiter':n_iter, 'disp':True, 'adaptive':True})

                if best_rmse == None or dream_res.fun<best_rmse:
                    best_rmse = dream_res.fun
                    best_params = dream_res.x

                print('Lowest RMSE found: ', dream_res.fun)
                print('Parameters: ', dream_res.x)

                with open('results/'+save_name+'/fit.txt', 'a') as f:
                    print('\n'+str(idx)+'\t'+str(dream_res.fun)+'\t',dream_res.x, file=f)

            dream_to_fit(best_params, exp_caf_congr, exp_caf_incongr, exp_cdf_congr, exp_cdf_incongr,
                        dt=dt, tmax=tmax, n_sims=n_sims, plots=True, save_name=save_name+'/fit')