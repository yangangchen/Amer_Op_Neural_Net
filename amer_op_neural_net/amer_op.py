'''
The American Option object for the American Option pricing and hedging algorithm
'''

from .config import *
from .options import levelset_function, payoff_function
from .myplot import evaluate_scatter, evaluate_exercise_boundary


class SmartNumpyArray:
    def __init__(self, dims, eff_dims=None, evaluate_eff=False, evaluate_exact=False, dtype=np_floattype, const=0):
        self.values = np.zeros(dims, dtype=dtype)
        if const != 0:
            self.values[:] = const
        if evaluate_exact:
            self.exact = np.zeros(dims, dtype=dtype)
            if const != 0:
                self.exact[:] = const

        if evaluate_eff:
            self.eff = lambda: None
            self.eff.values = np.zeros(eff_dims, dtype=dtype)
            if const != 0:
                self.eff.values[:] = const
            if evaluate_exact:
                self.eff.exact = np.zeros(eff_dims, dtype=dtype)
                if const != 0:
                    self.eff.exact[:] = const
        else:
            self.eff = None

class AmericanOption:
    def __init__(self, sample_size, evaluate_exact=False):
        '''
        American option simulation data, including
        self.X: underlying asset prices
        self.P: payoff of the underlying asset prices
        self.YLabel: discount payoff price
        self.control: binary exercise control
        self.Y: neural network trained price
        self.gradY: neural network trained delta

        self.eff: (for geometric options only) the effective underlying asset prices / payoff / discount payoff price / trained delta ...
        '''
        assert sample_size % len(X0) == 0
        self.sample_size = sample_size
        self.is_geometric = (option_type[1] == "geometric")
        self.eff_d = 1 if self.is_geometric else d
        self.evaluate_exact = evaluate_exact and self.eff_d == 1

        print("\n AmericanOption, initialization, phase 1 : ")
        self.X = SmartNumpyArray(dims=(self.sample_size, N + 1, d), eff_dims=(self.sample_size, N + 1, 1),
                                 evaluate_eff=self.is_geometric)
        self.P = SmartNumpyArray(dims=(self.sample_size, N + 1, 1))
        self.gradP = SmartNumpyArray(dims=(self.sample_size, N + 1, d, 1), eff_dims=(self.sample_size, N + 1, 1, 1),
                                     evaluate_eff=self.is_geometric)

        self.evolution_state()

        print("\n AmericanOption, initialization, phase 2 : ")
        self.control = SmartNumpyArray(dims=(self.sample_size, N + 1, 1), evaluate_exact=self.evaluate_exact, dtype=bool, const=True)
        self.YLabel = SmartNumpyArray(dims=(self.sample_size, N + 1, 1))
        self.Y = SmartNumpyArray(dims=(self.sample_size, N + 1, 1), evaluate_exact=self.evaluate_exact)
        self.gradY = SmartNumpyArray(dims=(self.sample_size, N + 1, d, 1), eff_dims=(self.sample_size, N + 1, 1, 1),
                                     evaluate_exact=self.evaluate_exact, evaluate_eff=self.is_geometric)
        self.Y0 = SmartNumpyArray(dims=1, evaluate_exact=self.evaluate_exact)
        self.gradY0 = SmartNumpyArray(dims=d, evaluate_exact=self.evaluate_exact)
        self.stop_index = SmartNumpyArray(dims=self.sample_size, evaluate_exact=self.evaluate_exact, dtype=int)
        self.stop_X = SmartNumpyArray(dims=(self.sample_size, d), evaluate_exact=self.evaluate_exact)
        self.hedge_error = lambda: None
        self.hedge_error.X_gradY = SmartNumpyArray(dims=(self.sample_size, N + 1, 1), evaluate_exact=self.evaluate_exact)
        self.hedge_error.X_gradY_pre = SmartNumpyArray(dims=(self.sample_size, N + 1, 1), evaluate_exact=self.evaluate_exact)

        self.evaluate_at_terminal_timestep()
        self.update_stop(n="all")
        
        ## Read the pre-stored exact values for error analysis
        if self.evaluate_exact:
            self.Y.exact[:] = 0  # Please fill in the correct values computed by finite difference
            self.gradY.exact[:] = 0  # Please fill in the correct values computed by finite difference
            if self.is_geometric:
                self.gradY.eff.exact[:] = 0  # Please fill in the correct values computed by finite difference
            self.control.exact[:] = 0  # Please fill in the correct values computed by finite difference
            self.stop_index.exact[:] = 0  # Please fill in the correct values computed by finite difference
            self.stop_X.exact[:] = 0  # Please fill in the correct values computed by finite difference

    def __evolution_state_timestep(self, Xn):
        dWn = (np.random.randn(self.sample_size, d).dot(rhoL) * math.sqrt(dt)).astype(np_floattype)
        dAn = sigma * Xn * dWn
        # Xnp1 = Xn + mu * Xn * dt + dAn
        Xnp1 = np.exp(mu * dt) * Xn + dAn
        Xnp1 = np.maximum(Xnp1, 1e-6)
        return Xnp1

    def evolution_state(self):
        init_t = time.time()
        print("   ", end="", flush=True)
        for n in range(N + 1):
            t0 = time.time()
            if AmerOp_seeds is not None:
                np.random.seed(AmerOp_seeds[n])
            if n == 0:
                self.X.values[:, n] = np.tile(X0, (self.sample_size // len(X0), 1))
            else:
                self.X.values[:, n] = self.__evolution_state_timestep(self.X.values[:, n - 1])
            print(str(n) + " (" + "{0:.2f}".format(time.time() - t0) + ", ", end="", flush=True)

            Ln, gradLn = levelset_function(self.X.values[:, n], eval_grad=True)
            self.P.values[:, n], self.gradP.values[:, n] = payoff_function(Ln, gradLn)

            if self.is_geometric:
                if option_type[0] == "call":
                    self.X.eff.values[:, n] = Ln + K
                elif option_type[0] == "put":
                    self.X.eff.values[:, n] = K - Ln
                self.gradP.eff.values[:, n, :, 0] = np.sum(self.X.values[:, n] * self.gradP.values[:, n, :, 0], axis=-1, keepdims=True) / self.X.eff.values[:, n]
            print("{0:.2f}".format(time.time() - t0) + "), ", end="", flush=True)
        print("\n evolution_state, time: ", "{0:.2f}".format(time.time() - init_t))

    '''
    Stopping boundary (exercise boundary)
    '''

    def __update_stop_timestep(self, n, which="values"):
        assert which in ["values", "exact"]
        if which == "values":
            control, stop_index, stop_X = self.control.values, self.stop_index.values, self.stop_X.values
        if which == "exact":
            control, stop_index, stop_X = self.control.exact, self.stop_index.exact, self.stop_X.exact

        if n == N:
            stop_index[:] = N
            stop_X[:] = self.X.values[:, N]
        else:
            valid_index = (control[:, n, 0] == 0)
            if valid_index.sum() > 0:
                stop_index[valid_index] = n
                stop_X[valid_index] = self.X.values[valid_index, n]

    def update_stop(self, n="all", which="values"):
        if n == "all":
            for n in reversed(range(N + 1)):
                self.__update_stop_timestep(n, which)
        else:
            self.__update_stop_timestep(n, which)

    '''
    Control, price, delta
    '''

    def evaluate_at_terminal_timestep(self, which="values"):
        assert which in ["values", "exact"]
        if which == "values":
            control, Y, gradY = self.control.values, self.Y.values, self.gradY.values
        if which == "exact":
            control, Y, gradY = self.control.exact, self.Y.exact, self.gradY.exact

        init_t = time.time()
        control[:, -1, :] = ~opregion_function(self.P.values[:, -1, :])
        self.YLabel.values[:, -1, :] = self.P.values[:, -1, :]

        print("   ", end="", flush=True)
        for n in range(N + 1):
            t0 = time.time()
            sLn, sgradLn = levelset_function(self.X.values[:, n], eval_grad=True, sharpness=sharpness)
            Y[:, n], gradY[:, n] = payoff_function(sLn, sgradLn, sharpness=sharpness)
            print(str(n) + " (" + "{0:.2f}".format(time.time() - t0) + "), ", end="", flush=True)

        self.update_hedging(n=N, which="values")
        print("\n initialize_results, time: ", "{0:.2f}".format(time.time() - init_t))

    def initialize_price_timestep(self, n):
        if n == N:
            self.YLabel.values[:, -1, :] = self.P.values[:, -1, :]
        else:
            self.YLabel.values[:, n, :] = self.YLabel.values[:, n + 1, :] * np.exp(-r * dt)

    def update_price_timestep(self, n, nu=0):
        if n == N:
            self.YLabel.values[:, n, :] = self.P.values[:, n, :]
        else:
            index = (self.control.values[:, n, 0] == 1)
            self.YLabel.values[~index, n, :] = self.P.values[~index, n, :]
            if nu == 0:
                self.YLabel.values[index, n, :] = self.YLabel.values[index, n + 1, :] * np.exp(-r * dt)
            else:
                self.YLabel.values[index, n, :] = nu * self.Y.values[index, n, :] \
                                           + (1 - nu) * self.YLabel.values[index, n + 1, :] * np.exp(-r * dt)
            # if nu == 0:
            #     self.YLabel.values[index, n, :] = np.exp(-r * (self.stop_index.values[index] - n) * dt) * self.stop_P.values[index]
            # else:
            #     self.YLabel.values[index, n, :] = nu * self.Y.values[index, n, :] + (1 - nu) * np.exp(
            #         -r * (self.stop_index.values[index] - n) * dt) * self.stop_P.values[index]

    def evaluate_control_timestep(self, n, which="values", control_tol=0):
        assert which in ["values", "exact"]
        if which == "values":
            control, Y = self.control.values, self.Y.values
        if which == "exact":
            control, Y = self.control.exact, self.Y.exact

        control[:, n, :] = (Y[:, n, :] > (1 + control_tol) * self.P.values[:, n, :]) | (
                self.P.values[:, n, :] < payoff_tol)

    def evaluate_at_initial_timestep(self):
        #### Compute Y0 and gradY0 based on the latest raw data
        Y0 = self.YLabel.values[:, 0, 0].mean()
        omega = self.YLabel.values[:, 0, 0].std()
        dA0 = self.X.values[:, 1] - np.exp(mu * dt) * self.X.values[:, 0]
        gradY0 = np.linalg.lstsq(
            dA0, np.exp(r * dt) * (self.YLabel.values[:, 0, 0] - Y0), rcond=None)[0]
        print("\n************** Results (Neural Network) **************")
        print(" The point X0 to evaluate is: ", self.X.values[:, 0].mean(axis=0))
        print(" The price at t=0 is: ", "{0:.6f}".format(Y0),
              ";  95% CI: [",
              "{0:.6f}".format(Y0 - 1.96 * omega / math.sqrt(len(dA0))),
              "{0:.6f}".format(Y0 + 1.96 * omega / math.sqrt(len(dA0))), "]")
        print(" The delta at t=0 is: ", gradY0, "\n")

        self.Y.values[:, 0, 0] = Y0
        self.gradY.values[:, 0, :, 0] = np.tile(gradY0, (len(dA0), 1))

        self.update_hedging(n=0, which="values")

    '''
    Evaluate Results
    '''

    def evaluate_results(self, which="values"):
        assert which in ["values", "exact"]
        if which == "values":
            title = "Simulation values"
            control, stop_index, stop_X, Y0, gradY0 = \
                self.control.values, self.stop_index.values, self.stop_X.values, self.Y0.values, self.gradY0.values
        if which == "exact":
            title = "Exact solution"
            control, stop_index, stop_X, Y0, gradY0 = \
                self.control.exact, self.stop_index.exact, self.stop_X.exact, self.Y0.exact, self.gradY0.exact

        initial_X = self.X.values[:, 0]
        stop_time = np.expand_dims(stop_index.astype(np_floattype) * dt, axis=-1)
        stop_P, stop_gradP = payoff_function(*levelset_function(stop_X, eval_grad=True))
        #### Compute Y0 and gradY0 based on the latest MC data
        YLabel = np.exp(-r * stop_time) * stop_P
        Y0[:] = YLabel.mean()
        omega = YLabel.std()
        gradYLabel = np.exp(-r * stop_time) * stop_gradP[:, :, 0] * stop_X / initial_X
        gradY0[:] = gradYLabel.mean(axis=0)
        print("\n************** Results (" + title + ") **************")
        print(" The point X0 to evaluate is: ", initial_X.mean(axis=0))
        print(" The price at t=0 is: ", "{0:.6f}".format(Y0[0]),
              ";  95% CI: [",
              "{0:.6f}".format(Y0[0] - 1.96 * omega / math.sqrt(len(YLabel))),
              "{0:.6f}".format(Y0[0] + 1.96 * omega / math.sqrt(len(YLabel))), "]")
        print(" The delta at t=0 is: ", gradY0, "\n")

    def evaluate_results_final(self, nu=0):
        print("\n\n************** Evaluation of the Entire Model, nu = ", str(nu), " **************")
        for n in reversed(range(N + 1)):
            self.initialize_price_timestep(n)
            if nu != 0 and n == 0:
                pass  ## Do not use Y and gradY at n = 0
            else:
                # self.evaluate_control_timestep(n, control_tol=control_tol)  ## Only if self.control, self.Y or self.gradY changes
                self.update_price_timestep(n, nu=nu)
                self.update_stop(n=n)

        self.evaluate_results(which="values")

    def evaluate_quality(self, n, title="neural network", plot_Y=True):
        print("\n Evaluate control (exercise boundary) : ")
        X = self.X.eff.values if self.is_geometric else self.X.values
        X_select, control_select = X[::(n_totalstep * num_channels)], self.control.values[::(n_totalstep * num_channels)]
        control_exact_select = self.control.exact[::(n_totalstep * num_channels)] if self.evaluate_exact else None
        evaluate_exercise_boundary(X_select, control_select, control_exact_select, self.eff_d, title, n)

        if self.evaluate_exact:
            print("   Confusion matrix: ",
                  "{:7d}".format(
                      ((self.control.exact[:, n, 0] == 0) & (self.control.values[:, n, 0] == 0)).sum()),
                  "{:7d}".format(
                      ((self.control.exact[:, n, 0] == 0) & (self.control.values[:, n, 0] == 1)).sum()),
                  "\n                     ",
                  "{:7d}".format(
                      ((self.control.exact[:, n, 0] == 1) & (self.control.values[:, n, 0] == 0)).sum()),
                  "{:7d}".format(
                      ((self.control.exact[:, n, 0] == 1) & (self.control.values[:, n, 0] == 1)).sum()))
        else:
            print("  Number of in-the-money points : ", (1 - self.control.values[:, n, :]).sum())

        if plot_Y and self.eff_d == 1:
            print("\n Evaluate Y and gradY, timestep-wise : ")
            gradY = self.gradY.eff.values if self.is_geometric else self.gradY.values
            Xn_select, controln_select, Yn_select, gradYn_select = \
                X[::n_totalstep, n, :], self.control.values[::n_totalstep, n, :], \
                self.Y.values[::n_totalstep, n, :], gradY[::n_totalstep, n, 0, :]
            if self.evaluate_exact:
                gradY_exact = self.gradY.eff.exact if self.is_geometric else self.gradY.exact
                Yexactn_select, gradYexactn_select = \
                    self.Y.exact[::n_totalstep, n, :], gradY_exact[::n_totalstep, n, 0, :]
            else:
                Yexactn_select, gradYexactn_select = None, None

            if d == 1:
                axeslabels = ["underlying asset (s)", "price (v)"]
            else:
                axeslabels = ["geometric average of underlying assets (s')", "price (v)"]
            evaluate_scatter(X=Xn_select, Y=Yn_select, Yexact=Yexactn_select,
                             YLabel=None, c=controln_select,
                             title="price at t =" + "{0:.3f}".format(n * dt),
                             axeslabels=axeslabels, eff_d=self.eff_d, Yscaling=True)
            if d == 1:
                axeslabels = ["underlying asset (s)", "delta (dv/ds)"]
            else:
                axeslabels = ["geometric average of underlying assets (s')", "delta (dv/ds')"]
            evaluate_scatter(X=Xn_select, Y=gradYn_select, Yexact=gradYexactn_select,
                             YLabel=None, c=controln_select,
                             title="delta at t =" + "{0:.3f}".format(n * dt),
                             axeslabels=axeslabels, eff_d=self.eff_d, Yscaling=False)

    '''
    Hedging
    '''

    def __update_hedging_timestep(self, n, which="values"):
        assert which in ["values", "exact"]
        if which == "values":
            X, gradY, X_gradY, X_gradY_pre = \
                self.X.values, self.gradY.values, self.hedge_error.X_gradY.values, self.hedge_error.X_gradY_pre.values
        if which == "exact":
            X, gradY, X_gradY, X_gradY_pre = \
                self.X.values, self.gradY.exact, self.hedge_error.X_gradY.exact, self.hedge_error.X_gradY_pre.exact

        X_gradY[:, n] = np.sum(X[:, n] * gradY[:, n, :, 0], axis=-1, keepdims=True)
        if n < N:
            X_gradY_pre[:, n + 1] = np.sum(X[:, n + 1] * gradY[:, n, :, 0] * np.exp(qq * dt), axis=-1, keepdims=True)

        if self.is_geometric:
            assert which == "values"
            self.gradY.eff.values[:, n, :, 0] = X_gradY[:, n] / self.X.eff.values[:, n]

    def update_hedging(self, n="all", which="values"):
        if n == "all":
            for n in range(N + 1):
                self.__update_hedging_timestep(n, which)
        else:
            self.__update_hedging_timestep(n, which)

    def delta_hedging(self, which="values"):
        assert which in ["values", "exact"]
        if which == "values":
            title = "Simulation values"
            control, stop_index, Y, X_gradY, X_gradY_pre = \
                self.control.values, self.stop_index.values, self.Y.values, self.hedge_error.X_gradY.values, self.hedge_error.X_gradY_pre.values
        if which == "exact":
            title = "Exact solution"
            control, stop_index, Y, X_gradY, X_gradY_pre = \
                self.control.exact, self.stop_index.exact, self.Y.exact, self.hedge_error.X_gradY.exact, self.hedge_error.X_gradY_pre.exact

        V = Y[:, 0, :].copy()
        alphaS = X_gradY[:, 0, :].copy()
        B = V - alphaS

        for n in range(1, N + 1):
            # index = np.arange(self.sample_size, dtype=int)
            index = stop_index >= n
            V[index] = Y[index, n, :]
            alphaS[index] = X_gradY[index, n, :]
            alphaS_pre = X_gradY_pre[index, n, :]
            B[index] = math.exp(r * dt) * B[index] - alphaS[index] + alphaS_pre
        Pi = -V + alphaS + B
        stop_time = np.expand_dims(stop_index.astype(np_floattype) * dt, axis=-1)
        self.hedge_error.values = np.exp(-r * stop_time) * Pi / Y[:, 0, :]

        print("\n\n************** Hedging Results (" + title + ") **************\n")
        print("\n P & L mean: ", self.hedge_error.values.mean(), ", std: ", self.hedge_error.values.std())

    def evaluate_delta_hedging(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.gca()
        ax.hist(self.hedge_error.values[:, 0], bins=200, range=(-1, 1), density=True, histtype='step', color="b", linewidth=3)
        if self.evaluate_exact and not self.is_geometric:
            ax.hist(self.hedge_error_exact.values[:, 0], bins=200, range=(-1, 1), density=True, histtype='step', color="r", linewidth=3)
        if d == 1:
            plt.title("hedging error, " + str(d) + "d, " + option_type[0], fontsize=16)
        else:
            plt.title("hedging error, " + str(d) + "d, " + option_type[0] + ", " + option_type[1], fontsize=16)
        plt.xlabel("relative P & L", fontsize=16)
        plt.ylabel("density", fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)
        if self.evaluate_exact and not self.is_geometric:
            plt.legend(["proposed method", "exact (finite difference)"], fontsize=14)
        else:
            plt.legend(["proposed method"], fontsize=14)
        if savefig_mode:
            fig.savefig(directory + "hedge_error.pdf", bbox_inches='tight')
        plt.close(fig)
