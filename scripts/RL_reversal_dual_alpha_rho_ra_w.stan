data {
  int<lower=1> N;                       // Number of subjects
  int<lower=1> T;                       // Maximum number of trials across subjects
  int<lower=1, upper=T> Tsubj[N];       // Number of trials/blocks for each subject
  int<lower=-1, upper=2> choice[N, T];  // The choices subjects made, code as 1 and 2
  real outcome[N, T];                   // The outcome
}

transformed data {
  // Default value for (re-)initializing parameter vectors
  vector[2] initV;
  initV = rep_vector(0.0, 2);
}

// Declare all parameters as vectors for vectorizing
parameters {
  // Hyper(group)-parameters
  vector[3] mu_pr;
  vector<lower=0>[3] sigma;

  // ---- group-level ----
  vector[7] mu_pr_extra;            // 7 parameters, rho, ra_*, and omega
  vector<lower=0>[7] sigma_extra;

  // Subject-level raw parameters (for Matt trick)
  vector[N] Apun_pr;
  vector[N] Arew_pr;
  vector[N] beta_pr;

  // ---- individual-level raw parameters ----
  vector[N] rho_rew_pr;
  vector[N] rho_pun_pr;
  vector[N] ra_rew1_pr;
  vector[N] ra_pun1_pr;
  vector[N] ra_rew2_pr;
  vector[N] ra_pun2_pr;
  vector[N] omega_pr;
}

transformed parameters {
  // Transform subject-level raw parameters
  vector<lower=0, upper=1>[N]  Apun;
  vector<lower=0, upper=1>[N]  Arew;
  vector<lower=0, upper=10>[N] beta;

  vector[N] rho_rew;
  vector[N] rho_pun;
  vector[N] ra_rew1;
  vector[N] ra_pun1;
  vector[N] ra_rew2;
  vector[N] ra_pun2;
  vector[N] omega;

  // Non-centered parameterization pr Matt trick
  for (i in 1:N) {
    Apun[i]  = Phi_approx(mu_pr[1] + sigma[1] * Apun_pr[i]);
    Arew[i]  = Phi_approx(mu_pr[2] + sigma[2] * Arew_pr[i]);
    beta[i]  = Phi_approx(mu_pr[3] + sigma[3] * beta_pr[i]) * 10;

    rho_rew[i] = mu_pr_extra[1] + sigma_extra[1] * rho_rew_pr[i];
    rho_pun[i] = mu_pr_extra[2] + sigma_extra[2] * rho_pun_pr[i];

    ra_rew1[i] = mu_pr_extra[3] + sigma_extra[3] * ra_rew1_pr[i];
    ra_pun1[i] = mu_pr_extra[4] + sigma_extra[4] * ra_pun1_pr[i];
    ra_rew2[i] = mu_pr_extra[5] + sigma_extra[5] * ra_rew2_pr[i];
    ra_pun2[i] = mu_pr_extra[6] + sigma_extra[6] * ra_pun2_pr[i];

    omega[i] = mu_pr_extra[7] + sigma_extra[7] * omega_pr[i];
  }
}


model {
  // --- group-level priors ---
  mu_pr ~ normal(0,1);
  sigma ~ normal(0,0.2);
  mu_pr_extra ~ normal(0,1);
  sigma_extra ~ normal(0,0.2);

  // --- individual-level priors ---
  Apun_pr  ~ normal(0,1);
  Arew_pr  ~ normal(0,1);
  beta_pr  ~ normal(0,1);

  rho_rew_pr ~ normal(0,1);
  rho_pun_pr ~ normal(0,1);
  ra_rew1_pr ~ normal(0,1);
  ra_pun1_pr ~ normal(0,1);
  ra_rew2_pr ~ normal(0,1);
  ra_pun2_pr ~ normal(0,1);
  omega_pr   ~ normal(0,1);

  for (i in 1:N) {
    // Expected values and prediction error
    vector[2] ev;   
    real PE;
    real out_mod;

    // Initialize expected values
    ev = initV;

    for (t in 1:Tsubj[i]) {
      // --- Choice perseveration ---
      vector[2] stickiness;
      if (t == 1)
        stickiness = rep_vector(0.0, 2); // no previous choice on first trial
      else {
        stickiness[1] = omega[i] * (choice[i, t-1] == 1 ? 1 : 0);
        stickiness[2] = omega[i] * (choice[i, t-1] == 2 ? 1 : 0);
      }

      // --- Softmax choice with stickiness ---
      choice[i, t] ~ categorical_logit(ev * beta[i] + stickiness);

      // --- Outcome modulation ---
      // outcome modified (outcome sensitivity rho, overall shift ra) only when out_gain > .5
      if (outcome[i, t] > 0.5) 
        out_mod = rho_rew[i] * outcome[i, t] + ra_rew1[i];
      else if (outcome[i, t] < -0.5) 
        out_mod = rho_pun[i] * outcome[i, t] + ra_pun1[i];
      else if (outcome[i, t] >= -0.5 && outcome[i, t] <= 0)
        out_mod = outcome[i, t] + ra_pun2[i];
      else
        out_mod = outcome[i, t] + ra_rew2[i];

      // --- Prediction error ---
      PE = out_mod - ev[choice[i, t]];

      // --- Update expected value ---
      if (outcome[i, t] > 0)
        ev[choice[i, t]] += Arew[i] * PE;
      else
        ev[choice[i, t]] += Apun[i] * PE;
    }
  }
}

generated quantities {
  // Interpretable group-level params
  real<lower=0, upper=1>  mu_Apun;
  real<lower=0, upper=1>  mu_Arew;
  real<lower=0, upper=10> mu_beta;

  // Log likelihood per subject
  real log_lik[N];

  // Trial-level regressors
  real ev_c[N, T];     // chosen Q-value
  real ev_nc[N, T];    // non-chosen Q-value
  real dq[N, T];       // ΔQ = Qchosen - Qnonchosen
  real pe[N, T];       // prediction error
  real stickiness_reg[N, T]; // stickiness regressor
  int  y_pred[N, T];   // posterior predictive

  // Initialize all variables to avoid NULL values
  for (i in 1:N) {
    for (t in 1:T) {
      ev_c[i,t]  = negative_infinity(); // clearly "invalid"
      pe[i,t]    = negative_infinity();
      dq[i,t]    = negative_infinity();
      stickiness_reg[i,t] = 0;
      y_pred[i,t] = 0;
    }
  }

  mu_Apun = Phi_approx(mu_pr[1]);
  mu_Arew = Phi_approx(mu_pr[2]);
  mu_beta = Phi_approx(mu_pr[3]) * 10;

  for (i in 1:N) {
    vector[2] ev;
    real PE;
    real out_mod;
    vector[2] stickiness;

    ev = rep_vector(0.0, 2);
    log_lik[i] = 0;

    for (t in 1:Tsubj[i]) {

      // stickiness term
      if (t == 1)
        stickiness = rep_vector(0.0, 2);
      else {
        stickiness[1] = omega[i] * (choice[i, t-1] == 1 ? 1 : 0);
        stickiness[2] = omega[i] * (choice[i, t-1] == 2 ? 1 : 0);
      }
      stickiness_reg[i,t] = stickiness[choice[i,t]];

      // log likelihood + posterior prediction
      log_lik[i] += categorical_logit_lpmf(choice[i,t] | ev * beta[i] + stickiness);
      y_pred[i,t] = categorical_rng(softmax(ev * beta[i] + stickiness));

      // outcome modulation
      if (outcome[i, t] > 0.5)
        out_mod = rho_rew[i] * outcome[i, t] + ra_rew1[i];
      else if (outcome[i, t] < -0.5)
        out_mod = rho_pun[i] * outcome[i, t] + ra_pun1[i];
      else if (outcome[i, t] >= -0.5 && outcome[i, t] <= 0)
        out_mod = outcome[i, t] + ra_pun2[i];
      else
        out_mod = outcome[i, t] + ra_rew2[i];

      // prediction error
      PE = out_mod - ev[choice[i,t]];
      pe[i,t] = PE;

      // store values
      ev_c[i,t] = ev[choice[i,t]];
      ev_nc[i,t] = ev[3 - choice[i,t]];
      dq[i,t] = ev_c[i,t] - ev_nc[i,t];

      // update Q value
      if (outcome[i,t] > 0)
        ev[choice[i,t]] += Arew[i] * PE;
      else
        ev[choice[i,t]] += Apun[i] * PE;
    }
  }
}

