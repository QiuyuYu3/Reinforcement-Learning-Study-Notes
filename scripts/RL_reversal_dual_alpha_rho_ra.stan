data {
  int<lower=1> N;                       // Number of subjects
  int<lower=1> T;                       // Maximum number of trials across subjects
  int<lower=1, upper=T> Tsubj[N];       // Number of trials per subject
  int<lower=-1, upper=2> choice[N, T];  // Choices (1 or 2)
  real outcome[N, T];                    // Outcomes
}

transformed data {
  vector[2] initV;
  initV = rep_vector(0.0, 2);
}

parameters {
  // Hyperparameters
  vector[3] mu_pr;
  vector<lower=0>[3] sigma;

  // Individual-level raw parameters
  vector[N] Apun_pr;
  vector[N] Arew_pr;
  vector[N] beta_pr;

  vector[N] rho_rew_pr;
  vector[N] rho_pun_pr;
  vector[N] ra_rew1_pr;
  vector[N] ra_pun1_pr;
  vector[N] ra_rew2_pr;
  vector[N] ra_pun2_pr;
}

transformed parameters {
  vector<lower=0, upper=1>[N] Apun;
  vector<lower=0, upper=1>[N] Arew;
  vector<lower=0, upper=10>[N] beta;

  vector[N] rho_rew;
  vector[N] rho_pun;
  vector[N] ra_rew1;
  vector[N] ra_pun1;
  vector[N] ra_rew2;
  vector[N] ra_pun2;

  for (i in 1:N) {
    Apun[i]  = Phi_approx(mu_pr[1] + sigma[1] * Apun_pr[i]);
    Arew[i]  = Phi_approx(mu_pr[2] + sigma[2] * Arew_pr[i]);
    beta[i]  = Phi_approx(mu_pr[3] + sigma[3] * beta_pr[i]) * 10;

    rho_rew[i] = rho_rew_pr[i];
    rho_pun[i] = rho_pun_pr[i];
    ra_rew1[i] = ra_rew1_pr[i];
    ra_pun1[i] = ra_pun1_pr[i];
    ra_rew2[i] = ra_rew2_pr[i];
    ra_pun2[i] = ra_pun2_pr[i];
  }
}

model {
  // Group-level priors
  mu_pr  ~ normal(0,1);
  sigma ~ normal(0,0.2);

  // Individual-level priors
  Apun_pr ~ normal(0,1);
  Arew_pr ~ normal(0,1);
  beta_pr ~ normal(0,1);

  rho_rew_pr ~ normal(0,1);
  rho_pun_pr ~ normal(0,1);
  ra_rew1_pr ~ normal(0,1);
  ra_pun1_pr ~ normal(0,1);
  ra_rew2_pr ~ normal(0,1);
  ra_pun2_pr ~ normal(0,1);

  for (i in 1:N) {
    vector[2] ev;
    real PE;
    real out_mod;

    ev = initV;

    for (t in 1:Tsubj[i]) {
      choice[i, t] ~ categorical_logit(ev * beta[i]);

      if (outcome[i,t] > 0.5) 
          out_mod = rho_rew[i]*outcome[i,t] + ra_rew1[i];
      else if (outcome[i,t] < -0.5)
          out_mod = rho_pun[i]*outcome[i,t] + ra_pun1[i];
      else if (outcome[i,t] >= -0.5 && outcome[i,t] <= 0)
          out_mod = outcome[i,t] + ra_pun2[i];
      else
          out_mod = outcome[i,t] + ra_rew2[i];

      PE = out_mod - ev[choice[i,t]];

      if (outcome[i,t] > 0)
        ev[choice[i,t]] += Arew[i] * PE;
      else
        ev[choice[i,t]] += Apun[i] * PE;
    }
  }
}

generated quantities {
  real<lower=0, upper=1>  mu_Apun;
  real<lower=0, upper=1>  mu_Arew;
  real<lower=0, upper=10> mu_beta;

  real log_lik[N];

  // Trial-level regressors
  real ev_c[N, T];
  real ev_nc[N, T];
  real dq[N, T];  // ΔQ
  real pe[N, T];
  int y_pred[N, T];

  // Initialize
  for (i in 1:N) {
    log_lik[i] = 0;
    for (t in 1:T) {
      ev_c[i,t] = 0;
      ev_nc[i,t] = 0;
      dq[i,t] = 0;
      pe[i,t] = 0;
      y_pred[i,t] = 1;
    }
  }

  mu_Apun = Phi_approx(mu_pr[1]);
  mu_Arew = Phi_approx(mu_pr[2]);
  mu_beta = Phi_approx(mu_pr[3]) * 10;

  for (i in 1:N) {
    vector[2] ev;
    real PE;
    real out_mod;

    ev = initV;
    log_lik[i] = 0;

    for (t in 1:Tsubj[i]) {
      log_lik[i] += categorical_logit_lpmf(choice[i,t] | ev * beta[i]);
      y_pred[i,t] = categorical_rng(softmax(ev * beta[i]));

      if (outcome[i,t] > 0.5)
        out_mod = rho_rew[i]*outcome[i,t] + ra_rew1[i];
      else if (outcome[i,t] < -0.5)
        out_mod = rho_pun[i]*outcome[i,t] + ra_pun1[i];
      else if (outcome[i,t] >= -0.5 && outcome[i,t] <= 0)
        out_mod = outcome[i,t] + ra_pun2[i];
      else
        out_mod = outcome[i,t] + ra_rew2[i];

      PE = out_mod - ev[choice[i,t]];
      pe[i,t] = PE;

      ev_c[i,t] = ev[choice[i,t]];
      ev_nc[i,t] = ev[3 - choice[i,t]];
      dq[i,t] = ev_c[i,t] - ev_nc[i,t];

      if (outcome[i,t] > 0)
        ev[choice[i,t]] += Arew[i] * PE;
      else
        ev[choice[i,t]] += Apun[i] * PE;
    }
  }
}
