/**
 * Stochastic volatility model for S&P 500 log return data.
 */
model StochasticVolatility {
  param phi_v, sigma_v, mu_y, sigma_y;
  noise eta;
  state v;
  obs y;

  sub parameter {
    phi_v ~ uniform(0.0, 1.0);
    sigma_v ~ uniform(0.0, 1.0);
    mu_y ~ uniform(-0.1, 0.1);
    sigma_y ~ uniform(0.0, 1.0);
  }

  sub proposal_parameter {
    phi_v ~ truncated_normal(phi_v, 0.01, 0.0, 1.0);
    sigma_v ~ truncated_normal(sigma_v, 0.04, 0.0, 1.0);
    mu_y ~ truncated_normal(mu_y, 0.001, -0.1, 0.1);
    sigma_y ~ truncated_normal(sigma_y, 0.001, 0.0, 1.0);
  }

  sub initial {
    /* stationary distribution for v */
    v ~ normal(0.0, sqrt(sigma_v**2/(1.0 - phi_v**2)));
  }

  sub transition {
    eta ~ normal();
    v <- phi_v*v + sigma_v*eta;
  }

  sub observation {
    y ~ normal(mu_y, sigma_y*exp(0.5*v));
  }
}
