library(MASS)

source('src/cot.R')
source('src/model.R')
source('src/utils.R')


compute_mc_stats_wass <- function(
    y_list,
    x,
    pipeline,
    V = length(y_list),   # default = leave-one-subject-out
    S = 2000,
    seed = 12345
){
  set.seed(seed)
  
  # Set parameters
  N <- length(y_list)
  p_grid <- pipeline$training$cache$p_grid
  supp_TY <- pipeline$training$cache$supp_TY
  y_trans_fun <- y_trans_to_fun[[pipeline$stages[[1]]$state$y_trans]]
  y_shift <- pipeline$stages[[1]]$state$y_shift
  
  if (length(x) != N) {
    stop("x must have one value per subject")
  }
  
  # Encode all subjects
  ctx <- new_context(
    payload = y_list,
    cache = pipeline$training$cache,
    meta = list()
  )
  ctx <- encode(pipeline, ctx, from = 0, to = 6)
  Z_all <- do.call(rbind, ctx$payload)
  K_ <- ncol(Z_all)
  
  # Subject folds
  fold_id <- sample(rep(1:V, length.out = N))
  
  T_i <- numeric(N)
  
  # ----- Loop over folds
  for(v in 1:V){
    print(str_glue("========== Fold {v} =========="))
    
    test_idx  <- which(fold_id == v)
    train_idx <- setdiff(1:N, test_idx)
    
    Z_train <- Z_all[train_idx,,drop=FALSE]
    x_train <- x[train_idx]
    
    # ----- Fit full model
    samps_full <- fit_model_bin_pred(
      Z_train, x_train,
      n_chains = 1,
      burn_in = 2000,
      thin_interval = 5,
      n_samps = S
    )
    
    # ----- Fit reduced model
    samps_red <- fit_model_mean(
      Z_train,
      n_chains = 1,
      burn_in = 2000,
      thin_interval = 5,
      n_samps = S
    )
    samps_red <- cbind(
      samps_red[,1:(K_**2)],
      matrix(0, nrow = S, ncol = K_), ## set betas = 0
      samps_red[,(K_**2+1):(K_**2+K_)]
    )
    
    ## ----- Loop over test subjects
    for(i in test_idx){
      print(str_glue("i = {i}"))
      
      Tyi <- y_trans_fun(y_list[[i]], y_shift)
      xi <- x[i]
      Ji <- length(Tyi)
      
      # helper: compute subject LPPD for one posterior sample matrix
      subject_wass <- function(samps){
        
        # Posterior predictive sampling
        Sigmas <- samps[,1:(K_^2)]
        betas  <- samps[,(K_^2+1):(K_^2+K_)]
        mus    <- samps[,(K_^2+K_+1):(K_^2+2*K_)]
        z_draws <- vector("list", S)
        for(s in seq_len(S)){
          z_draws[[s]] <- MASS::mvrnorm(
            1,
            mu = mus[s,] + betas[s,]*xi,
            Sigma = matrix(Sigmas[s,],K_)
          )
        }
        
        # Decode draws
        z_ctx <- new_context(
          payload = z_draws,
          cache = pipeline$training$cache,
          meta = list(Ji_vec = rep(Ji, length(z_draws)))
        )
        Qi_list <- decode(pipeline, z_ctx, from = 6, to = 2)$payload
        
        # Compute Wasserstein stats
        wass_vec <- matrix(0, S)
        for(s in seq_len(S)){
          wass_vec[s] <- sum((sort(Tyi) - Qi_list[[s]])^2) * (1/Ji)
        }
        
        # Sum over j
        mean(wass_vec)
      }
      
      # compute LPPD_0
      W1 <- subject_wass(samps_full)
      W0 <- subject_wass(samps_red)
      
      T_i[i] <- W1 - W0
    }
  }
  
  T_total <- sum(T_i)
  
  list(
    T = T_total,
    T_i = T_i
  )
}


compute_mc_stats_lik <- function(
    y_list,
    x,
    pipeline,
    V = length(y_list),   # default = leave-one-subject-out
    S = 2000,
    seed = 12345
){
  set.seed(seed)
  
  # Set parameters
  N <- length(y_list)
  p_grid <- pipeline$training$cache$p_grid
  supp_TY <- pipeline$training$cache$supp_TY
  y_trans_fun <- y_trans_to_fun[[pipeline$stages[[1]]$state$y_trans]]
  y_shift <- pipeline$stages[[1]]$state$y_shift
  
  if (length(x) != N) {
    stop("x must have one value per subject")
  }
  
  # Encode all subjects
  ctx <- new_context(
    payload = y_list,
    cache = pipeline$training$cache,
    meta = list()
  )
  ctx <- encode(pipeline, ctx, from = 0, to = 6)
  Z_all <- do.call(rbind, ctx$payload)
  K_ <- ncol(Z_all)
  
  # Subject folds
  fold_id <- sample(rep(1:V, length.out = N))
  
  T_i <- numeric(N)
  
  # ----- Loop over folds
  for(v in 1:V){
    print(str_glue("========== Fold {v} =========="))
    
    test_idx  <- which(fold_id == v)
    train_idx <- setdiff(1:N, test_idx)
    
    Z_train <- Z_all[train_idx,,drop=FALSE]
    x_train <- x[train_idx]
    
    # ----- Fit full model
    samps_full <- fit_model_bin_pred(
      Z_train, x_train,
      n_chains = 1,
      burn_in = 2000,
      thin_interval = 5,
      n_samps = S
    )
    
    # ----- Fit reduced model
    samps_red <- fit_model_mean(
      Z_train,
      n_chains = 1,
      burn_in = 2000,
      thin_interval = 5,
      n_samps = S
    )
    samps_red <- cbind(
      samps_red[,1:(K_**2)],
      matrix(0, nrow = S, ncol = K_), ## set betas = 0
      samps_red[,(K_**2+1):(K_**2+K_)]
    )
    
    ## ----- Loop over test subjects
    for(i in test_idx){
      print(str_glue("i = {i}"))
      
      Tyi <- y_trans_fun(y_list[[i]], y_shift)
      xi <- x[i]
      Ji <- length(Tyi)
      
      # helper: compute subject LPPD for one posterior sample matrix
      subject_LPPD <- function(samps){
        
        # Posterior predictive sampling
        Sigmas <- samps[,1:(K_^2)]
        betas  <- samps[,(K_^2+1):(K_^2+K_)]
        mus    <- samps[,(K_^2+K_+1):(K_^2+2*K_)]
        z_draws <- vector("list", S)
        for(s in seq_len(S)){
          z_draws[[s]] <- MASS::mvrnorm(
            1,
            mu = mus[s,] + betas[s,]*xi,
            Sigma = matrix(Sigmas[s,],K_)
          )
        }
        
        # Decode draws
        z_ctx <- new_context(
          payload = z_draws,
          cache = pipeline$training$cache,
          meta = list(Ji_vec = rep(Ji, length(z_draws)))
        )
        Q_list <- decode(pipeline, z_ctx, from = 6, to = 3)$payload
        
        # Compute likelihoods
        lik_mat <- matrix(0,S,Ji)
        for(s in seq_len(S)){
          lik_mat[s,] <- compute_likelihoods(
            Tyi,
            Q_list[[s]],
            p_grid,
            supp_Y = supp_TY,
            log = FALSE
          )
        }
          
        # Sum over j
        sum(log(colMeans(lik_mat)))
      }
      
      # compute LPPD_0
      L1 <- subject_LPPD(samps_full)
      L0 <- subject_LPPD(samps_red)
      
      T_i[i] <- (L1 - L0) / Ji
    }
  }
  
  T_total <- sum(T_i)
  
  list(
    T = T_total,
    T_i = T_i
  )
}



## ---------- Setup ---------- ##

dataset <- 'bicluster'

if (dataset == 'tean') {
  
  ## Load objects
  path_pipe <- file.path('artifacts', 'demo_tean-mc', 'pipe.rds')
  path_y <- file.path('data', 'processed', 'tean_v2.rds')
  path_cov <- file.path('data', 'processed', 'tean_v2_cov.rds')
  pipeline <- readRDS(path_pipe)
  y_list <- readRDS(path_y)
  df_cov <- readRDS(path_cov)
  df_cov <- df_cov[match(names(y_list), df_cov$id),]
  x <- as.numeric(df_cov$age > 14)
  
} else if (dataset == 'bicluster') {
  
  ## Load objects
  path_pipe <- file.path('artifacts', 'demo_bicluster-mc', 'pipe.rds')
  path_y <- file.path('data', 'processed', 'bicluster_v4.rds')
  path_x <- file.path('data', 'processed', 'bicluster_v4_groups.rds')
  pipeline <- readRDS(path_pipe)
  y_list <- readRDS(path_y)
  x <- readRDS(path_x)
  
} else {
  stop("Dataset not supported!")
}



## ---------- Perform Model Comparison ---------- ##

## Wasserstein MC
out <- compute_mc_stats_wass(
  y_list,
  x,
  pipeline,
  V = 5,
  S = 200,
  seed = 12345
)
path <- file.path('artifacts', str_glue('demo_{dataset}-mc'), 'Ti_wass.rds')
saveRDS(out$T_i, path)

## Likelihood MC
out <- compute_mc_stats_lik(
  y_list,
  x,
  pipeline,
  V = 5,
  S = 200,
  seed = 12345
)
path <- file.path('artifacts', str_glue('demo_{dataset}-mc'), 'Ti_lik.rds')
saveRDS(out$T_i, path)



## ---------- Get Mean Likelihoods ---------- ## 

# ## Paths
# path_idx_draws <- file.path('artifacts', 'demo_tean-mc', 'idx_draws.rds')
# path_0 <- file.path('artifacts', 'demo_tean-mc', 'draws_0.rds')
# path_1 <- file.path('artifacts', 'demo_tean-mc', 'draws_1.rds')
# 
# ## Load Q-draws
# idx_draws <- readRDS(path_idx_draws)
# Q0_list <- readRDS(path_0)$Q[idx_draws]
# Q1_list <- readRDS(path_1)$Q[idx_draws]
# S <- length(idx_draws)
# supp_TY <- pipeline$training$cache$supp_TY
# p_grid <- pipeline$training$cache$p_grid
# 
# ## Compute likelihoods
# lik_0 <- numeric(length(supp_TY))
# lik_1 <- numeric(length(supp_TY))
# for (s in 1:S) {
#   lik_0_s <- compute_likelihoods(supp_TY, Q0_list[[s]], p_grid, supp_TY) 
#   lik_1_s <- compute_likelihoods(supp_TY, Q1_list[[s]], p_grid, supp_TY) 
#   lik_0 <- lik_0 + lik_0_s / S
#   lik_1 <- lik_1 + lik_1_s / S
# }
# 
# ## Save likelihoods
# path_0 <- file.path('artifacts', 'lik_0.rds')
# path_1 <- file.path('artifacts', 'lik_1.rds')
# saveRDS(lik_0, path_0)
# saveRDS(lik_1, path_1)
# 
# mar_default <- c(5.1, 4.1, 4.1, 2.1)
# par(mar = c(5.1, 4.5, 4.1, 2.1))
# plot(1:5, 1:5, ylab = expression(tilde(Q)(p)))


