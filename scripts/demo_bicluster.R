library(MASS)

source('src/cot.R')
source('src/model.R')
source('src/utils.R')

## ---------- Globals ---------- ##

col_gp1 <- 'blue'
col_gp2 <- 'orange'
col_mean <- 'red'
col_draw <- 'gray'


## ---------- Data Loading ---------- ##

path <- file.path('data', 'processed', 'bicluster_v1.rds')
path_groups <- file.path('data', 'processed', 'bicluster_v1_groups.rds')
y_list <- readRDS(path)
groups <- readRDS(path_groups)
N <- length(y_list)
Ji <- length(y_list[[1]])
y_max <- max(unlist(y_list))


## ---------- CoT ---------- ##

## Define grid
p_grid <- 1:Ji / (1 + Ji)
# p_min <- 1 / (1 + Ji)
# p_max <- Ji / (1 + Ji)
# p_grid <- c(
#   seq(p_min, 0.9, length.out = 100),
#   seq(0.9, p_max, length.out = 101)
# )
# p_grid <- sort(unique(p_grid))

## Construct pipeline
pipeline <- construct_pipeline(
  p_grid = p_grid,
  supp_Y = 1:10,   
  y_trans = NULL, 
  K = NULL,
  epsilon = 0.05,
  alpha = 0.05,
  K_max = 15,
  loss_fun = one_minus_sqcor,
  V = 5,
  ratio_trans = NULL, 
  p_star = 0.5,
  y_min = NULL, 
  min_dQ = 1e-8, 
  flow_n_layers = 16,
  flow_epochs = 8000,
  flow_lr = 1e-4,
  flow_path = 'artifacts/flow_bicluster_1.pth',
  seed = 12345
)

## Fitting
pipeline <- fit(pipeline, y_list)

## New context
y_ctx <- new_context(
  payload = y_list,
  cache = pipeline$cache_init,
  meta = list()
)

## Encoding
Ty_ctx <- encode(pipeline, y_ctx, from = 0, to = 1)
Qi_ctx <- encode(pipeline, Ty_ctx, from = 1, to = 2)
Q_ctx <- encode(pipeline, Qi_ctx, from = 2, to = 3)
G_Q_star_ctx <- encode(pipeline, Q_ctx, from = 3, to = 4)
c_ctx <- encode(pipeline, G_Q_star_ctx, from = 4, to = 5)
z_ctx <- encode(pipeline, c_ctx, from = 5, to = 6)

## Decoding
c_ctx_ <- decode(pipeline, z_ctx, from = 6, to = 5)
G_Q_star_ctx_ <- decode(pipeline, c_ctx_, from = 5, to = 4)
Q_ctx_ <- decode(pipeline, G_Q_star_ctx, from = 4, to = 3)
Qi_ctx_ <- decode(pipeline, Q_ctx, from = 3, to = 2)
Ty_ctx_ <- decode(pipeline, Qi_ctx_, from = 2, to = 1)
y_ctx_ <- decode(pipeline, Ty_ctx_, from = 1, to = 0)

## Plotting params
i <- 10
pi_grid <- get_uniform_p_grid(Ji)

## Plot encode/decode
breaks_y <- seq(0, 10, by = 1)
breaks_Ty <- seq(0, 10, by = 1)
valid_col <- rgb(1, 0, 0, alpha = 0.5)
par(mfrow=c(2,4))
h <- hist(y_ctx$payload[[i]], breaks = breaks_y)
hist(y_ctx_$payload[[i]], add = TRUE, col = valid_col, breaks = breaks_y)
h <- hist(Ty_ctx$payload[[i]], breaks = breaks_Ty)
hist(Ty_ctx_$payload[[i]], add = TRUE, col = valid_col, breaks = breaks_Ty)
plot(pi_grid, Qi_ctx$payload[[i]], type = 'l', xlim = c(0, 1))
lines(pi_grid, Qi_ctx_$payload[[i]], type = 'l', col = valid_col)
plot(pi_grid, Qi_ctx$payload[[i]], type = 'l', xlim = c(0, 1), col = 'gray')
lines(p_grid, Q_ctx$payload[[i]], type = 'l')
lines(p_grid, Q_ctx_$payload[[i]], type = 'l', col = valid_col)
plot(p_grid, G_Q_star_ctx$payload$G_list[[i]], type = 'l', xlim = c(0, 1))
lines(p_grid, G_Q_star_ctx_$payload$G_list[[i]], type = 'l', col = valid_col)
plot(c_ctx$payload[[i]])
points(c_ctx_$payload[[i]], col = valid_col)
plot(z_ctx$payload[[i]])

## Plot EQFs
fun_list <- Qi_ctx$payload
colors <- c(
  rep(col_gp1,   sum(groups == 0)),
  rep(col_gp2, sum(groups == 1))
)
widths <- rep(1, length(fun_list))
types  <- rep(1, length(fun_list))
color_width_type_labels <- data.frame(
  color = c(col_gp1, col_gp2),
  width = c(1, 1),
  type  = c(1, 1),
  label = c("Group 1", "Group 2")
)
plot_funs(
  fun_list = fun_list,
  p_grid = pi_grid,
  ylim = c(1, 10),
  colors = colors,
  widths = widths,
  types = types,
  color_width_type_labels = color_width_type_labels
)



## ---------- Model ---------- ##

Z <- do.call(rbind, z_ctx$payload)
samps <- fit_mean_model(
  Z, 
  n_chains = 1, 
  burn_in = 2000, 
  thin_interval = 5, 
  n_samps = 5000
)



## ---------- Posterior Predictive Sampling ---------- ##

## Set parameters
K <- ncol(Z)  # really K_plus = K + 1
S <- nrow(samps)
n_param <- ncol(samps)

## Posterior predictive samples
set.seed(12345)
z_draws <- vector('list', length = K)
for (s in 1:S) {
  mu <- samps[s,(n_param-K+1):n_param]
  Sigma <- matrix(samps[s,1:(n_param-K)], nrow = K)
  z_draws[[s]] <- mvrnorm(1, mu = mu, Sigma = Sigma) 
}

## ----- Decode Draws

## Context setup
z_draws_ctx <- new_context(
  payload = z_draws,
  cache = pipeline$cache_init,
  meta = list()
)

## Payload indices
idx_draws <- 1:S
idx_z_mean <- S + 1
idx_c_mean <- S + 2
idx_Q_mean <- S + 3

## Z --> C
z_mean <- colMeans(do.call(rbind, z_draws_ctx$payload[idx_draws]))
z_draws_ctx$payload[[idx_z_mean]] <- z_mean
c_draws_ctx <- decode(pipeline, z_draws_ctx, from = 6, to = 5)

## C --> G
c_mean <- colMeans(do.call(rbind, c_draws_ctx$payload[idx_draws]))
c_draws_ctx$payload[[idx_c_mean]] <- c_mean
G_Q_star_draws_ctx <- decode(pipeline, c_draws_ctx, from = 5, to = 4)

## G --> Q
Q_draws_ctx <- decode(pipeline, G_Q_star_draws_ctx, from = 4, to = 3)

## Q --> Qi
Q_mean <- colMeans(do.call(rbind, Q_draws_ctx$payload[idx_draws]))
Q_draws_ctx$payload[[idx_Q_mean]] <- Q_mean
Q_draws_ctx$meta$Ji_vec <- rep(100, length(Q_draws_ctx$payload))
Qi_draws_ctx <- decode(pipeline, Q_draws_ctx, from = 3, to = 2)

## Qi --> TY
Ty_draws_ctx <- decode(pipeline, Qi_draws_ctx, from = 2, to = 1)

## TY --> Y
y_draws_ctx <- decode(pipeline, Ty_draws_ctx, from = 1, to = 0)


################## Old vs. New Compute Quantlets ###############################
Q_draws <- Q_draws_ctx$payload[idx_draws]
Q_draws <- do.call(rbind, Q_draws)
quantile(Q_draws[,100], c(0.9, 0.95, 0.99, 0.999))
## Old Quantlets:
##          90%          95%          99%        99.9% 
## 1.001264e+01 1.008558e+01 4.104443e+02 6.072798e+09 
##
## New Quantlets:
##      90%       95%       99%     99.9% 
## 9.932402 10.130395 10.992782 12.428218 
################################################################################

## ---------- Plotting ---------- ##

## z-draws vs. z-embeddings plot
Z_to_plot <- rbind(
  do.call(rbind, z_draws_ctx$payload[idx_draws]),
  do.call(rbind, z_ctx$payload)
)
colors = c(
  rep(col_draw, length.out = S),
  rep(col_gp1, length.out = sum(groups == 0)),
  rep(col_gp2, length.out = sum(groups == 1))
)
shapes <- c(
  rep(19, length.out = S + N)
)
sizes <- c(
  rep(1, length.out = S + N)
)
color_shape_size_labels = data.frame(
  color = c(col_gp1, col_gp2, col_draw),
  shape = c(19, 19, 19),
  size = c(1, 1, 1),
  label = c('Group 1', 'Group 2', 'Z-draws')
)
plot_embeddings(
  Z_to_plot, 
  colors = colors,
  shapes = shapes,
  sizes = sizes,
  color_shape_size_labels = color_shape_size_labels,
  stats = FALSE
)

## c-draws vs. c-embeddings plot
C_to_plot <- rbind(
  do.call(rbind, c_draws_ctx$payload[idx_draws]),
  do.call(rbind, c_ctx$payload)
)
colors = c(
  rep(col_draw, length.out = S),
  rep(col_gp1, length.out = sum(groups == 0)),
  rep(col_gp2, length.out = sum(groups == 1))
)
shapes <- c(
  rep(19, length.out = S + N)
)
sizes <- c(
  rep(1, length.out = S + N)
)
color_shape_size_labels = data.frame(
  color = c(col_gp1, col_gp2, col_draw),
  shape = c(19, 19, 19),
  size = c(1, 1, 1),
  label = c('Group 1', 'Group 2', 'Z-draws')
)
plot_embeddings(
  C_to_plot, 
  colors = colors,
  shapes = shapes,
  sizes = sizes,
  color_shape_size_labels = color_shape_size_labels,
  stats = FALSE
)

## z-mean plot
Z_to_plot <- rbind(
  do.call(rbind, z_ctx$payload),
  z_draws_ctx$payload[[idx_z_mean]]
)
colors = c(
  rep(col_gp1, length.out = sum(groups == 0)),
  rep(col_gp2, length.out = sum(groups == 1)),
  col_mean
)
shapes <- c(
  rep(19, length.out = N),
  3
)
sizes <- c(
  rep(1, length.out = N),
  2
)
color_shape_size_labels = data.frame(
  color = c(col_gp1, col_gp2, col_mean),
  shape = c(19, 19, 3),
  size = c(1, 1, 2),
  label = c('Group 1', 'Group 2', 'Z-Mean')
)
plot_embeddings(
  Z_to_plot, 
  colors = colors,
  shapes = shapes,
  sizes = sizes,
  color_shape_size_labels = color_shape_size_labels
)

## c-mean plot
C_to_plot <- rbind(
  do.call(rbind, c_ctx$payload),
  c_draws_ctx$payload[[idx_z_mean]], 
  c_draws_ctx$payload[[idx_c_mean]]
)
colors = c(
  rep(col_gp1, length.out = sum(groups == 0)),
  rep(col_gp2, length.out = sum(groups == 1)),
  col_mean, 
  col_mean
)
shapes <- c(
  rep(19, length.out = N),
  3,
  4
)
sizes <- c(
  rep(1, length.out = N),
  2,
  2
)
color_shape_size_labels = data.frame(
  color = c(col_gp1, col_gp2, col_mean, col_mean),
  shape = c(19, 19, 3, 4),
  size = c(1, 1, 2, 2),
  label = c('Group 1', 'Group 2', 'Z-Mean', 'C-Mean')
)
plot_embeddings(
  C_to_plot, 
  colors = colors,
  shapes = shapes,
  sizes = sizes,
  color_shape_size_labels = color_shape_size_labels
)


## Q-mean plot
Q_to_plot <- c(
  Q_ctx$payload,
  list(Q_draws_ctx$payload[[idx_z_mean]]), 
  list(Q_draws_ctx$payload[[idx_c_mean]]),
  list(Q_draws_ctx$payload[[idx_Q_mean]])
)
colors <- c(
  rep(col_gp1,   sum(groups == 0)),
  rep(col_gp2, sum(groups == 1)),
  col_mean, 
  col_mean, 
  col_mean
)
widths <- c(
  rep(1, length(fun_list)),
  2, 
  2, 
  2
)
types <- c(
  rep(1, length(fun_list)),
  2, 
  3, 
  4
)
color_width_type_labels <- data.frame(
  color = c(col_gp1, col_gp2, col_mean, col_mean, col_mean),
  width = c(1, 1, 2, 2, 2),
  type  = c(1, 1, 2, 3, 4),
  label = c("Group 1", "Group 2", "Z-mean", "C-mean", "Q-mean")
)
plot_funs(
  fun_list = Q_to_plot,
  p_grid = p_grid,
  ylim = c(1, 10),
  colors = colors,
  widths = widths,
  types = types,
  color_width_type_labels = color_width_type_labels
)


## PMFs
f_data <- y_to_pmf(unlist(y_list))
f_data_1 <- y_to_pmf(unlist(y_list[groups == 0]))
f_data_2 <- y_to_pmf(unlist(y_list[groups == 1]))
f_z_mean <- y_to_pmf(y_draws_ctx$payload[[idx_z_mean]])
f_c_mean <- y_to_pmf(y_draws_ctx$payload[[idx_c_mean]])
f_Q_mean <- y_to_pmf(y_draws_ctx$payload[[idx_Q_mean]])
f_y_mean <- y_to_pmf(unlist(y_draws_ctx$payload[idx_draws]))
par(mfrow=c(4,1))
## Data PMF
plot(
  f_data$value,
  f_data$prob,
  type = "h",
  lwd = 2,
  xlim = c(1, 10),
  xlab = "y",
  ylab = "P(Y = y)",
  col = "black"
)
points(f_data$value, f_data$prob, pch = 19, col = "black")
## Group 1 PMF
plot(
  f_data_1$value,
  f_data_1$prob,
  type = "h",
  lwd = 2,
  xlim = c(1, 10),
  xlab = "y",
  ylab = "P(Y = y)",
  col = col_gp1
)
points(f_data_1$value, f_data_1$prob, pch = 19, col = col_gp1)
## Group 2 PMF
plot(
  f_data_2$value,
  f_data_2$prob,
  type = "h",
  lwd = 2,
  xlim = c(1, 10),
  xlab = "y",
  ylab = "P(Y = y)",
  col = col_gp2
)
points(f_data_2$value, f_data_2$prob, pch = 19, col = col_gp2)
## Summary PMF
f_to_plot <- f_y_mean
plot(
  f_to_plot$value,
  f_to_plot$prob,
  type = "h",
  lwd = 2,
  xlim = c(1, 10),
  xlab = "y",
  ylab = "P(Y = y)",
  col = col_mean
)
points(f_to_plot$value, f_to_plot$prob, pch = 19, col = col_mean)

