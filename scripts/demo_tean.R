source('src/cot.R')

## Load data
path <- file.path('data', 'processed', 'tean_v1.rds')
y_list <- readRDS(path)
y_list <- y_list[1:100]
y_max <- max(unlist(y_list))

## Define grid
p_grid_left <- seq(0, 0.1, length.out = 100)
p_grid_mid <- seq(0.1, 0.9, length.out = 100)
p_grid_right <- seq(0.9, 1, length.out = 100)
p_grid <- c(p_grid_left, p_grid_mid, p_grid_right)
p_grid <- sort(unique(p_grid))
p_grid <- p_grid[2:(length(p_grid) - 1)]
# p_grid <- seq(0.01, 0.99, by = 0.01)

## Construct pipeline
pipeline <- construct_pipeline(
  p_grid = p_grid,
  supp_Y = 0:(2*y_max),   
  y_trans = log_transform, 
  K = NULL,
  epsilon = 0.001,
  alpha = 0.05,
  K_max = 15,
  loss_fun = one_minus_sqcor,
  V = 5,
  ratio_trans = NULL, 
  p_star = 0.5,
  y_min = 0, 
  min_dQ = 1e-8, 
  flow_n_layers = 16,
  flow_epochs = 200,
  flow_lr = 1e-3,
  flow_path = 'artifacts/flow_1.pth',
  seed = 12345
)

## Fitting
pipeline <- fit(pipeline, y_list)
y_ctx <- list(data = y_list, cache = NULL)

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
Ji_vec <- lengths(y_list)
pi_grid <- get_uniform_p_grid(Ji_vec[[i]])
# p_grid <- context$cache$p_grid

## Plotting
breaks_y <- seq(0, 10000, by = 500)
breaks_Ty <- seq(0, 12, by = 0.5)
valid_col <- rgb(1, 0, 0, alpha = 0.5)
par(mfrow=c(2,4))
h <- hist(y_ctx$data[[i]], breaks = breaks_y)
hist(y_ctx_$data[[i]], add = TRUE, col = valid_col, breaks = breaks_y)
h <- hist(Ty_ctx$data[[i]], breaks = breaks_Ty)
hist(Ty_ctx_$data[[i]], add = TRUE, col = valid_col, breaks = breaks_Ty)
plot(pi_grid, Qi_ctx$data[[i]], type = 'l', xlim = c(0, 1))
lines(pi_grid, Qi_ctx_$data[[i]], type = 'l', col = valid_col)
plot(pi_grid, Qi_ctx$data[[i]], type = 'l', xlim = c(0, 1), col = 'gray')
lines(p_grid, Q_ctx$data[[i]], type = 'l')
lines(p_grid, Q_ctx_$data[[i]], type = 'l', col = valid_col)
plot(p_grid, G_Q_star_ctx$data$G_list[[i]], type = 'l', xlim = c(0, 1))
lines(p_grid, G_Q_star_ctx_$data$G_list[[i]], type = 'l', col = valid_col)
plot(c_ctx$data[[i]])
points(c_ctx_$data[[i]], col = valid_col)
plot(z_ctx$data[[i]])

