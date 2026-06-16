
## Globals
N <- 250
n <- 100
seed <- 12345
path_out <- file.path('data', 'processed', 'normals_v1.rds')

## Get means and std devs
set.seed(seed)
mean_vec <- rt(N, df = 3)
sd_vec <- exp(rnorm(N, mean = 0, sd = 1))

## Simulate data
y_list <- vector("list", length = N)
for (i in 1:N) {
  y_list[[i]] <- rnorm(n, mean = mean_vec[i], sd = sd_vec[i])
}
saveRDS(y_list, path_out)
