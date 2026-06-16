## Globals
N <- 250
n <- 1000
seed <- 12345
path_out <- file.path('data', 'processed', 'disc_normals_v1.rds')

## Get means and std devs
set.seed(seed)
mean_vec <- rt(N, df = 3)
sd_vec <- exp(rnorm(N, mean = 0, sd = 0.25))

## Simulate data
y_list <- vector("list", length = N)
for (i in 1:N) {
  y_list[[i]] <- round(rnorm(n, mean = mean_vec[i], sd = sd_vec[i]), 2)
}
saveRDS(y_list, path_out)
