
## Globals
N <- 100
N_1 <- round(0.6 * N)
n <- 100
# n_cut <- round(0.5*n)
seed <- 12345
path_out <- file.path('data', 'processed', 'bicluster_v1.rds')
path_out_groups <- file.path('data', 'processed', 'bicluster_v1_groups.rds')

## Set groups
groups <- c(rep(0, N_1), rep(1, N - N_1))

## Simulate data
set.seed(seed)
y_list <- vector("list", length = N)
for (i in 1:N) {
  
  if (groups[i] == 0) {
    y_list[[i]] <- sample(
      1:10, size = n, replace = TRUE,
      prob = c(0.2, 0.2, 0.15, 0.1, 0.1, 0.1, 0.1, 0.05, 0, 0)
      # prob = c(0.4, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0, 0)
    )
  } else {
    y_list[[i]] <- sample(
      1:10, size = n, replace = TRUE,
      prob = c(0.2, 0.2, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1)
      # prob = c(0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05)
    )
  }
  
}
saveRDS(y_list, path_out)
saveRDS(groups, path_out_groups)




