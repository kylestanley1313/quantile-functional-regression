## Globals
N <- 500
n <- 100
seed <- 12345
path_out <- file.path('data', 'processed', str_glue('multinomial-{N}.rds'))
path_out_x <- file.path('data', 'processed', str_glue('multinomial-{N}_x.rds'))

## Set x
x <- runif(N)
# x <- rbinom(N, size = 1, prob = 0.5)

## Simulate data
set.seed(seed)
y_list <- vector("list", length = N)
for (i in 1:N) {
  n_1 <- x[i] * n
  n_2 <- n - n_1
  y_1 <- sample(
    1:10, size = n_1, replace = TRUE,
    prob = c(0.2, 0.2, 0.15, 0.1, 0.1, 0.1, 0.1, 0.05, 0, 0)
  )
  y_2 <- sample(
    1:10, size = n_2, replace = TRUE,
    prob = c(0.2, 0.2, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1)
  )
  y_list[[i]] <- c(y_1, y_2)
}
saveRDS(y_list, path_out)
saveRDS(x, path_out_x)
