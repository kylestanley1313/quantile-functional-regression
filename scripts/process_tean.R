path_in <- file.path('data', 'raw', 'tean_accelerometer_final_v2.csv')
path_out <- file.path('data', 'processed', 'tean_v1.rds')

## Read data
df_data <- read.csv(path_in)

## Get "valid" days and replace '-999' placeholder with NA
df_data <- df_data[df_data$VALID == 1,]
df_data[df_data == -999] <- NA

## Get subject distributions (aggregate across days)
ids <- unique(df_data$ID)
y_list <- list()
for (id in ids) {
  df_data_id <- df_data[df_data$ID == id,5:ncol(df_data)]
  n_days <- nrow(df_data_id)
  if (n_days > 0) {
    y_list[[id]] <- c()
    for (d in 1:n_days) {
      tmp <- df_data_id[d,]
      mask <- !is.na(tmp)
      y_list[[id]] <- c(y_list[[id]], tmp[mask])
    }
  }
}

## Save
saveRDS(y_list, path_out)

