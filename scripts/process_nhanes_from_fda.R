library(dplyr)


## ---------- Multi-Level Data ---------- ##

## Read data
path_in <- file.path('data', 'raw', 'nhanes_fda_with_R_ml.rds')
df <- readRDS(path_in)

# mutate(MIMS_UNIT = pmax(MIMS_UNIT, 0))

## Filter dataframe
df <- df %>%
  group_by(SEQN) %>%
  filter(n() == 7) %>%
  ungroup()


## Get y_list
idx_list <- split(seq_len(nrow(df)), df$SEQN)
y_list <- lapply(idx_list, function(idx) {
  pmax(df$MIMS[idx, , drop = FALSE], 0)
})

## Save y_list
path <- file.path('data', 'processed', 'nhanes_v1_nofilter.rds')
saveRDS(y_list, path)

## NOTES
##  - Subjects have between 3 and 7 days of data
##  - Minimum MIMS is 0
##  - Second smallest table
##      0.006 0.007 0.008  0.01 0.012 
##        457    38     3     1     1 


## ---------- Single-Level Data ---------- ##

## Read data
path_in <- file.path('data', 'raw', 'nhanes_fda_with_R.rds')
df <- readRDS(path_in)

## Extract covariate dataframe
df_cov <- df %>%
  mutate(sub_id = as.character(SEQN)) %>%
  dplyr::select(sub_id, age, gender, race, BMI, PIR, education)

## Save df_cov
path <- file.path('data', 'processed', 'nhanes_cov.rds')
saveRDS(df_cov, path)

