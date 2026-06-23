library(haven)
library(dplyr)


## ---------- Process Activity Data ---------- ##

path_1 <- file.path('data', 'raw', 'PAXMIN_G_2011_2012.xpt')
path_2 <- file.path('data', 'raw', 'PAXMIN_H_2013_2014.xpt')
df_1 <- read_xpt(
  path_1, 
  col_select = c(
    'SEQN',      ## subject number
    'PAXDAYM',   ## day of wear
    'PAXMTSM',   ## MIMS triaxial-value for minute
    'PAXPREDM',  ## wake/sleep/non-wear prediction for minute
    'PAXFLGSM'
  )
)
df_2 <- read_xpt(
  path_2, 
  col_select = c(
    'SEQN',
    'PAXDAYM',
    'PAXMTSM',
    'PAXPREDM',
    'PAXFLGSM'
  )
)
df <- rbind(df_1, df_2)

# tmp <- df %>%
#   count(SEQN, PAXDAYM)

# df %>%
#   filter(SEQN == 76690) %>%
#   count(PAXFLGSM)


## CODES
##  1 --> wake wear
##  2 --> sleep wear
##  3 --> non wear
##  4 --> unknown


## ----- Preprocessing Strategy 1

## IDEA: 

## Preprocessing
##  - Filter out rows with PAXPREDM that are not equal to 1 (wake aware minutes)
##  - Remove rows with PAXMTSM equal to -0.01 or greater than 500 (-0.01 is an 
##    error code and I am marking 500 as physiologically impossible)
##  - Remove rows with (SEQN, PAXDAYM) counts that are less than 480 (i.e., 
##    subject-days where there are fewer than 8 hours of wake wear time)
##  - Remove subjects with fewer than 7 qualifying days. Truncate subjects 
##    with more than 7 qualifying days.
##  - Pool PAXMTSM values (i.e., minute level MIMS units) across days for each 
##    subject. The result should be a list y_list that maps subject IDs to a 
##    vector of MIMS.   

## Filter to "clean" rows
df_clean <- df %>%
  filter(PAXPREDM != 3) %>%  ## remove nonwear
  filter(PAXMTSM != -0.01, PAXMTSM <= 500) %>%  ## remove rows with MIMS outside feasible range
  filter(PAXFLGSM == "")  ## remove rows with quality flag

## Keep only subject-days with at least 8 hours (480 minutes) of clean rows
valid_days <- df_clean %>%
  count(SEQN, PAXDAYM) %>%
  filter(n >= 480)
df_clean <- df_clean %>%
  semi_join(valid_days, by = c('SEQN', 'PAXDAYM'))

## Keep subjects with >= 7 qualifying days; truncate to the first 7 days
day_rank <- df_clean %>%
  distinct(SEQN, PAXDAYM) %>%
  arrange(SEQN, PAXDAYM) %>%
  group_by(SEQN) %>%
  mutate(day_index = row_number(), n_days = n()) %>%
  ungroup()
keep_days <- day_rank %>%
  filter(n_days >= 7, day_index <= 7)
df_clean <- df_clean %>%
  semi_join(keep_days, by = c('SEQN', 'PAXDAYM'))

## Pool minute-level MIMS across days within each subject. y_list maps
## subject IDs (SEQN) to a vector of pooled minute-level MIMS values.
df_clean <- df_clean %>%
  arrange(SEQN, PAXDAYM)
y_list <- split(df_clean$PAXMTSM, df_clean$SEQN)

## Save data
path <- file.path('data', 'processed', 'nhanes_cdc_v1.rds')
saveRDS(y_list, path)


## NOTE: 
##  - Subject 8093 (76690 CDC) has a weird (length-10080) QF that is slightly throwing


## ---------- Process Demographic Data ---------- ##



path_1 <- file.path('data', 'raw', 'DEMO_G_2011_2012.xpt')
path_2 <- file.path('data', 'raw', 'DEMO_H_2013_2014.xpt')
df_1 <- read_xpt(
  path_1, 
  col_select = c(
    'SEQN',      ## subject number
    'RIDAGEYR',  ## age
    'RIAGENDR',  ## gender
    'INDFMPIR'   ## pir
  )
)
df_2 <- read_xpt(
  path_2, 
  col_select = c(
    'SEQN',
    'RIDAGEYR',
    'RIAGENDR',
    'INDFMPIR'
  )
)
df <- rbind(df_1, df_2)
df <- df %>% 
  rename(
    sub_id = SEQN,
    gender = RIAGENDR,
    age = RIDAGEYR,
    PIR = INDFMPIR
  )
path <- file.path('data', 'processed', 'nhanes_cov_cdc.rds')
saveRDS(df, path)
