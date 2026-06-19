library(dplyr)
library(lubridate)
library(stringr)
library(tidyr)



## ---------- Globals ---------- ##

## Directories
dir_data <- '/Users/stanley5/Box/CHOP_SGrow2'
dir_demo_dxa <- file.path(dir_data, 'data/Demographics_and_DXA')
dir_enmo <- file.path(dir_data, 'data/raw/ENMO')
dir_mims <- file.path(dir_data, 'data/raw/mims')
dir_sleep <- file.path(dir_data, 'data/raw/sleep')
dir_nonwear <- file.path(dir_data, 'data/raw/non_wear')
dir_mvpa <- file.path(dir_data, 'data/raw/MVPA')

## Other
rewear_ids <- c(10302, 10642, 11100)


## ---------- Utilities ---------- ##

get_data_file <- function(sub_id, file_type, rewear = FALSE) {
  str_glue('{sub_id}_v1_{file_type}.csv')
}

get_date_from_ts <- function(ts) {
  d <- as.POSIXct(ts, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  as.Date(d)
}

get_hour_from_ts <- function(ts) {
  d <- as.POSIXct(ts, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  as.integer(format(d, "%H"))
}


## ---------- MIMS (awake wear time) ---------- ##

## Prepare day-level data
path_out <- file.path('data', 'processed', 'chop-mims-day_v1.rds')
y_list <- list()
paths_in <- list.files(path = dir_mims, full.names = TRUE)
i <- 1
for (path_y in paths_in) {
  print(str_glue("i = {i}"))

  ## Extract sub_id
  sub_id <- strsplit(tail(strsplit(path_y, "/")[[1]], n = 1), '_')[[1]][1]
  
  ## Set paths
  path_sleep <- file.path(
    dir_sleep, get_data_file(
      sub_id, 'sleep',
      rewear = ifelse(sub_id %in% rewear_ids, TRUE, FALSE)
    )
  )
  if (!file.exists(path_sleep)) {
    print(str_glue("File does not exist: {path_sleep}"))
    next
  }
  path_nonwear <- file.path(
    dir_nonwear, get_data_file(
      sub_id, 'nonwear',
      rewear = ifelse(sub_id %in% rewear_ids, TRUE, FALSE)
    )
  )
  if (!file.exists(path_nonwear)) {
    print(str_glue("File does not exist: {path_nonwear}"))
    next
  }
  
  ## Read dataframes
  df_y <- read.csv(path_y, header = TRUE)
  df_sleep <- read.csv(path_sleep, header = TRUE)
  df_nonwear <- read.csv(path_nonwear, header = TRUE)
  
  ## Parse timestamps
  df_y <- df_y %>%
    mutate(
      ts = as.POSIXct(HEADER_TIME_STAMP, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
    )
  df_sleep <- df_sleep %>%
    mutate(
      sleeponset_ts = as.POSIXct(sleeponset_ts, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
      wakeup_ts     = as.POSIXct(wakeup_ts,     format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
    )
  df_nonwear <- df_nonwear %>%
    mutate(
      ts_start = as.POSIXct(timestamp, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
      ts_end   = ts_start + 15 * 60   # 15-minute epochs
    )
  
  ## Remove rows inside sleep intervals
  df_nonwear_int <- df_nonwear %>%
    filter(nonwear == 1)
  df_y_awake_wear <- df_y %>%

    ## Clamp negatives to zero and remove unreasonably high values
    filter(MIMS_UNIT != -0.01, MIMS_UNIT <= 500) %>%
    
    ## Remove sleep
    left_join(
      df_sleep,
      join_by(
        ts >= sleeponset_ts,
        ts <= wakeup_ts
      )
    ) %>%
    filter(is.na(sleeponset_ts)) %>%
    dplyr::select(-sleeponset_ts, -wakeup_ts) %>%
    
    ## Remove nonwear
    left_join(
      df_nonwear_int,
      join_by(
        ts >= ts_start,
        ts <  ts_end
      )
    ) %>%
    filter(is.na(ts_start)) %>%
    dplyr::select(-ts_start, -ts_end, -nonwear)

  ## Define distributions (by day, using Eastern time)
  df_y_awake_wear <- df_y_awake_wear %>%
    mutate(date = as.character(as.Date(ts, tz = "America/New_York")))
  day_list <- split(df_y_awake_wear$MIMS_UNIT, df_y_awake_wear$date)

  ## Keep days with >= 480 minutes (8 hours) of valid wear time
  day_list <- Filter(function(v) length(v) >= 480, day_list)

  ## Keep weekdays
  # is_weekday <- as.POSIXlt(names(day_list))$wday %in% 1:5
  # day_list <- day_list[is_weekday]

  ## If there are not more than 3 unique values, monotonic spline interpolation
  ## will fail. Remove offending days.
  day_n_unique <- sapply(day_list, function(y) length(unique(y)))
  day_list <- day_list[day_n_unique > 3]

  if (length(day_list) > 0) {
    y_list[[sub_id]] <- day_list
  }
  i <- i + 1
  
}
saveRDS(y_list, path_out)

## Aggregte across days
path_out <- file.path('data', 'processed', 'chop-mims_v1.rds')
y_list_sub <- list()
for (sub_id in names(y_list)) {
  y_list_sub[[sub_id]] <- unlist(y_list[[sub_id]])
} 
saveRDS(y_list_sub, path_out)


## ---------- MIMS (for NHANES Validation; wear time) ---------- ##

## Prepare day-level data
path_out <- file.path('data', 'processed', 'chop-mims-day_v6.rds')
y_list <- list()
paths_in <- list.files(path = dir_mims, full.names = TRUE)
i <- 1
for (path_y in paths_in) {
  print(str_glue("i = {i}"))

  ## Extract sub_id
  sub_id <- strsplit(tail(strsplit(path_y, "/")[[1]], n = 1), '_')[[1]][1]
  
  ## Set paths
  # path_sleep <- file.path(
  #   dir_sleep, get_data_file(
  #     sub_id, 'sleep',
  #     rewear = ifelse(sub_id %in% rewear_ids, TRUE, FALSE)
  #   )
  # )
  # if (!file.exists(path_sleep)) {
  #   print(str_glue("File does not exist: {path_sleep}"))
  #   next
  # }
  path_nonwear <- file.path(
    dir_nonwear, get_data_file(
      sub_id, 'nonwear',
      rewear = ifelse(sub_id %in% rewear_ids, TRUE, FALSE)
    )
  )
  if (!file.exists(path_nonwear)) {
    print(str_glue("File does not exist: {path_nonwear}"))
    next
  }
  
  ## Read dataframes
  df_y <- read.csv(path_y, header = TRUE)
  # df_sleep <- read.csv(path_sleep, header = TRUE)
  df_nonwear <- read.csv(path_nonwear, header = TRUE)
  
  ## Parse timestamps
  df_y <- df_y %>%
    mutate(
      ts = as.POSIXct(HEADER_TIME_STAMP, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
    )
  # df_sleep <- df_sleep %>%
  #   mutate(
  #     sleeponset_ts = as.POSIXct(sleeponset_ts, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  #     wakeup_ts     = as.POSIXct(wakeup_ts,     format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  #   )
  df_nonwear <- df_nonwear %>%
    mutate(
      ts_start = as.POSIXct(timestamp, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
      ts_end   = ts_start + 15 * 60   # 15-minute epochs
    )
  
  ## Remove rows inside sleep intervals
  df_nonwear_int <- df_nonwear %>%
    filter(nonwear == 1)
  df_y_wear <- df_y %>%

    ## Clamp negatives to zero and remove unreasonably high values
    filter(MIMS_UNIT != -0.01, MIMS_UNIT <= 500) %>%
    
    ## Remove sleep
    # left_join(
    #   df_sleep,
    #   join_by(
    #     ts >= sleeponset_ts,
    #     ts <= wakeup_ts
    #   )
    # ) %>%
    # filter(is.na(sleeponset_ts)) %>%
    # dplyr::select(-sleeponset_ts, -wakeup_ts) %>%
    
    ## Remove nonwear
    left_join(
      df_nonwear_int,
      join_by(
        ts >= ts_start,
        ts <  ts_end
      )
    ) %>%
    filter(is.na(ts_start)) %>%
    dplyr::select(-ts_start, -ts_end, -nonwear)

  ## Define distributions (by day, using Eastern time)
  df_y_wear <- df_y_wear %>%
    mutate(date = as.character(as.Date(ts, tz = "America/New_York")))
  day_list <- split(df_y_wear$MIMS_UNIT, df_y_wear$date)

  ## Keep days with >= 480 minutes (8 hours) of valid wear time
  day_list <- Filter(function(v) length(v) >= 480, day_list)

  ## Keep weekdays
  # is_weekday <- as.POSIXlt(names(day_list))$wday %in% 1:5
  # day_list <- day_list[is_weekday]

  ## If there are not more than 3 unique values, monotonic spline interpolation
  ## will fail. Remove offending days.
  day_n_unique <- sapply(day_list, function(y) length(unique(y)))
  day_list <- day_list[day_n_unique > 3]

  ## Require exactly 7 days: drop subjects with fewer than 7 qualifying days
  ## and truncate subjects with more than 7 to the first 7 (chronological).
  if (length(day_list) >= 7) {
    day_list <- day_list[order(names(day_list))][1:7]
    y_list[[sub_id]] <- day_list
  }
  i <- i + 1

}
saveRDS(y_list, path_out)

## Aggregte across days
path_out <- file.path('data', 'processed', 'chop-mims_v6.rds')
y_list_sub <- list()
for (sub_id in names(y_list)) {
  y_list_sub[[sub_id]] <- unlist(y_list[[sub_id]])
} 
saveRDS(y_list_sub, path_out)


## ---------- MIMS (for NHANES Validation; awake wear time) ---------- ##

## Prepare day-level data
path_out <- file.path('data', 'processed', 'chop-mims-day_v5.rds')
y_list <- list()
paths_in <- list.files(path = dir_mims, full.names = TRUE)
i <- 1
for (path_y in paths_in) {
  print(str_glue("i = {i}"))

  ## Extract sub_id
  sub_id <- strsplit(tail(strsplit(path_y, "/")[[1]], n = 1), '_')[[1]][1]
  
  ## Set paths
  path_sleep <- file.path(
    dir_sleep, get_data_file(
      sub_id, 'sleep',
      rewear = ifelse(sub_id %in% rewear_ids, TRUE, FALSE)
    )
  )
  if (!file.exists(path_sleep)) {
    print(str_glue("File does not exist: {path_sleep}"))
    next
  }
  path_nonwear <- file.path(
    dir_nonwear, get_data_file(
      sub_id, 'nonwear',
      rewear = ifelse(sub_id %in% rewear_ids, TRUE, FALSE)
    )
  )
  if (!file.exists(path_nonwear)) {
    print(str_glue("File does not exist: {path_nonwear}"))
    next
  }
  
  ## Read dataframes
  df_y <- read.csv(path_y, header = TRUE)
  df_sleep <- read.csv(path_sleep, header = TRUE)
  df_nonwear <- read.csv(path_nonwear, header = TRUE)
  
  ## Parse timestamps
  df_y <- df_y %>%
    mutate(
      ts = as.POSIXct(HEADER_TIME_STAMP, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
    )
  df_sleep <- df_sleep %>%
    mutate(
      sleeponset_ts = as.POSIXct(sleeponset_ts, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
      wakeup_ts     = as.POSIXct(wakeup_ts,     format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
    )
  df_nonwear <- df_nonwear %>%
    mutate(
      ts_start = as.POSIXct(timestamp, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
      ts_end   = ts_start + 15 * 60   # 15-minute epochs
    )
  
  ## Remove rows inside sleep intervals
  df_nonwear_int <- df_nonwear %>%
    filter(nonwear == 1)
  df_y_awake_wear <- df_y %>%

    ## Clamp negatives to zero and remove unreasonably high values
    filter(MIMS_UNIT != -0.01, MIMS_UNIT <= 500) %>%
    
    ## Remove sleep
    left_join(
      df_sleep,
      join_by(
        ts >= sleeponset_ts,
        ts <= wakeup_ts
      )
    ) %>%
    filter(is.na(sleeponset_ts)) %>%
    dplyr::select(-sleeponset_ts, -wakeup_ts) %>%
    
    ## Remove nonwear
    left_join(
      df_nonwear_int,
      join_by(
        ts >= ts_start,
        ts <  ts_end
      )
    ) %>%
    filter(is.na(ts_start)) %>%
    dplyr::select(-ts_start, -ts_end, -nonwear)

  ## Define distributions (by day, using Eastern time)
  df_y_awake_wear <- df_y_awake_wear %>%
    mutate(date = as.character(as.Date(ts, tz = "America/New_York")))
  day_list <- split(df_y_awake_wear$MIMS_UNIT, df_y_awake_wear$date)

  ## Keep days with >= 480 minutes (8 hours) of valid wear time
  day_list <- Filter(function(v) length(v) >= 480, day_list)

  ## Keep weekdays
  # is_weekday <- as.POSIXlt(names(day_list))$wday %in% 1:5
  # day_list <- day_list[is_weekday]

  ## If there are not more than 3 unique values, monotonic spline interpolation
  ## will fail. Remove offending days.
  day_n_unique <- sapply(day_list, function(y) length(unique(y)))
  day_list <- day_list[day_n_unique > 3]

  ## Require exactly 7 days: drop subjects with fewer than 7 qualifying days
  ## and truncate subjects with more than 7 to the first 7 (chronological).
  if (length(day_list) >= 7) {
    day_list <- day_list[order(names(day_list))][1:7]
    y_list[[sub_id]] <- day_list
  }
  i <- i + 1

}
saveRDS(y_list, path_out)

## Aggregte across days
path_out <- file.path('data', 'processed', 'chop-mims_v5.rds')
y_list_sub <- list()
for (sub_id in names(y_list)) {
  y_list_sub[[sub_id]] <- unlist(y_list[[sub_id]])
} 
saveRDS(y_list_sub, path_out)



## ---------- MIMS (for NHANES validation; all time) ---------- ##

## Processing
##  - Remove -0.01 values
##  - How many exceedingly large MIMS observations are there?

## Prepare day-level data (minimal processing)
path_out <- file.path('data', 'processed', 'chop-mims-day_v3_nofilter.rds')
y_list <- list()

paths_in <- list.files(path = dir_mims, full.names = TRUE)
i <- 1
for (path_y in paths_in) {
  print(str_glue("i = {i}"))

  ## Extract sub_id
  sub_id <- strsplit(tail(strsplit(path_y, "/")[[1]], n = 1), '_')[[1]][1]

  ## Read data
  df_y <- read.csv(path_y, header = TRUE) %>%
    mutate(
      ts = as.POSIXct(HEADER_TIME_STAMP, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
      date = as.character(as.Date(ts, tz = "America/New_York"))
    ) %>%
    filter(MIMS_UNIT != -0.01, MIMS_UNIT <= 500)  ## Outside physiologically reasonable range (need citation)

  ## Split by day and keep days with >= 480 minutes (8 hours)
  day_list <- split(df_y$MIMS_UNIT, df_y$date)
  day_list <- Filter(function(v) length(v) >= 480, day_list)

  if (length(day_list) >= 7) {
    y_list[[sub_id]] <- day_list[1:7]
  }
  i <- i + 1

}
saveRDS(y_list, path_out)

## Aggregate across days
path_out <- file.path('data', 'processed', 'chop-mims_v3_nofilter.rds')
y_list_sub <- list()
for (sub_id in names(y_list)) {
  y_list_sub[[sub_id]] <- unlist(y_list[[sub_id]])
}
saveRDS(y_list_sub, path_out)



## ---------- ENMO (awake wear time) ---------- ##

## TODO: If using ENMO, don't forget to...
##  - Clamp at minimum and unreasonable ENMO units
##  - Use eastern timezone dates

## Prepare day-level data
path_out <- file.path('data', 'processed', 'chop-enmo-day_v1.rds')
y_list <- list()
paths_in <- list.files(path = dir_enmo, full.names = TRUE)
i <- 1
for (path_y in paths_in) {
  print(str_glue("i = {i}"))

  ## Extract sub_id
  sub_id <- strsplit(tail(strsplit(path_y, "/")[[1]], n = 1), '_')[[1]][1]
  
  ## Set paths
  path_sleep <- file.path(
    dir_sleep, get_data_file(
      sub_id, 'sleep',
      rewear = ifelse(sub_id %in% rewear_ids, TRUE, FALSE)
    )
  )
  if (!file.exists(path_sleep)) {
    print(str_glue("File does not exist: {path_sleep}"))
    next
  }
  path_nonwear <- file.path(
    dir_nonwear, get_data_file(
      sub_id, 'nonwear',
      rewear = ifelse(sub_id %in% rewear_ids, TRUE, FALSE)
    )
  )
  if (!file.exists(path_nonwear)) {
    print(str_glue("File does not exist: {path_nonwear}"))
    next
  }
  
  ## Read dataframes
  df_y <- read.csv(path_y, header = TRUE)
  df_sleep <- read.csv(path_sleep, header = TRUE)
  df_nonwear <- read.csv(path_nonwear, header = TRUE)
  
  ## Parse timestamps
  df_y <- df_y %>%
    mutate(
      ts = as.POSIXct(timestamp, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
    )
  df_sleep <- df_sleep %>%
    mutate(
      sleeponset_ts = as.POSIXct(sleeponset_ts, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
      wakeup_ts     = as.POSIXct(wakeup_ts,     format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
    )
  df_nonwear <- df_nonwear %>%
    mutate(
      ts_start = as.POSIXct(timestamp, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
      ts_end   = ts_start + 15 * 60   # 15-minute epochs
    )
  
  ## Remvoe rows inside sleep intervals
  df_nonwear_int <- df_nonwear %>%
    filter(nonwear == 1)
  df_y_awake_wear <- df_y %>%
    
    ## Remove sleep
    left_join(
      df_sleep,
      join_by(
        ts >= sleeponset_ts,
        ts <= wakeup_ts
      )
    ) %>%
    filter(is.na(sleeponset_ts)) %>%
    dplyr::select(-sleeponset_ts, -wakeup_ts) %>%
    
    ## Remove nonwear
    left_join(
      df_nonwear_int,
      join_by(
        ts >= ts_start,
        ts <  ts_end
      )
    ) %>%
    filter(is.na(ts_start)) %>%
    dplyr::select(-ts_start, -ts_end, -nonwear)
  
  ## Define distributions (by day)
  df_y_awake_wear <- df_y_awake_wear %>%
    mutate(date = as.character(as.Date(ts)))
  day_list <- split(df_y_awake_wear$ENMO, df_y_awake_wear$date)

  ## Keep days with >= 480 minutes of valid wear time
  day_list <- Filter(function(v) length(v) >= 480, day_list)

  ## Keep weekdays
  # is_weekday <- as.POSIXlt(names(day_list))$wday %in% 1:5
  # day_list <- day_list[is_weekday]

  if (length(day_list) > 0) {
    y_list[[sub_id]] <- day_list
  }
  i <- i + 1
  
}
saveRDS(y_list, path_out)

## Aggregte across days
path_out <- file.path('data', 'processed', 'chop-enmo_v1.rds')
y_list_sub <- list()
for (sub_id in names(y_list)) {
  y_list_sub[[sub_id]] <- unlist(y_list[[sub_id]])
} 
saveRDS(y_list_sub, path_out)



## ---------- Sleep ---------- ##

path_out <- file.path('data', 'processed', 'chop-sleep-day_v1.rds')
df <- data.frame(
  sub_id = character(),
  date = character(),
  sleep_dur = numeric(),
  sleep_eff = numeric()
)
paths_in <- list.files(path = dir_sleep, full.names = TRUE)
i <- 1
for (path_sleep in paths_in) {
  print(str_glue("i = {i}"))

  ## Extract sub_id
  sub_id <- strsplit(tail(strsplit(path_sleep, "/")[[1]], n = 1), '_')[[1]][1]

  df_i <- read.csv(path_sleep, header = TRUE) %>%
    rename(
      sleep_dur = SleepDurationInSpt
    ) %>%
    mutate(
      date = as.character(as.Date(as.POSIXct(date, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"))),
      sleep_eff = sleep_dur / (sleep_dur + WASO)
    ) %>%
    dplyr::select(date, sleep_dur, sleep_eff)
  df_i$sub_id <- sub_id
  df_i <- df_i[,c('sub_id', 'date', 'sleep_dur', 'sleep_eff')]

  df <- rbind(df, df_i)

  i <- i + 1
}
saveRDS(df, path_out)

## Aggregate across days
path_out <- file.path('data', 'processed', 'chop-sleep_v1.rds')
df_sub <- df %>%
  mutate(
    tot_hours = sleep_dur / sleep_eff
  ) %>%
  group_by(sub_id) %>%
  summarize(
    sleep_eff = sum(sleep_dur) / sum(tot_hours),
    sleep_dur = mean(sleep_dur)
  ) %>%
  dplyr::select(sub_id, sleep_dur, sleep_eff)
saveRDS(df_sub, path_out)



## ---------- Non-Wear (Day-level) ---------- ##

path_out <- file.path('data', 'processed', 'chop-nonwear-day_v1.rds') ## cols: subject_id, date, nonwear
df <- data.frame(
  sub_id = character(),
  date = character(),
  nonwear = numeric(),
  n_epochs = integer()
)
paths_in <- list.files(path = dir_nonwear, full.names = TRUE)
i <- 1
for (path_nonwear in paths_in) {
  print(str_glue("i = {i}"))

  ## Extract sub_id
  sub_id <- strsplit(tail(strsplit(path_nonwear, "/")[[1]], n = 1), '_')[[1]][1]

  df_i <- read.csv(path_nonwear, header = TRUE) %>%
    mutate(
      date = as.character(as.Date(as.POSIXct(timestamp, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")))
    ) %>%
    group_by(date) %>%
    summarize(
      nonwear = mean(nonwear, na.rm = TRUE),
      n_epochs = n()
    ) %>%
    dplyr::select(date, nonwear, n_epochs)

  df_i$sub_id <- sub_id
  df <- rbind(df, df_i[,c('sub_id', 'date', 'nonwear', 'n_epochs')])

  i <- i + 1
}
saveRDS(df[,c('sub_id', 'date', 'nonwear')], path_out)

## Aggregate nonwear across days (weighted by epochs per day)
path_out <- file.path('data', 'processed', 'chop-nonwear_v1.rds')
df_sub <- df %>%
  group_by(sub_id) %>%
  summarize(nonwear = weighted.mean(nonwear, n_epochs))
saveRDS(df_sub, path_out)


## ---------- MVPA (Manual via ENMO) ---------- ##

## Day-level
path_out <- file.path('data', 'processed', 'chop-mvpa-day_v1.rds')
path_in_y <- file.path('data', 'processed', 'chop-enmo-day_v1.rds')
y_list <- readRDS(path_in_y)
df <- data.frame(
  sub_id = character(),
  date = character(),
  mvpa_rate = numeric()
)
for (sub_id in names(y_list)) {
  for (date in names(y_list[[sub_id]])) {
    mvpa_rate <- mean(y_list[[sub_id]][[date]] > 0.1)
    row <- data.frame(sub_id = sub_id, date = date, mvpa_rate = mvpa_rate)
    df <- rbind(df, row)
  }
}
saveRDS(df, path_out)

## Subject-level
path_out <- file.path('data', 'processed', 'chop-mvpa_v1.rds')
path_in_y <- file.path('data', 'processed', 'chop-enmo_v1.rds')
y_list <- readRDS(path_in_y)
df <- data.frame(
  sub_id = character(),
  mvpa_rate = numeric()
)
for (sub_id in names(y_list)) {
  mvpa_rate <- mean(y_list[[sub_id]] > 0.1)
  row <- data.frame(sub_id = sub_id, mvpa_rate = mvpa_rate)
  df <- rbind(df, row)
}
saveRDS(df, path_out)



## ---------- Other Predictors ---------- ##

path_out <- file.path('data', 'processed', 'chop_misc-preds.rds')
path_1 <- file.path(dir_demo_dxa, 'age_sex_race_for_Zscore_calc.csv')
path_2 <- file.path(dir_demo_dxa, 'Anthropometry.csv')
path_3 <- file.path(dir_demo_dxa, 'CRF_PAVS.csv')
path_4 <- file.path(dir_demo_dxa, 'DXA_Body_Comp_Fat_Mass.csv')
path_5 <- file.path(dir_demo_dxa, 'SGrow2_BMD_tot_hip_sp_20260320.csv')
path_6 <- file.path(dir_demo_dxa, 'SGrow2_High_Low_Impact_PA.csv')
df_1 <- read.csv(path_1, header = TRUE)
df_2 <- read.csv(path_2, header = TRUE)
df_3 <- read.csv(path_3, header = TRUE)
df_4 <- read.csv(path_4, header = TRUE)
df_5 <- read.csv(path_5, header = TRUE)
df_6 <- read.csv(path_6, header = TRUE)
df <- df_1 %>%
  inner_join(df_2, by = 'subject_id') %>%
  inner_join(df_3, by = 'subject_id') %>%
  inner_join(df_4, by = 'subject_id') %>%
  inner_join(df_5, by = 'subject_id') %>%
  inner_join(df_6, by = 'subject_id') %>%
  mutate(
    subject_id = as.character(subject_id),
    age_cat = sapply(cdemo_age, function(age) ifelse(age <= 12, '<=12', '>12'))
  ) %>%
  rename(sub_id = subject_id, sex = cdemo_sex, bmiz = cBMIz) %>%
  filter(sex %in% c(1, 2)) %>%
  dplyr::select(
    sub_id, 
    age_cat, 
    sex, 
    bmiz, 
    fmi_all_z,
    subtot_bmd_age_z_all,
    hip_neck_bmd_age_z_all,
    spine_bmd_age_z_all,
    spine_bmad_age_z_all,
    tot_hip_bmd_age_z_all,
    radius13_bmd_age_z_all,
    udradius_bmd_age_z_all,
    impact01_hr_day,
    impact23_hr_day
  ) # %>%
  # drop_na()
saveRDS(df, path_out)

