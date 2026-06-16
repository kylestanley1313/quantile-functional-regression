library(dplyr)
library(hms)
library(lubridate)
library(tidyr)

## ========== Preprocessing V1 ========== ##

# path_in <- file.path('data', 'raw', 'tean_accelerometer_final_v2.csv')
# path_out <- file.path('data', 'processed', 'tean_v1.rds')
# 
# ## Read data
# df_data <- read.csv(path_in, header = TRUE)
# 
# ## Get "valid" days and replace '-999' placeholder with NA
# df_data <- df_data[df_data$VALID == 1,]
# df_data[df_data == -999] <- NA
# 
# ## Get subject distributions (aggregate across days)
# ids <- unique(df_data$ID)
# y_list <- list()
# for (id in ids) {
#   df_data_id <- df_data[df_data$ID == id,5:ncol(df_data)]
#   n_days <- nrow(df_data_id)
#   if (n_days > 0) {
#     y_list[[id]] <- c()
#     for (d in 1:n_days) {
#       tmp <- df_data_id[d,]
#       mask <- !is.na(tmp)
#       y_list[[id]] <- c(y_list[[id]], tmp[mask])
#     }
#   }
# }
# 
# ## Save
# saveRDS(y_list, path_out)



## ========== Preprocessing V2 ========== ##


## ---------- Accelerometer Data

path_in <- file.path('data', 'raw', '30sec_raw_accelerometer.csv')
path_out <- file.path('data', 'processed', 'tean_v2.rds')

df <- read.csv(path_in, header = TRUE)
df1 <- df %>%
  drop_na(Participant_ID) %>%
  mutate(
    Participant_ID = as.character(Participant_ID),
    Date = lubridate::ymd(Date),
    Hour = as.integer(Hour),
    Minute = as.integer(Minute),
    Time_Period = as.integer(Time_Period),
    sec = if_else(Time_Period == 60L, 0L, Time_Period),
    hour_adj = if_else(Hour == 24L & Minute == 0L & sec == 0L, 0L, Hour),
    date_adj = if_else(Hour == 24L & Minute == 0L & sec == 0L, Date + lubridate::days(1), Date),
    
    Time = hms::hms(hours = hour_adj, minutes = Minute, seconds = sec),
    Date = date_adj,
    
    # Build datetime by adding seconds since midnight to Date
    datetime = as.POSIXct(Date) + as.numeric(Time)  # seconds
    # If you prefer an explicit timezone: as.POSIXct(Date, tz = "UTC") + as.numeric(Time)
  )

# 1) Keep waking hours only: 06:00:00 to 23:30:00 (inclusive)
df_wake <- df1 %>%
  filter(Time >= as_hms("06:00:00"), Time <= as_hms("23:30:00"))

# 2) Weekdays only (Mon–Fri)
df_weekdays <- df_wake %>%
  filter(!wday(Date) %in% c(1, 7))

# 3) Remove non-wear (≥60 consecutive 30-sec epochs with Activity == 0)
df_wear <- df_weekdays %>%
  arrange(Participant_ID, Date, datetime) %>%
  group_by(Participant_ID, Date) %>%
  mutate(
    is_zero = Activity == 0,
    seq_id = cumsum(is_zero != lag(is_zero, default = first(is_zero)))
  ) %>%
  group_by(Participant_ID, Date, seq_id) %>%
  mutate(
    seq_len = n(),
    is_nonwear = is_zero & seq_len >= 60
  ) %>%
  ungroup() %>%
  filter(!is_nonwear) %>%
  select(-is_zero, -seq_id, -seq_len, -is_nonwear)

# 4) Valid day: ≥ 8 hours wear after non-wear removal
day_summary <- df_wear %>%
  group_by(Participant_ID, Date) %>%
  summarise(
    wear_epochs = n(),
    wear_hours = wear_epochs * 30 / 3600,
    valid_day = wear_hours >= 8,
    .groups = "drop"
  )

df_valid_days <- df_wear %>%
  inner_join(day_summary %>% filter(valid_day) %>% select(Participant_ID, Date),
             by = c("Participant_ID", "Date"))

# 5) Valid subject: ≥ 3 valid days
valid_subjects <- day_summary %>%
  filter(valid_day) %>%
  count(Participant_ID, name = "n_valid_days") %>%
  filter(n_valid_days >= 3)

df_final <- df_valid_days %>%
  inner_join(valid_subjects %>% select(Participant_ID), by = "Participant_ID") %>%
  arrange(Participant_ID, Date, datetime)

## NOTE: (Preprocessing)
##  - Restrict analysis to waking hours: 6:00am - 11:30pm
##  - Segments with >= 60 consecutive epochs with Activity = 0 considered nonwear
##  - Include only "valid" days (i.e., days with >= 8 hours of valid wear)
##  - Include only "valid" subjects (i.e., subjects with >= 3 "valid" days)

## Get subject distributions (aggregate across days)
ids <- unique(df_final$Participant_ID)
y_list <- list()
for (id in ids) {
  df_final_id <- filter(df_final, Participant_ID == id)
  y_list[[id]] <- df_final_id$Activity
}

## Save
saveRDS(y_list, path_out)



## ---------- Covariates

path_in_cov <- file.path('data', 'raw', '30sec_raw_covariates.csv')
path_out_cov <- file.path('data', 'processed', 'tean_v2_cov.rds')
df_cov <- read.csv(path_in_cov, header = TRUE)
df_cov <- df_cov %>%
  mutate(Participant_ID = as.character(Participant_ID)) %>%
  rename(
    id = Participant_ID,
    site = Site,
    age = Child_age_Cov,
    gender = Child_gender_Cov,
    bmi_pct = C_BMIPCT_CDC
  ) %>%
  dplyr::select(id, site, age, gender, bmi_pct)
saveRDS(df_cov, path_out_cov)


## ---------- Subsampling

N <- 250
set.seed(12345)
path_in <- file.path('data', 'processed', 'tean_v2.rds')
path_out <- file.path('data', 'processed', str_glue('tean_v2_N-{N}.rds'))
y_list <- readRDS(path_in)
idx <- sample(1:length(y_list), N)
y_list <- y_list[idx]
saveRDS(y_list, path_out)

