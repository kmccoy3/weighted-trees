

# Load necessary libraries
library(tidyverse)
library(BART)

# Read in results from runs
prior_results <- read_csv("./out/2025-06-18_14-41mu_3.csv")

mu = 3

# Loop over various values of n, observations per group
for (n in c(20, 50, 100, 500)){

    # Loop over 20 seeds
    for (seed in 0:19){

        # Set seed
        set.seed(seed)
        
        # Read in the data
        filename <- paste0("./syn_data/data_n-", n, "-Seed-", seed, "-mu-", mu, ".csv")

        df <- read_csv(filename) %>% 
            select(c("X1", "X2", "X3", "X4", "X5", "fake", "fake2", "fake3", "fake4", "fake5", "group", "Y")) %>%
            mutate(group = as.factor(group))
        
        # Split into training and test sets
        df_train <- df %>%
            filter(as.numeric(group) <= 16)

        df_test <- df %>%
            filter(as.numeric(group) > 16)
        
        
        start.time <- Sys.time()
        
        # Fit the BART model
        bart_model <- BART::wbart(
            x.train = as.data.frame(df_train %>% select(-c(Y, group))),
            y.train = df_train$Y,
            x.test = as.data.frame(df_test %>% select(-c(Y, group))),
            ntree = 16
        )
        
        end.time <- Sys.time()
        
        # Calculate elapsed time
        elapsed.time <- end.time - start.time
        
        # Store the results
        prior_results <- prior_results %>%
            add_row(
                method = "BART",
                n = n,
                seed = seed,
                time = as.numeric(elapsed.time),
                MSE = mean((df_test$Y - bart_model$yhat.test.mean)^2),
            )
        
        

    
    }
}

new_df9 <- prior_results
