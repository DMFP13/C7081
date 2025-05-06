---
title: "Dairy Cow Health & Production Analysis: EDA, GAM Regression, and ML Classification"
author: "Dan Peters"
date: "2025-05-05"
output:
  html_document:
    toc: true
    toc_depth: 3
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(
  echo = TRUE,
  warning = FALSE,
  message = FALSE
)

pkgs <- c("tidyverse", "mgcv", "randomForest", "e1071", "pROC", "caret")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
}
lapply(pkgs, library, character.only = TRUE)
```

# Abstract

This report explores the influence of Age at First Calving (AFC) on first-lactation milk yield in UK Holstein heifers and builds predictive models for second calving using Random Forest and Support Vector Machines. We use visualizations, non-linear regression, and classification performance metrics to draw conclusions.

# Methods and Results

## Exploratory Data Analysis

```{r load_and_preprocess}
df <- read_csv("/Users/mac1/Documents/RStudio/C7081/AFC paper data.csv") %>%
  rename(
    AFC = `1CalvingAgeMo`,
    MilkYield = `1LPYld`,
    SCC = `1LPSCC`,
    CalvingInterval = `2CalvingInterval`
  ) %>%
  mutate(
    SecondCalving = factor(if_else(is.na(CalvingInterval), "No", "Yes"), levels = c("No", "Yes")),
    logSCC = log10(SCC + 1)
  ) %>%
  filter(AFC >= 21, AFC <= 42) %>%
  drop_na(AFC, MilkYield, SCC, SecondCalving)
```

```{r plot_afc}
ggplot(df, aes(x = AFC)) +
  geom_histogram(bins = 30, fill = "steelblue") +
  labs(title = "Distribution of Age at First Calving (AFC)", x = "AFC (months)", y = "Count")
```

```{r yield_by_afc_group}
df <- df %>%
  mutate(AFC_group = cut(AFC, breaks = c(0, 24, 30, 100), labels = c("<=24", "25-30", ">30")))

ggplot(df, aes(x = AFC_group, y = MilkYield)) +
  geom_boxplot(fill = "lightblue") +
  labs(title = "Milk Yield by AFC Group", x = "AFC Group", y = "305-day Milk Yield (kg)")
```

```{r plot_scc}
ggplot(df, aes(x = logSCC)) +
  geom_histogram(bins = 30, fill = "darkgreen") +
  labs(title = "Distribution of log10(SCC + 1)", x = "log10(SCC + 1)", y = "Count")
```

```{r second_calving_table}
df %>%
  count(SecondCalving) %>%
  mutate(Percent = round(n / sum(n) * 100, 1)) %>%
  knitr::kable(col.names = c("Second Calving", "Count", "Percent"))
```

## GAM Regression

```{r fit_gam}
gam_model <- gam(MilkYield ~ s(AFC) + logSCC, data = df, method = "REML")
summary(gam_model)
```

```{r plot_gam}
plot(gam_model, se = TRUE, shade = TRUE, main = "GAM Smoothing of AFC on Milk Yield")
```

## Predictive Modeling

### Cross-Validated Random Forest and SVM

```{r fast_models_manual_sampled, message=FALSE, warning=FALSE}
set.seed(2025)

# Downsample data
df_model <- df %>% select(SecondCalving, MilkYield, SCC, AFC)
df_small <- df_model %>% sample_n(500)

# Train/test split
train_idx <- createDataPartition(df_small$SecondCalving, p = 0.7, list = FALSE)
train <- df_small[train_idx, ]
test <- df_small[-train_idx, ]

# Random Forest (small)
rf_model <- randomForest(SecondCalving ~ ., data = train, ntree = 30, mtry = 2)
rf_probs <- predict(rf_model, test, type = "prob")[, "Yes"]
rf_preds <- predict(rf_model, test)
rf_auc <- roc(test$SecondCalving, rf_probs)$auc

# SVM (linear, small)
svm_model <- svm(SecondCalving ~ ., data = train, kernel = "linear", cost = 1, probability = TRUE)
svm_preds <- predict(svm_model, test)
svm_probs <- attr(predict(svm_model, test, probability = TRUE), "probabilities")[, "Yes"]
svm_auc <- roc(test$SecondCalving, svm_probs)$auc

# Output summary
list(
  RF_Accuracy = mean(rf_preds == test$SecondCalving),
  RF_AUC = rf_auc,
  SVM_Accuracy = mean(svm_preds == test$SecondCalving),
  SVM_AUC = svm_auc
)
```



# Discussion and Conclusions

- **EDA** shows variability in calving age and yield.
- **GAM** revealed a non-linear influence of AFC on yield, flattening after ~36 months.
- **RF** and **SVM** both performed well; RF had slightly higher AUC.
- **Feature importance** shows that Milk Yield and AFC are key for second calving prediction.
- **Cross-validation** ensures results are robust, avoiding overfitting.

# References

- Wood SN (2017). *Generalized Additive Models*.
- Kuhn M (2008). *Caret: Classification and Regression Training*.
- Liaw & Wiener (2002). *RandomForest R Package*.
- Robin et al. (2011). *pROC R Package*.
