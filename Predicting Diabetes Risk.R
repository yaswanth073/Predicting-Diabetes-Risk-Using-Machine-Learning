# =============================================================================
# CSP 571 - Machine Learning Project
# Predicting Diabetes Risk Using Machine Learning
#
# Team: Yaswanth Moapda, Srujith Borra, Uday Ankam
# Spring 2025
#
# Dataset: CDC BRFSS 2015 Diabetes Health Indicators (50/50 balanced split)
# Source : https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
# File   : diabetes_binary_5050split_health_indicators_BRFSS2015.csv
#
# How to run:
#   setwd("~/Desktop")
#   source("Diabetes_Complete.R")
#
# Pipeline steps:
#   1. Setup       — install/load packages
#   2. Preprocess  — load, dedup, balance, encode, standardize, split
#   3. EDA         — 10 plots in 2 windows
#   4. Train       — 9 models trained
#   5. Evaluate    — full metrics on held-out test set
#   6. Display     — 5 result plots in window 3 + console table
# =============================================================================

rm(list = ls()); gc()
options(stringsAsFactors = FALSE, scipen = 999)
set.seed(571)

# --- change this path if csv is elsewhere ---
CSV_PATH <- "~/Desktop/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"

cat("================================================================\n")
cat(" CSP 571 — Diabetes Risk Prediction Pipeline\n")
cat(" Working dir :", getwd(), "\n")
cat("================================================================\n\n")


# ---------------------------------------------------------------
# STEP 1 — load all packages we need
# ---------------------------------------------------------------
cat("[1/6] Loading packages ...\n")

pkgs <- c("data.table","dplyr","ggplot2","tidyr","scales","tibble",
          "caret","glmnet","randomForest","gbm","e1071","nnet",
          "xgboost","class","rpart","rpart.plot",
          "pROC","corrplot","gridExtra","grid","png")

new_pkgs <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(new_pkgs)) {
  cat("  Installing:", paste(new_pkgs, collapse = ", "), "\n")
  install.packages(new_pkgs, repos = "https://cloud.r-project.org", quiet = TRUE)
}
invisible(lapply(pkgs, function(p)
  suppressPackageStartupMessages(library(p, character.only = TRUE))))

cat("[1/6] Done.\n\n")


# ---------------------------------------------------------------
# STEP 2 — load and clean the data
# ---------------------------------------------------------------
cat("[2/6] Loading and preprocessing ...\n")

if (!file.exists(CSV_PATH))
  stop("Cannot find CSV at: ", CSV_PATH,
       "\nSet CSV_PATH at the top of this script to the correct location.")

df <- as.data.frame(data.table::fread(CSV_PATH))
cat(sprintf("  Raw            : %d rows x %d cols\n", nrow(df), ncol(df)))

# remove exact duplicate rows
n0 <- nrow(df); df <- df[!duplicated(df), ]
cat(sprintf("  After dedup    : %d rows (%d removed)\n", nrow(df), n0 - nrow(df)))

# downsample to perfect 50/50 (file is approximately balanced but not exact)
min_n <- min(table(df$Diabetes_binary))
df <- df %>% group_by(Diabetes_binary) %>%
  slice_sample(n = min_n) %>% ungroup() %>% as.data.frame()
cat(sprintf("  After balance  : %d rows (50/50)\n", nrow(df)))

# target as factor with readable labels
df$Diabetes_binary <- factor(df$Diabetes_binary, levels = c(0,1),
                             labels = c("No","Yes"))

# 14 binary predictors -> factor
bin_cols <- c("HighBP","HighChol","CholCheck","Smoker","Stroke",
              "HeartDiseaseorAttack","PhysActivity","Fruits","Veggies",
              "HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","DiffWalk","Sex")
for (v in bin_cols)
  if (v %in% names(df)) df[[v]] <- factor(df[[v]], levels = c(0,1))

# 7 continuous predictors -> z-score standardize
cont_cols <- intersect(c("BMI","GenHlth","MentHlth","PhysHlth",
                         "Age","Education","Income"), names(df))
for (v in cont_cols) df[[v]] <- scale(df[[v]])[,1]

# stratified 70/30 train/test split
idx   <- caret::createDataPartition(df$Diabetes_binary, p = 0.70, list = FALSE)
train <- df[ idx, ]; test <- df[-idx, ]
cat(sprintf("  Train: %d rows | Test: %d rows | Features: %d\n",
            nrow(train), nrow(test), ncol(train)-1))
cat(sprintf("  Train balance  : No=%d  Yes=%d  (%.1f%% positive)\n\n",
            sum(train$Diabetes_binary=="No"),
            sum(train$Diabetes_binary=="Yes"),
            100*mean(train$Diabetes_binary=="Yes")))


# ---------------------------------------------------------------
# STEP 3 — build all 10 EDA plots
# helper that captures base graphics into a grob (for grid.arrange)
# ---------------------------------------------------------------
cat("[3/6] Building EDA plots ...\n")

th   <- theme_minimal(base_size = 9)
pal2 <- c("No"="#2E86AB","Yes"="#E63946")

base_grob <- function(draw_fn) {
  tmp <- tempfile(fileext = ".png")
  grDevices::png(tmp, width = 900, height = 700, res = 120)
  draw_fn()
  grDevices::dev.off()
  grid::rasterGrob(png::readPNG(tmp), interpolate = TRUE)
}

# plot 01 — class balance check
p01 <- ggplot(train, aes(Diabetes_binary, fill = Diabetes_binary)) +
  geom_bar() +
  geom_text(stat="count", aes(label=scales::comma(after_stat(count))),
            vjust=-0.4, size=3) +
  scale_fill_manual(values=pal2) +
  labs(title="01 — Class Balance", x="Diabetes", y="Count") +
  th + theme(legend.position="none")

# plot 02 — BMI density
p02 <- ggplot(train, aes(BMI, fill=Diabetes_binary)) +
  geom_density(alpha=0.55) +
  scale_fill_manual(values=pal2) +
  labs(title="02 — BMI Distribution", x="BMI (z-score)",
       y="Density", fill="Diabetes") + th

# plot 03 — diabetes rate by age bucket
age_df <- train %>%
  mutate(Age_bin = cut(Age, breaks=8)) %>%
  group_by(Age_bin) %>%
  summarise(rate = mean(Diabetes_binary=="Yes"), n=n(), .groups="drop")

p03 <- ggplot(age_df, aes(Age_bin, rate)) +
  geom_col(fill="#E63946", alpha=0.85) +
  geom_text(aes(label=sprintf("%.0f%%",100*rate)), vjust=-0.3, size=2.3) +
  scale_y_continuous(labels=scales::percent) +
  labs(title="03 — Diabetes Rate by Age", x="Age (binned)", y="Prevalence") +
  th + theme(axis.text.x=element_text(angle=45, hjust=1))

# plot 04 — correlation heatmap
num_df <- train
num_df$Diabetes_binary <- as.numeric(num_df$Diabetes_binary=="Yes")
for (v in names(num_df))
  if (is.factor(num_df[[v]])) num_df[[v]] <- as.numeric(as.character(num_df[[v]]))
corr_mat <- cor(num_df, use="pairwise.complete.obs")

p04 <- base_grob(function() {
  corrplot::corrplot(corr_mat, method="color", type="lower",
                     tl.col="black", tl.cex=0.55, tl.srt=45,
                     col=colorRampPalette(c("#2E86AB","white","#E63946"))(100),
                     title="04 — Correlation Matrix", mar=c(0,0,2,0))
})

# print top correlations with target (useful for the report)
cat("  Top features correlated with Diabetes_binary:\n")
target_cor <- sort(corr_mat["Diabetes_binary",], decreasing=TRUE)
print(round(head(target_cor[-1], 6), 3))
cat("\n")

# plot 05 — general health vs diabetes prevalence
gh_df <- train %>%
  group_by(g = round(GenHlth,1)) %>%
  summarise(rate=mean(Diabetes_binary=="Yes"), n=n(), .groups="drop") %>%
  filter(n>=50)

p05 <- ggplot(gh_df, aes(g, rate)) +
  geom_point(aes(size=n), color="#E63946", alpha=0.7) +
  geom_smooth(method="loess", se=TRUE, color="#2E86AB") +
  scale_y_continuous(labels=scales::percent) +
  labs(title="05 — General Health vs Diabetes", x="GenHlth (z-score)",
       y="Prevalence", size="n") + th

# plot 06 — boxplots of all continuous features
cl <- train %>%
  select(Diabetes_binary, BMI, GenHlth, MentHlth, PhysHlth,
         Age, Education, Income) %>%
  pivot_longer(-Diabetes_binary, names_to="Feature", values_to="Value")

p06 <- ggplot(cl, aes(Diabetes_binary, Value, fill=Diabetes_binary)) +
  geom_boxplot(outlier.size=0.3, alpha=0.7) +
  facet_wrap(~Feature, scales="free_y", ncol=4) +
  scale_fill_manual(values=pal2) +
  labs(title="06 — Continuous Features by Outcome",
       x="Diabetes", y="z-score") +
  th + theme(legend.position="none")

# plot 07 — binary risk factor prevalence
bvars <- c("HighBP","HighChol","Smoker","Stroke",
           "HeartDiseaseorAttack","PhysActivity","DiffWalk")
bl <- train %>%
  select(Diabetes_binary, all_of(bvars)) %>%
  mutate(across(all_of(bvars), ~as.numeric(as.character(.)))) %>%
  pivot_longer(-Diabetes_binary, names_to="Feature", values_to="Val") %>%
  group_by(Diabetes_binary, Feature) %>%
  summarise(rate=mean(Val==1), .groups="drop")

p07 <- ggplot(bl, aes(Feature, rate, fill=Diabetes_binary)) +
  geom_col(position="dodge", alpha=0.85) +
  geom_text(aes(label=sprintf("%.0f%%",100*rate)),
            position=position_dodge(0.9), vjust=-0.3, size=2) +
  scale_fill_manual(values=pal2) +
  scale_y_continuous(labels=scales::percent) +
  labs(title="07 — Binary Risk Factor Prevalence",
       x=NULL, y="Prevalence", fill="Diabetes") +
  th + theme(axis.text.x=element_text(angle=30, hjust=1))

# plot 08 — income vs diabetes (stacked)
inc_df <- train %>%
  mutate(Inc_bin=cut(Income, breaks=6)) %>%
  group_by(Inc_bin, Diabetes_binary) %>%
  summarise(n=n(), .groups="drop") %>%
  group_by(Inc_bin) %>% mutate(pct=n/sum(n))

p08 <- ggplot(inc_df, aes(Inc_bin, pct, fill=Diabetes_binary)) +
  geom_col() +
  scale_fill_manual(values=pal2) +
  scale_y_continuous(labels=scales::percent) +
  labs(title="08 — Diabetes by Income Level",
       x="Income (binned)", y="Proportion", fill="Diabetes") +
  th + theme(axis.text.x=element_text(angle=30, hjust=1))

# plot 09 — BMI vs GenHlth scatter (sample for speed)
set.seed(571)
samp <- train %>% slice_sample(n=3000)

p09 <- ggplot(samp, aes(BMI, GenHlth, color=Diabetes_binary)) +
  geom_point(alpha=0.35, size=0.9) +
  geom_smooth(method="lm", se=FALSE, linewidth=1) +
  scale_color_manual(values=pal2) +
  labs(title="09 — BMI vs GenHlth (n=3,000)",
       x="BMI (z-score)", y="GenHlth (z-score)", color="Diabetes") + th

# plot 10 — education vs diabetes rate
edu_df <- train %>%
  mutate(Edu_bin=cut(Education, breaks=5)) %>%
  group_by(Edu_bin) %>%
  summarise(rate=mean(Diabetes_binary=="Yes"), n=n(), .groups="drop")

p10 <- ggplot(edu_df, aes(Edu_bin, rate)) +
  geom_col(fill="#9B5DE5", alpha=0.85) +
  geom_text(aes(label=sprintf("%.1f%%",100*rate)), vjust=-0.4, size=2.8) +
  scale_y_continuous(labels=scales::percent) +
  labs(title="10 — Diabetes Rate by Education",
       x="Education (binned)", y="Prevalence") +
  th + theme(axis.text.x=element_text(angle=30, hjust=1))

cat("[3/6] Done.\n\n")


# ---------------------------------------------------------------
# STEP 4 — train all 9 models
# parametric: logistic, lasso
# tree-based: decision tree, random forest, GBM, XGBoost
# distance/kernel: kNN, SVM
# neural: 1-hidden-layer NN
#
# the heavier models (SVM, NN, kNN) use a 10% subsample due to
# memory and runtime constraints
# ---------------------------------------------------------------
cat("[4/6] Training 9 models ...\n")
fits <- list()

# 10% stratified subsample for SVM, NN, kNN
set.seed(571)
si          <- caret::createDataPartition(train$Diabetes_binary, p=0.10, list=FALSE)
train_small <- train[si, ]
cat(sprintf("  SVM/NN/kNN sub-sample: %d rows\n", nrow(train_small)))

# model 1 — logistic regression baseline
cat("  (1/9) Logistic regression ... ")
t0 <- Sys.time()
fits$logit <- glm(Diabetes_binary ~ ., data=train, family=binomial())
cat(sprintf("%.1fs\n", as.numeric(Sys.time()-t0, units="secs")))

# model 2 — lasso (L1-regularized logistic) with 5-fold CV for lambda
cat("  (2/9) Lasso (glmnet) ... ")
t0 <- Sys.time()
Xtr <- model.matrix(Diabetes_binary ~ . -1, data=train)
ytr <- train$Diabetes_binary
fits$lasso <- cv.glmnet(Xtr, ytr, family="binomial", alpha=1,
                        nfolds=5, type.measure="auc")
cat(sprintf("lambda=%.5f  %.1fs\n", fits$lasso$lambda.min,
            as.numeric(Sys.time()-t0, units="secs")))

# model 3 — single decision tree (interpretability baseline)
cat("  (3/9) Decision Tree (rpart) ... ")
t0 <- Sys.time()
fits$tree <- rpart::rpart(Diabetes_binary ~ ., data=train,
                          method="class", cp=0.001, maxdepth=6)
cat(sprintf("%.1fs\n", as.numeric(Sys.time()-t0, units="secs")))

# model 4 — random forest, 300 trees, mtry=4
cat("  (4/9) Random Forest (300 trees) ... ")
t0 <- Sys.time()
fits$rf <- randomForest::randomForest(Diabetes_binary ~ ., data=train,
                                      ntree=300, mtry=4, importance=TRUE)
cat(sprintf("%.1fs\n", as.numeric(Sys.time()-t0, units="secs")))

# model 5 — GBM (gradient boosting)
cat("  (5/9) GBM (500 trees) ... ")
t0 <- Sys.time()
tgbm <- train; tgbm$Diabetes_binary <- as.numeric(tgbm$Diabetes_binary=="Yes")
fits$gbm <- gbm::gbm(Diabetes_binary ~ ., data=tgbm, distribution="bernoulli",
                     n.trees=500, interaction.depth=3,
                     shrinkage=0.05, cv.folds=0, verbose=FALSE)
cat(sprintf("%.1fs\n", as.numeric(Sys.time()-t0, units="secs")))

# model 6 — XGBoost (regularized gradient boosting)
cat("  (6/9) XGBoost ... ")
t0 <- Sys.time()
xgb_train <- xgb.DMatrix(data=Xtr,
                         label=as.numeric(train$Diabetes_binary=="Yes"))
fits$xgb <- xgboost::xgb.train(
  params=list(objective="binary:logistic", eval_metric="auc",
              eta=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8),
  data=xgb_train, nrounds=400, verbose=0)
cat(sprintf("%.1fs\n", as.numeric(Sys.time()-t0, units="secs")))

# model 7 — linear SVM on subsample
cat("  (7/9) SVM (sub-sample) ... ")
t0 <- Sys.time()
fits$svm <- e1071::svm(Diabetes_binary ~ ., data=train_small,
                       kernel="linear", cost=1, probability=TRUE)
cat(sprintf("%.1fs\n", as.numeric(Sys.time()-t0, units="secs")))

# model 8 — single hidden layer neural network on subsample
cat("  (8/9) Neural network ... ")
t0 <- Sys.time()
fits$nn <- nnet::nnet(Diabetes_binary ~ ., data=train_small,
                      size=8, decay=0.01, maxit=200, trace=FALSE)
cat(sprintf("%.1fs\n", as.numeric(Sys.time()-t0, units="secs")))

# model 9 — k-Nearest Neighbors (k=21) on subsample
# kNN needs numeric matrix not dataframe with factors
cat("  (9/9) k-NN (k=21, sub-sample) ... ")
t0 <- Sys.time()
Xtr_small_num <- model.matrix(Diabetes_binary ~ . -1, data=train_small)
ytr_small     <- train_small$Diabetes_binary
fits$knn_train_X <- Xtr_small_num
fits$knn_train_y <- ytr_small
fits$knn_k       <- 21
cat(sprintf("%.1fs\n", as.numeric(Sys.time()-t0, units="secs")))

cat("[4/6] Done.\n\n")


# ---------------------------------------------------------------
# STEP 5 — evaluate all 9 models on the test set
# threshold = 0.5 for class assignment
# ---------------------------------------------------------------
cat("[5/6] Evaluating models ...\n")

truth <- test$Diabetes_binary

# metrics helper
eval_one <- function(nm, prob, truth) {
  pred <- factor(ifelse(prob>=0.5,"Yes","No"), levels=c("No","Yes"))
  cm   <- caret::confusionMatrix(pred, truth, positive="Yes")
  auc  <- as.numeric(pROC::auc(pROC::roc(truth, prob, quiet=TRUE,
                                         levels=c("No","Yes"), direction="<")))
  data.frame(Model=nm,
             Accuracy   =round(unname(cm$overall["Accuracy"]),   4),
             Sensitivity=round(unname(cm$byClass["Sensitivity"]),4),
             Specificity=round(unname(cm$byClass["Specificity"]),4),
             Precision  =round(unname(cm$byClass["Precision"]),  4),
             F1         =round(unname(cm$byClass["F1"]),         4),
             AUC        =round(auc, 4))
}

# get predicted probabilities from each model
Xte      <- model.matrix(Diabetes_binary ~ . -1, data=test)
p_logit  <- predict(fits$logit, newdata=test, type="response")
p_lasso  <- as.numeric(predict(fits$lasso, newx=Xte, s="lambda.min", type="response"))
p_tree   <- predict(fits$tree, newdata=test, type="prob")[,"Yes"]
p_rf     <- predict(fits$rf,   newdata=test, type="prob")[,"Yes"]
p_gbm    <- gbm::predict.gbm(fits$gbm, newdata=test, n.trees=500, type="response")
p_xgb    <- predict(fits$xgb, xgb.DMatrix(data=Xte))
svmp     <- predict(fits$svm,  newdata=test, probability=TRUE)
p_svm    <- attr(svmp,"probabilities")[,"Yes"]
p_nn     <- as.numeric(predict(fits$nn, newdata=test, type="raw"))

# kNN — use class::knn directly with prob output
set.seed(571)
knn_pred <- class::knn(train=fits$knn_train_X, test=Xte,
                       cl=fits$knn_train_y, k=fits$knn_k, prob=TRUE)
# convert winning-class probability into P(Yes)
knn_winner_prob <- attr(knn_pred, "prob")
p_knn <- ifelse(knn_pred=="Yes", knn_winner_prob, 1 - knn_winner_prob)

# build results table sorted by AUC
results_df <- rbind(
  eval_one("Logistic",     p_logit, truth),
  eval_one("Lasso",        p_lasso, truth),
  eval_one("DecisionTree", p_tree,  truth),
  eval_one("RandomForest", p_rf,    truth),
  eval_one("GBM",          p_gbm,   truth),
  eval_one("XGBoost",      p_xgb,   truth),
  eval_one("SVM",          p_svm,   truth),
  eval_one("NeuralNet",    p_nn,    truth),
  eval_one("kNN",          p_knn,   truth)
)
results_df <- results_df[order(-results_df$AUC), ]

# ROC objects for plotting
roc_list <- list(
  Logistic     = pROC::roc(truth, p_logit, quiet=TRUE, levels=c("No","Yes"), direction="<"),
  Lasso        = pROC::roc(truth, p_lasso, quiet=TRUE, levels=c("No","Yes"), direction="<"),
  DecisionTree = pROC::roc(truth, p_tree,  quiet=TRUE, levels=c("No","Yes"), direction="<"),
  RandomForest = pROC::roc(truth, p_rf,    quiet=TRUE, levels=c("No","Yes"), direction="<"),
  GBM          = pROC::roc(truth, p_gbm,   quiet=TRUE, levels=c("No","Yes"), direction="<"),
  XGBoost      = pROC::roc(truth, p_xgb,   quiet=TRUE, levels=c("No","Yes"), direction="<"),
  SVM          = pROC::roc(truth, p_svm,   quiet=TRUE, levels=c("No","Yes"), direction="<"),
  NeuralNet    = pROC::roc(truth, p_nn,    quiet=TRUE, levels=c("No","Yes"), direction="<"),
  kNN          = pROC::roc(truth, p_knn,   quiet=TRUE, levels=c("No","Yes"), direction="<")
)

# 9-color palette
cols9 <- c("#E63946","#2E86AB","#06A77D","#F4A261","#264653",
           "#9B5DE5","#F77F00","#43AA8B","#577590")

cat("[5/6] Done.\n\n")


# ---------------------------------------------------------------
# STEP 6 — build result plots and display all 3 windows
# ---------------------------------------------------------------
cat("[6/6] Building result plots ...\n")

# plot 11 — ROC curves for all 9 models
p11 <- base_grob(function() {
  plot(roc_list[[1]], col=cols9[1], lwd=2,
       main="11 — ROC Curves: All 9 Models")
  for (i in 2:length(roc_list)) lines(roc_list[[i]], col=cols9[i], lwd=2)
  legend("bottomright",
         legend=sprintf("%s (%.3f)", names(roc_list),
                        sapply(roc_list, function(r) as.numeric(pROC::auc(r)))),
         col=cols9, lwd=2, cex=0.65, bty="n")
})

# plot 12 — RF variable importance
p12 <- base_grob(function() {
  randomForest::varImpPlot(fits$rf, n.var=10,
                           main="12 — RF Variable Importance (top 10)")
})

# plot 13 — model comparison bar (Accuracy / F1 / AUC)
ml <- results_df %>%
  select(Model, Accuracy, F1, AUC) %>%
  pivot_longer(-Model, names_to="Metric", values_to="Value")

p13 <- ggplot(ml, aes(reorder(Model, Value), Value, fill=Metric)) +
  geom_col(position="dodge", alpha=0.85) +
  geom_text(aes(label=sprintf("%.3f",Value)),
            position=position_dodge(0.9), hjust=-0.05, size=2) +
  coord_flip() +
  scale_fill_manual(values=c("AUC"="#06A77D","Accuracy"="#2E86AB","F1"="#E63946")) +
  scale_y_continuous(limits=c(0,0.96)) +
  labs(title="13 — Model Comparison: AUC / Accuracy / F1",
       x=NULL, y="Score", fill="Metric") + th

# plot 14 — Sensitivity vs Specificity scatter
p14 <- ggplot(results_df, aes(Specificity, Sensitivity,
                              color=Model, label=Model)) +
  geom_point(size=3.5, alpha=0.9) +
  geom_text(vjust=-0.9, size=2.5, show.legend=FALSE) +
  scale_color_manual(values=setNames(cols9, results_df$Model)) +
  labs(title="14 — Sensitivity vs Specificity",
       x="Specificity", y="Sensitivity") +
  th + theme(legend.position="none")

# plot 15 — confusion matrix for the best model
best_nm    <- results_df$Model[1]
best_probs <- list(Logistic=p_logit, Lasso=p_lasso, DecisionTree=p_tree,
                   RandomForest=p_rf, GBM=p_gbm, XGBoost=p_xgb,
                   SVM=p_svm, NeuralNet=p_nn, kNN=p_knn)[[best_nm]]

bp       <- factor(ifelse(best_probs>=0.5,"Yes","No"), levels=c("No","Yes"))
cm_best  <- caret::confusionMatrix(bp, truth, positive="Yes")
cmt      <- as.data.frame(cm_best$table)
names(cmt) <- c("Predicted","Actual","Count")

p15 <- ggplot(cmt, aes(Actual, Predicted, fill=Count)) +
  geom_tile(color="white", linewidth=1) +
  geom_text(aes(label=scales::comma(Count)), size=6, fontface="bold") +
  scale_fill_gradient(low="#d9eaf7", high="#2E86AB") +
  labs(title=sprintf("15 — Confusion Matrix: %s (Best)", best_nm),
       x="Actual", y="Predicted") +
  th + theme(legend.position="none")

cat("[6/6] Done.\n\n")


# ---------------------------------------------------------------
# DISPLAY — 3 windows in RStudio plot pane
# ---------------------------------------------------------------
cat("Rendering 3 plot windows ...\n")

# window 1 — EDA Part 1
gridExtra::grid.arrange(
  p01, p02, p03, p04, p05,
  ncol = 3,
  top  = grid::textGrob(
    "EDA Part 1 — Class Balance | BMI | Age | Correlation | General Health",
    gp = grid::gpar(fontsize=12, fontface="bold"))
)

# window 2 — EDA Part 2
gridExtra::grid.arrange(
  p06, p07, p08, p09, p10,
  ncol = 3,
  top  = grid::textGrob(
    "EDA Part 2 — Boxplots | Binary Features | Income | BMI vs GenHlth | Education",
    gp = grid::gpar(fontsize=12, fontface="bold"))
)

# window 3 — Model Results
gridExtra::grid.arrange(
  p11, p12, p13, p14, p15,
  ncol = 3,
  top  = grid::textGrob(
    "Model Results — ROC | RF Importance | Comparison | Sens/Spec | Confusion Matrix",
    gp = grid::gpar(fontsize=12, fontface="bold"))
)

cat("Done. Use the  <-  ->  arrows in the Plot pane to switch windows.\n\n")


# ---------------------------------------------------------------
# print the final results table to console
# ---------------------------------------------------------------
cat("================================================================\n")
cat(" FINAL MODEL COMPARISON (sorted by AUC, all 9 models)\n")
cat("================================================================\n")
print(results_df, row.names = FALSE)
cat("\n All done.\n")
