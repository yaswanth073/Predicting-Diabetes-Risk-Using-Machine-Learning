# CSP 571 - Machine Learning Project
# Predicting Diabetes Risk Using ML
# Team: Yaswanth Moapda, Srujith Borra, Uday Ankam
# Spring 2025
#
# Dataset: CDC BRFSS 2015 (from UCI ML Repository)
# File: diabetes_binary_5050split_health_indicators_BRFSS2015.csv
#
# How to run:
#   setwd("~/Desktop")
#   source("Diabetes_Complete.R")
#
# The plots will show up in 3 windows in the RStudio plot pane
# use the arrow buttons to go between them

rm(list = ls()); gc()
options(stringsAsFactors = FALSE, scipen = 999)
set.seed(571)

# change this if your csv is somewhere else
CSV_PATH <- "~/Desktop/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"

cat("================================================================\n")
cat(" CSP 571 — Diabetes Risk Prediction Pipeline\n")
cat(" Working dir :", getwd(), "\n")
cat("================================================================\n\n")


# ---------------------------------------------------------------
# STEP 1 - load all the packages we need
# if something isnt installed it will auto install
# ---------------------------------------------------------------
cat("[1/5] Installing / loading packages ...\n")

pkgs <- c("data.table","dplyr","ggplot2","tidyr","scales",
          "caret","glmnet","randomForest","gbm","e1071",
          "nnet","pROC","corrplot","gridExtra","grid","png")

new_pkgs <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(new_pkgs)) {
  cat("  Installing:", paste(new_pkgs, collapse = ", "), "\n")
  install.packages(new_pkgs, repos = "https://cloud.r-project.org", quiet = TRUE)
}
invisible(lapply(pkgs, function(p)
  suppressPackageStartupMessages(library(p, character.only = TRUE))))

cat("[1/5] Done.\n\n")


# ---------------------------------------------------------------
# STEP 2 - load the data and clean it up
#
# what we do here:
#   - read the csv
#   - remove duplicate rows (there were 1635 of them)
#   - balance the classes to 50/50 so the models dont get biased
#   - convert binary columns to factors
#   - standardize the continuous columns (z-score)
#   - split 70/30 into train and test
# ---------------------------------------------------------------
cat("[2/5] Loading and preprocessing ...\n")

if (!file.exists(CSV_PATH))
  stop("Cannot find CSV at: ", CSV_PATH,
       "\nSet CSV_PATH at the top of this script to the correct location.")

df <- as.data.frame(data.table::fread(CSV_PATH))
cat(sprintf("  Raw            : %d rows x %d cols\n", nrow(df), ncol(df)))

# remove exact duplicate rows
n0 <- nrow(df); df <- df[!duplicated(df), ]
cat(sprintf("  After dedup    : %d rows (%d removed)\n", nrow(df), n0 - nrow(df)))

# downsample majority class so both classes have same number of rows
# this is important otherwise models just predict the majority class
min_n <- min(table(df$Diabetes_binary))
df <- df %>% group_by(Diabetes_binary) %>%
  slice_sample(n = min_n) %>% ungroup() %>% as.data.frame()
cat(sprintf("  After balance  : %d rows (50/50)\n", nrow(df)))

# turn the target into a factor with readable labels
df$Diabetes_binary <- factor(df$Diabetes_binary, levels = c(0,1),
                             labels = c("No","Yes"))

# these are all the binary yes/no columns - convert to factor
bin_cols <- c("HighBP","HighChol","CholCheck","Smoker","Stroke",
              "HeartDiseaseorAttack","PhysActivity","Fruits","Veggies",
              "HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","DiffWalk","Sex")
for (v in bin_cols)
  if (v %in% names(df)) df[[v]] <- factor(df[[v]], levels = c(0,1))

# z-score standardize the continuous columns
# needed especially for logistic regression and SVM
cont_cols <- intersect(c("BMI","GenHlth","MentHlth","PhysHlth",
                         "Age","Education","Income"), names(df))
for (v in cont_cols) df[[v]] <- scale(df[[v]])[,1]

# stratified 70/30 split - keeps the 50/50 balance in both sets
idx   <- caret::createDataPartition(df$Diabetes_binary, p = 0.70, list = FALSE)
train <- df[ idx, ]; test <- df[-idx, ]
cat(sprintf("  Train: %d rows | Test: %d rows | Features: %d\n",
            nrow(train), nrow(test), ncol(train)-1))
cat(sprintf("  Train balance  : No=%d  Yes=%d  (%.1f%% positive)\n\n",
            sum(train$Diabetes_binary=="No"),
            sum(train$Diabetes_binary=="Yes"),
            100*mean(train$Diabetes_binary=="Yes")))


# ---------------------------------------------------------------
# STEP 3 - build the EDA plots
#
# we build all plots first and store them, then display at the end
# using grid.arrange so they show in 3 combined windows
#
# the base_grob helper is needed for corrplot and pROC plots
# because those use base R graphics, not ggplot
# we render them to a temp png then read back as a grob
# ---------------------------------------------------------------
cat("[3/5] Building EDA plots ...\n")

th   <- theme_minimal(base_size = 9)
pal2 <- c("No"="#2E86AB","Yes"="#E63946")

# helper function to convert base R plots into grobs for grid.arrange
base_grob <- function(draw_fn) {
  tmp <- tempfile(fileext = ".png")
  grDevices::png(tmp, width = 900, height = 700, res = 120)
  draw_fn()
  grDevices::dev.off()
  grid::rasterGrob(png::readPNG(tmp), interpolate = TRUE)
}

# plot 01 - just checking that our 50/50 balance worked
p01 <- ggplot(train, aes(Diabetes_binary, fill = Diabetes_binary)) +
  geom_bar() +
  geom_text(stat="count", aes(label=scales::comma(after_stat(count))),
            vjust=-0.4, size=3) +
  scale_fill_manual(values=pal2) +
  labs(title="01 — Class Balance", x="Diabetes", y="Count") +
  th + theme(legend.position="none")

# plot 02 - BMI distributions look different between the two groups
p02 <- ggplot(train, aes(BMI, fill=Diabetes_binary)) +
  geom_density(alpha=0.55) +
  scale_fill_manual(values=pal2) +
  labs(title="02 — BMI Distribution", x="BMI (z-score)",
       y="Density", fill="Diabetes") + th

# plot 03 - age is a really strong predictor, prevalence goes from 8% to 61%
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

# plot 04 - correlation heatmap
# need to convert everything to numeric first for cor()
# using base_grob because corrplot doesnt work with ggplot
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

# plot 05 - general health vs diabetes rate, almost a perfect linear trend
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

# plot 06 - boxplots of all continuous features side by side
# pivot_longer makes it easy to facet by feature
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

# plot 07 - prevalence of each binary risk factor by diabetes status
# HighBP has the biggest gap (67% vs 37%)
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

# plot 08 - income level vs diabetes, clear socioeconomic pattern
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

# plot 09 - scatter of BMI vs GenHlth, using a sample of 3000 points
# otherwise the plot is way too slow to render with 47k points
set.seed(571)
samp <- train %>% slice_sample(n=3000)

p09 <- ggplot(samp, aes(BMI, GenHlth, color=Diabetes_binary)) +
  geom_point(alpha=0.35, size=0.9) +
  geom_smooth(method="lm", se=FALSE, linewidth=1) +
  scale_color_manual(values=pal2) +
  labs(title="09 — BMI vs GenHlth (n=3,000)",
       x="BMI (z-score)", y="GenHlth (z-score)", color="Diabetes") + th

# plot 10 - education level vs diabetes rate
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

cat("[3/5] Done.\n\n")


# ---------------------------------------------------------------
# STEP 4 - train the 6 models
#
# we ran logistic, lasso, random forest and GBM on the full
# training set. SVM and neural net were too slow on 47k rows
# so we used a 10% stratified subsample for those two
# ---------------------------------------------------------------
cat("[4/5] Training 6 models ...\n")
fits <- list()

# 10% subsample for SVM and neural net
set.seed(571)
si          <- caret::createDataPartition(train$Diabetes_binary, p=0.10, list=FALSE)
train_small <- train[si, ]
cat(sprintf("  SVM/NN sub-sample: %d rows\n", nrow(train_small)))

# model 1 - logistic regression as our baseline
cat("  (1/6) Logistic regression ... ")
t0 <- Sys.time()
fits$logit <- glm(Diabetes_binary ~ ., data=train, family=binomial())
cat(sprintf("%.1fs\n", as.numeric(Sys.time()-t0, units="secs")))

# model 2 - lasso, alpha=1 means pure lasso not ridge
# cv picks the best lambda automatically using 5-fold cross validation
cat("  (2/6) Lasso (glmnet) ... ")
t0 <- Sys.time()
Xtr <- model.matrix(Diabetes_binary ~ . -1, data=train)
ytr <- train$Diabetes_binary
fits$lasso <- cv.glmnet(Xtr, ytr, family="binomial", alpha=1,
                        nfolds=5, type.measure="auc")
cat(sprintf("lambda=%.5f  %.1fs\n", fits$lasso$lambda.min,
            as.numeric(Sys.time()-t0, units="secs")))

# model 3 - random forest, 300 trees
# mtry=4 means 4 features tried at each split (roughly sqrt of 21)
cat("  (3/6) Random Forest (300 trees) ... ")
t0 <- Sys.time()
fits$rf <- randomForest::randomForest(Diabetes_binary ~ ., data=train,
                                      ntree=300, mtry=4, importance=TRUE)
cat(sprintf("%.1fs\n", as.numeric(Sys.time()-t0, units="secs")))

# model 4 - gradient boosting
# need to convert target to 0/1 numeric for gbm to work with bernoulli
cat("  (4/6) GBM (500 trees) ... ")
t0 <- Sys.time()
tgbm <- train; tgbm$Diabetes_binary <- as.numeric(tgbm$Diabetes_binary=="Yes")
fits$gbm <- gbm::gbm(Diabetes_binary ~ ., data=tgbm, distribution="bernoulli",
                     n.trees=500, interaction.depth=3,
                     shrinkage=0.05, cv.folds=0, verbose=FALSE)
cat(sprintf("%.1fs\n", as.numeric(Sys.time()-t0, units="secs")))

# model 5 - linear SVM on the small subsample
# probability=TRUE needed so we can get probability outputs for ROC
cat("  (5/6) SVM (sub-sample) ... ")
t0 <- Sys.time()
fits$svm <- e1071::svm(Diabetes_binary ~ ., data=train_small,
                       kernel="linear", cost=1, probability=TRUE)
cat(sprintf("%.1fs\n", as.numeric(Sys.time()-t0, units="secs")))

# model 6 - single hidden layer neural network, 8 units
# decay is the weight regularization parameter
cat("  (6/6) Neural network ... ")
t0 <- Sys.time()
fits$nn <- nnet::nnet(Diabetes_binary ~ ., data=train_small,
                      size=8, decay=0.01, maxit=200, trace=FALSE)
cat(sprintf("%.1fs\n", as.numeric(Sys.time()-t0, units="secs")))

cat("[4/5] Done.\n\n")


# ---------------------------------------------------------------
# STEP 5 - evaluate all models on the test set
#
# eval_one() takes a model name + predicted probabilities + true labels
# and returns a row with all 6 metrics
#
# we also build the result plots here (plots 11-15)
# ---------------------------------------------------------------
cat("[5/5] Evaluating models and building result plots ...\n")

truth <- test$Diabetes_binary

# helper to compute all metrics for one model at threshold 0.5
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

# get predicted probabilities from each model on the test set
Xte      <- model.matrix(Diabetes_binary ~ . -1, data=test)
p_logit  <- predict(fits$logit, newdata=test, type="response")
p_lasso  <- as.numeric(predict(fits$lasso, newx=Xte, s="lambda.min", type="response"))
p_rf     <- predict(fits$rf,   newdata=test, type="prob")[,"Yes"]
p_gbm    <- gbm::predict.gbm(fits$gbm, newdata=test, n.trees=500, type="response")
svmp     <- predict(fits$svm,  newdata=test, probability=TRUE)
p_svm    <- attr(svmp,"probabilities")[,"Yes"]
p_nn     <- as.numeric(predict(fits$nn, newdata=test, type="raw"))

# build results table and sort by AUC
results_df <- rbind(
  eval_one("Logistic",     p_logit, truth),
  eval_one("Lasso",        p_lasso, truth),
  eval_one("RandomForest", p_rf,    truth),
  eval_one("GBM",          p_gbm,   truth),
  eval_one("SVM",          p_svm,   truth),
  eval_one("NeuralNet",    p_nn,    truth)
)
results_df <- results_df[order(-results_df$AUC), ]

# ROC curve objects for all 6 models
roc_list <- list(
  Logistic     = pROC::roc(truth, p_logit, quiet=TRUE, levels=c("No","Yes"), direction="<"),
  Lasso        = pROC::roc(truth, p_lasso, quiet=TRUE, levels=c("No","Yes"), direction="<"),
  RandomForest = pROC::roc(truth, p_rf,    quiet=TRUE, levels=c("No","Yes"), direction="<"),
  GBM          = pROC::roc(truth, p_gbm,   quiet=TRUE, levels=c("No","Yes"), direction="<"),
  SVM          = pROC::roc(truth, p_svm,   quiet=TRUE, levels=c("No","Yes"), direction="<"),
  NeuralNet    = pROC::roc(truth, p_nn,    quiet=TRUE, levels=c("No","Yes"), direction="<")
)

cols6 <- c("#E63946","#2E86AB","#06A77D","#F4A261","#264653","#9B5DE5")

# plot 11 - ROC curves for all 6 models on the same axes
# GBM should come out on top
p11 <- base_grob(function() {
  plot(roc_list[[1]], col=cols6[1], lwd=2,
       main="11 — ROC Curves: All Models")
  for (i in 2:6) lines(roc_list[[i]], col=cols6[i], lwd=2)
  legend("bottomright",
         legend=sprintf("%s (%.3f)", names(roc_list),
                        sapply(roc_list, function(r) as.numeric(pROC::auc(r)))),
         col=cols6, lwd=2, cex=0.7, bty="n")
})

# plot 12 - variable importance from the random forest model
# this shows which features the RF found most useful
p12 <- base_grob(function() {
  randomForest::varImpPlot(fits$rf, n.var=10,
                           main="12 — RF Variable Importance (top 10)")
})

# plot 13 - grouped bar chart comparing accuracy, f1 and auc across models
ml <- results_df %>%
  select(Model, Accuracy, F1, AUC) %>%
  pivot_longer(-Model, names_to="Metric", values_to="Value")

p13 <- ggplot(ml, aes(reorder(Model, Value), Value, fill=Metric)) +
  geom_col(position="dodge", alpha=0.85) +
  geom_text(aes(label=sprintf("%.3f",Value)),
            position=position_dodge(0.9), hjust=-0.1, size=2.3) +
  coord_flip() +
  scale_fill_manual(values=c("AUC"="#06A77D","Accuracy"="#2E86AB","F1"="#E63946")) +
  scale_y_continuous(limits=c(0,0.96)) +
  labs(title="13 — Model Comparison: AUC / Accuracy / F1",
       x=NULL, y="Score", fill="Metric") + th

# plot 14 - sensitivity vs specificity scatter
# for medical screening we want high sensitivity (catch all the diabetic cases)
p14 <- ggplot(results_df, aes(Specificity, Sensitivity,
                              color=Model, label=Model)) +
  geom_point(size=4, alpha=0.9) +
  geom_text(vjust=-0.9, size=2.8, show.legend=FALSE) +
  scale_color_manual(values=setNames(cols6, results_df$Model)) +
  labs(title="14 — Sensitivity vs Specificity",
       x="Specificity", y="Sensitivity") +
  th + theme(legend.position="none")

# plot 15 - confusion matrix for whichever model had the best AUC
# showing TP, TN, FP, FN counts
best_nm    <- results_df$Model[1]
best_probs <- list(Logistic=p_logit, Lasso=p_lasso, RandomForest=p_rf,
                   GBM=p_gbm, SVM=p_svm, NeuralNet=p_nn)[[best_nm]]

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

cat("[5/5] Done.\n\n")


# ---------------------------------------------------------------
# STEP 6 - display all 15 plots in 3 windows
#
# grid.arrange puts multiple plots in one window
# window 1 = EDA part 1 (plots 1-5)
# window 2 = EDA part 2 (plots 6-10)
# window 3 = model results (plots 11-15)
# ---------------------------------------------------------------
cat("Rendering 3 plot windows ...\n")

# window 1
gridExtra::grid.arrange(
  p01, p02, p03, p04, p05,
  ncol = 3,
  top  = grid::textGrob(
    "EDA Part 1 — Class Balance | BMI | Age | Correlation | General Health",
    gp = grid::gpar(fontsize=12, fontface="bold"))
)

# window 2
gridExtra::grid.arrange(
  p06, p07, p08, p09, p10,
  ncol = 3,
  top  = grid::textGrob(
    "EDA Part 2 — Boxplots | Binary Features | Income | BMI vs GenHlth | Education",
    gp = grid::gpar(fontsize=12, fontface="bold"))
)

# window 3
gridExtra::grid.arrange(
  p11, p12, p13, p14, p15,
  ncol = 3,
  top  = grid::textGrob(
    "Model Results — ROC | RF Importance | Comparison | Sens/Spec | Confusion Matrix",
    gp = grid::gpar(fontsize=12, fontface="bold"))
)

cat("Done. Use the  <-  ->  arrows in the Plot pane to switch windows.\n\n")


# ---------------------------------------------------------------
# STEP 7 - print the final results table to the console
# ---------------------------------------------------------------
cat("================================================================\n")
cat(" FINAL MODEL COMPARISON (sorted by AUC)\n")
cat("================================================================\n")
print(results_df, row.names = FALSE)
cat("\n All done!\n")