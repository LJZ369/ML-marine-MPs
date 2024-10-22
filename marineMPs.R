# This code serves the article: Decoding the Plastic Patch: Exploring the 
# Global Distribution of Surface Microplastics in Marine regions with Interpretable 
# Machine Learning. 
# Code author: Linjie Zhang. 
# For any questions, please contact the email: lin_jiezhang@126.com.

library(tidyverse)
library(mlr3verse)
library(rnaturalearth)
library(rnaturalearthdata)
library(tidymodels)
library(egg)
library(DALEX)
library(ggpmisc)
library(sf)
library(ggsci)

world <- ne_countries(scale = "medium", returnclass = "sf")
# ML ----
## 1 Partitioning of data sets
mps_task <- mps_all[,-c(1:2)] |> mutate(conc = log10(conc+1))
# set.seed()
mps_task_split <- initial_split(mps_task, prop = 0.9, strata = 'sea')
mps_task_train <- training(mps_task_split)
mps_task_test  <-  testing(mps_task_split)

# Parallel processing
future::plan("multisession", workers = 10)

# 1 Modeling ----
## 1.1 RF ----
task_MPs_reg <- as_task_regr(mps_task_train , target = 'conc')
task_MPs_reg$set_col_roles('sea', c('stratum'))
lrn_ranger <- lrn('regr.ranger')
set.seed(111) 
# Forward selection, rmse
instance1 = fselect(
  fselector =fs("sequential"),
  task =  task_MPs_reg,
  learner = lrn_ranger,
  resampling = rsmp("cv", folds = 10),
  measure = msr('regr.rmse')
)

# 
rf_feature_p <-
  autoplot(instance1, type = "performance") +
  theme_bw() +
  theme(
    legend.position = c(0.7,0.4),
    legend.background = element_blank(),
    legend.key.size = unit(20,'pt'),
    legend.key = element_rect(fill = NA),
    axis.text.x = element_text(size = 15, colour = 'black'),
    axis.text.y = element_text(size = 15, colour = 'black'),
    panel.border = element_rect(color = "black", linewidth = 1.5, fill = NA)
  ) 

task_MPs_reg$select(instance1$result_feature_set)

ggsave("./rf_feature_p.pdf",
       set_panel_size(rf_feature_p, width=unit(4, "in"), height=unit(3, "in")), 
       width = 6, height =5, units = 'in')


# Hyperparameter optimization, 10 grid
tnr_grid_search = tnr("grid_search", resolution = 10)

search_space_RF = ps(
  mtry  = p_int(lower = 1, upper = 19), #分叉18
  num.trees = p_int(lower = 100, upper =  1900) #树的数量1450
)

set.seed(222)
system.time({
  instance2 = ti(
    task = task_MPs_reg,
    learner = lrn_ranger,
    resampling = rsmp("cv", folds = 10),
    measures = msr('regr.rmse'),
    search_space = search_space_RF,
    terminator = trm("none")
  )
  tnr_grid_search$optimize(instance2)
})

rf_tune <- as.data.table(instance2$archive)[,.(mtry,num.trees,regr.rmse)] |> 
  arrange(regr.rmse)
# 18  100

tune_p1 <-
  rf_tune |> ggplot() +
  geom_point(aes(mtry,num.trees,col = regr.rmse), shape = 19,
             size = 5)  +
  scale_color_viridis_c() + 
  theme_bw() +
  theme(
    axis.text.x = element_text(size = 10, colour = 'black'),
    axis.text.y = element_text(size = 10, colour = 'black'),
    axis.title = element_text(size = 15, colour = 'black'),
    panel.border = element_rect(color = "black", linewidth = NA, fill = NA)
  ) +
  scale_y_continuous(expand = c(0.05,0.05))+
  scale_x_continuous(expand = c(0.05,0.05))


ggsave("./tune_p1.pdf",
       set_panel_size(tune_p1, width=unit(4, "in"), height=unit(3, "in")), 
       width = 6, height =5, units = 'in')



## 1.2 SVM ----
task_MPs_reg_svm <- as_task_regr(mps_task_train , target = 'conc')
task_MPs_reg_svm$set_col_roles('sea', c('stratum'))
lrn_svm <- lrn('regr.svm')
set.seed(111)

instance3 = fselect(
  fselector = fs("sequential"),
  task =  task_MPs_reg_svm,
  learner = lrn_svm,
  resampling = rsmp("cv", folds = 10),
  measure = msr('regr.rmse')
)

svm_feature_p <-
  autoplot(instance3, type = "performance") +
  theme_bw() +
  theme(
    legend.position = c(0.7,0.4),
    legend.background = element_blank(),
    legend.key.size = unit(20,'pt'),
    legend.key = element_rect(fill = NA),
    axis.text.x = element_text(size = 15, colour = 'black'),
    axis.text.y = element_text(size = 15, colour = 'black'),
    panel.border = element_rect(color = "black", linewidth = 1.5, fill = NA)
  ) 

ggsave("./svm_feature_p.pdf",
       set_panel_size(svm_feature_p, width=unit(4, "in"), height=unit(3, "in")), 
       width = 6, height =5, units = 'in')

task_MPs_reg_svm$select(instance3$result_feature_set)


search_space_svm = ps(
  cost  = p_dbl(lower = 1, upper = 5), 
  gamma = p_dbl(lower = 0.1, upper = 1.9),
  type = p_fct('eps-regression'),
  kernel = p_fct('radial')
)

set.seed(222)

system.time({
  instance4 = ti(
    task = task_MPs_reg_svm,
    learner = lrn_svm,
    resampling = rsmp("cv", folds = 10),
    measures = msr('regr.rmse'),
    search_space =  search_space_svm,
    terminator = trm("none")
  )
  tnr_grid_search$optimize(instance4)
})

svm_tune <- as.data.table(instance4$archive)[,.(cost,gamma,regr.rmse)] |> 
  arrange(regr.rmse)


tune_p2 <- 
  svm_tune |> ggplot() +
  geom_point(aes(cost,gamma,col = regr.rmse), shape = 19,
             size = 5)  +
  scale_color_viridis_c() + 
  theme_bw() +
  theme(
    axis.text.x = element_text(size = 10, colour = 'black'),
    axis.text.y = element_text(size = 10, colour = 'black'),
    axis.title = element_text(size = 15, colour = 'black'),
    panel.border = element_rect(color = "black", linewidth = NA, fill = NA)
  ) +
  scale_y_continuous(expand = c(0.05,0.05))+
  scale_x_continuous(expand = c(0.05,0.05))

ggsave("./tune_p2.pdf",
       set_panel_size(tune_p2, width=unit(4, "in"), height=unit(3, "in")), 
       width = 6, height =5, units = 'in')


## 1.3 KNN ----

task_MPs_reg_knn <- as_task_regr(mps_task_train , target = 'conc')
task_MPs_reg_knn$set_col_roles('sea', c('stratum'))
lrn_kknn <- lrn('regr.kknn')

set.seed(111)
instance5 = fselect(
  fselector = fs("sequential"),
  task =  task_MPs_reg_knn,
  learner = lrn_kknn,
  resampling = rsmp("cv", folds = 10),
  measure = msr('regr.rmse')
)

knn_feature_p <-
  autoplot(instance5, type = "performance") +
  theme_bw() +
  theme(
    legend.position = c(0.7,0.4),
    legend.background = element_blank(),
    legend.key.size = unit(20,'pt'),
    legend.key = element_rect(fill = NA),
    axis.text.x = element_text(size = 15, colour = 'black'),
    axis.text.y = element_text(size = 15, colour = 'black'),
    panel.border = element_rect(color = "black", linewidth = 1.5, fill = NA)
  ) 

ggsave("./knn_feature_p.pdf",
       set_panel_size(knn_feature_p, width=unit(4, "in"), height=unit(3, "in")), 
       width = 6, height =5, units = 'in')


task_MPs_reg_knn$select(instance5$result_feature_set)


search_space_kknn = ps(
  k  = p_int(lower = 3, upper = 12), 
  distance = p_dbl(lower = 0.1, upper = 1)
)

set.seed(222)
system.time({
  instance6 = ti(
    task = task_MPs_reg_knn,
    learner = lrn_kknn,
    resampling = rsmp("cv", folds = 10),
    measures = msr('regr.rmse'),
    search_space = search_space_kknn,
    terminator = trm("none")
  )
  tnr_grid_search$optimize(instance6)
})

knn_tune <- as.data.table(instance6$archive)[,.(k,distance,regr.rmse)] |>  
  arrange(regr.rmse)
# 9  0.9

tune_p3 <- 
  knn_tune |> ggplot() +
  geom_point(aes(k,distance,col = regr.rmse), shape = 19,
             size = 5)  +
  scale_color_viridis_c() + 
  theme_bw() +
  theme(
    axis.text.x = element_text(size = 10, colour = 'black'),
    axis.text.y = element_text(size = 10, colour = 'black'),
    axis.title = element_text(size = 15, colour = 'black'),
    panel.border = element_rect(color = "black", linewidth = NA, fill = NA)
  ) +
  #  annotate('text',x = 1.2, y = 1.12, label = '0.6815',size = 5)+
  scale_y_continuous(expand = c(0.05,0.05))+
  scale_x_continuous(expand = c(0.05,0.05))


ggsave("./tune_p3.pdf",
       set_panel_size(tune_p3, width=unit(4, "in"), height=unit(3, "in")), 
       width = 6, height =5, units = 'in')


## 1.4 GBM ----
task_MPs_reg_gbm <- as_task_regr(mps_task_train , target = 'conc')
task_MPs_reg_gbm$set_col_roles('sea', c('stratum'))
lrn_gbm <- lrn('regr.gbm')

set.seed(111)
instance7 = fselect(
  fselector = fs("sequential"),
  task =  task_MPs_reg_gbm,
  learner = lrn_gbm,
  resampling = rsmp("cv", folds = 10),
  measure = msr('regr.rmse')
)

gbm_feature_p <-
  autoplot(instance7, type = "performance") +
  theme_bw() +
  theme(
    legend.position = c(0.7,0.4),
    legend.background = element_blank(),
    legend.key.size = unit(20,'pt'),
    legend.key = element_rect(fill = NA),
    axis.text.x = element_text(size = 15, colour = 'black'),
    axis.text.y = element_text(size = 15, colour = 'black'),
    panel.border = element_rect(color = "black", linewidth = 1.5, fill = NA)
  ) 

ggsave("./gbm_feature_p.pdf",
       set_panel_size(gbm_feature_p, width=unit(4, "in"), height=unit(3, "in")), 
       width = 6, height =5, units = 'in')

task_MPs_reg_gbm$select(instance7$result_feature_set)

search_space_gbm = ps(
  n.trees  = p_int(lower = 100, upper = 775), 
  interaction.depth = p_int(lower = 10, upper = 37)
)

set.seed(222)
system.time({
  instance8 = ti(
    task = task_MPs_reg_gbm,
    learner = lrn_gbm,
    resampling = rsmp("cv", folds = 10),
    measures = msr('regr.rmse'),
    search_space = search_space_gbm,
    terminator = trm("none")
  )
  tnr_grid_search$optimize(instance8)
})

gbm_tune <- as.data.table(instance8$archive)[,.(n.trees,interaction.depth,regr.rmse)]|>
  arrange(regr.rmse)

tune_p4 <- 
  gbm_tune |> ggplot() +
  geom_point(aes(n.trees,interaction.depth,col = regr.rmse), shape = 19,
             size = 5)  +
  scale_color_viridis_c() + 
  theme_bw() +
  theme(
    axis.text.x = element_text(size = 10, colour = 'black'),
    axis.text.y = element_text(size = 10, colour = 'black'),
    axis.title = element_text(size = 15, colour = 'black'),
    panel.border = element_rect(color = "black", linewidth = NA, fill = NA)
  ) +
  scale_y_continuous(expand = c(0.05,0.05))+
  scale_x_continuous(expand = c(0.05,0.05))

ggsave("./tune_p4.pdf",
       set_panel_size(tune_p4, width=unit(4, "in"), height=unit(3, "in")), 
       width = 6, height =5, units = 'in')



# 2 Evaluation  ----
#  RF
lrn_ranger_tune <-  lrn('regr.ranger', predict_type = "se")
lrn_ranger_tune$param_set$values <- instance2$result_learner_param_vals
task_MPs_reg_all1 <- as_task_regr(mps_task_train, target = 'conc')
task_MPs_reg_all1$select(instance1$result_feature_set)
lrn_ranger_tune$train(task_MPs_reg_all1)

# SVM
lrn_svm_tune <-  lrn('regr.svm')
lrn_svm_tune$param_set$values <- instance4$result_learner_param_vals
task_MPs_reg_all2 <- as_task_regr(mps_task_train, target = 'conc')
task_MPs_reg_all2$select(instance3$result_feature_set)
lrn_svm_tune$train(task_MPs_reg_all2)

# KNN
lrn_knn_tune <-  lrn('regr.kknn')
lrn_knn_tune$param_set$values <- instance6$result_learner_param_vals
task_MPs_reg_all3 <- as_task_regr(mps_task_train, target = 'conc')
task_MPs_reg_all3$select(instance5$result_feature_set)
lrn_knn_tune$train(task_MPs_reg_all3)

# GBM
lrn_gbm_tune <-  lrn('regr.gbm')
lrn_gbm_tune$param_set$values <- instance8$result_learner_param_vals
task_MPs_reg_all4 <- as_task_regr(mps_task_train, target = 'conc')
task_MPs_reg_all4$select(instance7$result_feature_set)
lrn_gbm_tune$train(task_MPs_reg_all4)


## 2.1 Performance evaluation----
### RF ----
prediction_rf1 = lrn_ranger_tune$predict_newdata(mps_task_train)
prediction_rf2 = lrn_ranger_tune$predict_newdata(mps_task_test)

rf_train = prediction_rf1$score(msrs(c('regr.rsq','regr.rmse')))
rf_test = prediction_rf2$score(msrs(c('regr.rsq','regr.rmse')))

r2_rmse_rf <- data.table(Response = c(prediction_rf1$response,
                                      prediction_rf2$response),
                         Truth = c(prediction_rf1$truth,
                                   prediction_rf2$truth),
                         type = c(rep('Train',length(prediction_rf1$response)),
                                  rep('Test',length(prediction_rf2$response)))) |> 
  ggplot(aes(Response,Truth)) +
  geom_point(aes(col = type),
             size = 2, alpha = 0.5) +
  geom_smooth(aes(col = type),method = 'lm') +
  geom_abline(intercept = 0, slope = 1, size = 1) +
  annotate('text', x = 0.5, y = 2.3, 
           label = paste0('Train R2 = ',round(rf_train[1],3),', ','RMSE = ',round(rf_train[2],3)))+
  annotate('text', x = 0.5, y = 2, 
           label = paste0('Test R2 = ',round(rf_test[1],3),', ','RMSE = ',round(rf_test[2],3)))+
  scale_color_manual(values = c('#fe0606','#19add9'))+
  scale_shape_manual(values = c(15,16)) +
  theme_bw() +
  theme(
    axis.text = element_text(size=16,colour = 'black'),
    axis.title = element_text(size =20),
    axis.ticks.length.x = unit(0.2,"cm"),
    axis.ticks.x = element_line(colour = "black"),
    axis.ticks.length.y = unit(0.2,"cm"),
    axis.ticks.y = element_line(colour = "black"),
    legend.background = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 1.5, fill = NA),
    #legend.position = 'none'
  ) 

ggsave("./r2_rmse_rf.pdf",
       set_panel_size(r2_rmse_rf, width=unit(4, "in"), height=unit(4, "in")), 
       width = 9, height =6, units = 'in')


### SVM ----
prediction_svm1 = lrn_svm_tune$predict_newdata(mps_task_train)
prediction_svm2 = lrn_svm_tune$predict_newdata(mps_task_test)

svm_train = prediction_svm1$score(msrs(c('regr.rsq','regr.rmse')))
svm_test = prediction_svm2$score(msrs(c('regr.rsq','regr.rmse')))

r2_rmse_svm <- data.table(Response = c(prediction_svm1$response,
                                       prediction_svm2$response),
                          Truth = c(prediction_svm1$truth,
                                    prediction_svm2$truth),
                          type = c(rep('Train',length(prediction_svm1$response)),
                                   rep('Test',length(prediction_svm2$response)))) |> 
  ggplot(aes(Response,Truth)) +
  geom_point(aes(col = type),
             size = 2, alpha = 0.5) +
  geom_smooth(aes(col = type),method = 'lm') +
  geom_abline(intercept = 0, slope = 1, size = 1) +
  annotate('text', x = 0.2, y = 2.3, 
           label = paste0('Train R2 = ',round(svm_train[1],3),', ','RMSE = ',round(svm_train[2],3)))+
  annotate('text', x = 0.2, y = 2, 
           label = paste0('Test R2 = ',round(svm_test[1],3),', ','RMSE = ',round(svm_test[2],3)))+
  scale_color_manual(values = c('#fe0606','#19add9'))+
  scale_shape_manual(values = c(15,16)) +
  theme_bw() +
  theme(
    axis.text = element_text(size=16,colour = 'black'),
    axis.title = element_text(size =20),
    axis.ticks.length.x = unit(0.2,"cm"),
    axis.ticks.x = element_line(colour = "black"),
    axis.ticks.length.y = unit(0.2,"cm"),
    axis.ticks.y = element_line(colour = "black"),
    legend.background = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 1.5, fill = NA)
    #legend.position = 'none'
  ) 

ggsave("./r2_rmse_svm.pdf",
       set_panel_size(r2_rmse_svm, width=unit(4, "in"), height=unit(4, "in")), 
       width = 9, height =6, units = 'in')


### KNN ----
prediction_knn1 = lrn_knn_tune$predict_newdata(mps_task_train)
prediction_knn2 = lrn_knn_tune$predict_newdata(mps_task_test)

knn_train = prediction_knn1$score(msrs(c('regr.rsq','regr.rmse')))
knn_test = prediction_knn2$score(msrs(c('regr.rsq','regr.rmse')))

r2_rmse_knn <- data.table(Response = c(prediction_knn1$response,
                                       prediction_knn2$response),
                          Truth = c(prediction_knn1$truth,
                                    prediction_knn2$truth),
                          type = c(rep('Train',length(prediction_knn1$response)),
                                   rep('Test',length(prediction_knn2$response)))) |> 
  ggplot(aes(Response,Truth)) +
  geom_point(aes(col = type),
             size = 2, alpha = 0.5) +
  geom_smooth(aes(col = type),method = 'lm') +
  geom_abline(intercept = 0, slope = 1, size = 1) +
  annotate('text', x = 0.35, y = 2.3, 
           label = paste0('Train R2 = ',round(knn_train[1],3),', ','RMSE = ',round(knn_train[2],3)))+
  annotate('text', x = 0.35, y = 2, 
           label = paste0('Test R2 = ',round(knn_test[1],3),', ','RMSE = ',round(knn_test[2],3)))+
  scale_color_manual(values = c('#fe0606','#19add9'))+
  scale_shape_manual(values = c(15,16)) +
  theme_bw() +
  theme(
    axis.text = element_text(size=16,colour = 'black'),
    axis.title = element_text(size =20),
    axis.ticks.length.x = unit(0.2,"cm"),
    axis.ticks.x = element_line(colour = "black"),
    axis.ticks.length.y = unit(0.2,"cm"),
    axis.ticks.y = element_line(colour = "black"),
    legend.background = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 1.5, fill = NA)
    #legend.position = 'none'
  ) 


ggsave("./r2_rmse_knn.pdf",
       set_panel_size(r2_rmse_knn, width=unit(4, "in"), height=unit(4, "in")), 
       width = 9, height =6, units = 'in')


### GBM ----
prediction_gbm1 = lrn_gbm_tune$predict_newdata(mps_task_train)
prediction_gbm2 = lrn_gbm_tune$predict_newdata(mps_task_test)

gbm_train = prediction_gbm1$score(msrs(c('regr.rsq','regr.rmse')))
gbm_test = prediction_gbm2$score(msrs(c('regr.rsq','regr.rmse')))

r2_rmse_gbm <- data.table(Response = c(prediction_gbm1$response,
                                       prediction_gbm2$response),
                          Truth = c(prediction_gbm1$truth,
                                    prediction_gbm2$truth),
                          type = c(rep('Train',length(prediction_gbm1$response)),
                                   rep('Test',length(prediction_gbm2$response)))) |> 
  ggplot(aes(Response,Truth)) +
  geom_point(aes(col = type),
             size = 2, alpha = 0.5) +
  geom_smooth(aes(col = type),method = 'lm') +
  geom_abline(intercept = 0, slope = 1, size = 1) +
  annotate('text', x = 0.3, y = 2.3, 
           label = paste0('Train R2 = ',round(gbm_train[1],3),', ','RMSE = ',round(gbm_train[2],3)))+
  annotate('text', x = 0.3, y = 2, 
           label = paste0('Test R2 = ',round(gbm_test[1],3),', ','RMSE = ',round(gbm_test[2],3)))+
  scale_color_manual(values = c('#fe0606','#19add9'))+
  scale_shape_manual(values = c(15,16)) +
  theme_bw() +
  theme(
    axis.text = element_text(size=16,colour = 'black'),
    axis.title = element_text(size =20),
    axis.ticks.length.x = unit(0.2,"cm"),
    axis.ticks.x = element_line(colour = "black"),
    axis.ticks.length.y = unit(0.2,"cm"),
    axis.ticks.y = element_line(colour = "black"),
    legend.background = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 1.5, fill = NA)
    #legend.position = 'none'
  ) +
  scale_y_continuous(breaks = c(0, 1, 2))


ggsave("./r2_rmse_gbm.pdf",
       set_panel_size(r2_rmse_gbm, width=unit(4, "in"), height=unit(4, "in")), 
       width = 9, height =6, units = 'in')



# 3 Model interpretation----
## 3.1 Feature importance----
### 3.1.1 Global----
# 19 feature
Influencing_factor_type <- data.frame(variable = instance1$result_feature_set,
                                      type = c('Hum','Bioc','Bioc','Hum','Bioc',
                                               'Hum','Bioc','Hum','Bioc','Phys',
                                               'Phys','Atmo','Bioc','Atmo','Atmo',
                                               'Phys','Atmo','Phys','Atmo'))

explain_rf <- DALEXtra::explain_mlr3(model = lrn_ranger_tune,  
                                     data = mps_task_train[,instance1$result_feature_set], 
                                     y = mps_task_train$conc,
                                     label = "Random Forest",
                                     colorize = FALSE)


rf_effect = model_parts(explain_rf)

rf_importance_data <- plot(rf_effect, show_boxplots = FALSE)$data |> 
  left_join(Influencing_factor_type, by = 'variable')

rf_contribution_p1 <-
  rf_importance_data |> 
  group_by(type) |> 
  summarise(RMSE = mean(dropout_loss.x)) |> 
  ggplot(aes(x = reorder(type,RMSE), y = RMSE, fill = type)) +
  geom_bar(stat = 'identity',width = 0.7) +
  scale_fill_manual(values = c('#FF9800','#00C853','#FF0037','#2962FF')) +
  coord_flip() + 
  scale_y_continuous(expand = expansion(mult = c(0,0.05))) +
  labs( y = 'RMSE', x= '') +
  theme_minimal() +
  theme(
    axis.text = element_text(size = 16,colour = 'black'),
    axis.title.x = element_text(size = 20,colour = 'black'),
    axis.ticks.length.x = unit(0.2,"cm"),
    axis.ticks.length.y = unit(0,"cm"),
    panel.grid.major.y = element_blank(),  
    panel.grid.minor.y = element_blank(),
    panel.grid.major.x = element_line(colour = "grey", size = 0.5), 
    panel.grid.minor.x = element_blank(),  
    legend.position = 'none'
  )

ggsave("./rf_contribution_p1.pdf",
       set_panel_size(rf_contribution_p1, width=unit(4, "in"), height=unit(4, "in")), 
       width = 6, height =5, units = 'in')


rf_importance_p1 <-  rf_importance_data |> 
  arrange(dropout_loss.x) |> 
  mutate(variable = fct_inorder(variable)) |> 
  ggplot() +
  geom_bar(aes(variable,dropout_loss.x, fill = type),stat = 'identity',width = 0.6) +
  scale_fill_manual(values = c('#FF9800','#00C853','#FF0037','#2962FF')) +
  coord_flip() + 
  scale_y_continuous(expand = expansion(mult = c(0,0.1))) +
  labs( y = 'RMSE') +
  theme_bw() +
  theme(
    axis.text = element_text(size=16,colour = 'black'),
    axis.title = element_text(size =20),
    axis.ticks.length.x = unit(0.2,"cm"),
    axis.ticks.x = element_line(colour = "black"),
    axis.ticks.length.y = unit(0.2,"cm"),
    axis.ticks.y = element_line(colour = "black"),
    legend.background = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 1.5, fill = NA)
    #legend.position = 'none'
  )

ggsave("./文章图片D/rf_importance_p1.pdf",
       set_panel_size(rf_importance_p1, width=unit(4, "in"), height=unit(3, "in")), 
       width = 6, height =5, units = 'in')


### 3.1.2 Each region ----
# [1] "South Pacific Ocean"                      "Arctic Ocean"                            
# [3] "Southern Ocean"                           "South Atlantic Ocean"                    
# [5] "Mediterranean Region"                     "Baltic Sea"                              
# [7] "Indian Ocean"                             "South China and Easter Archipelagic Seas"
# [9] "North Atlantic Ocean"                     "North Pacific Ocean" 
#### South Pacific Ocean ----

explain_SPO <- DALEXtra::explain_mlr3(model = lrn_ranger_tune,  
                                      data = mps_task_train[mps_task_train$sea == "South Pacific Ocean",
                                                            instance1$result_feature_set], 
                                      y = mps_task_train$conc[mps_task_train$sea == "South Pacific Ocean"],
                                      label = "Random Forest",
                                      colorize = FALSE)


SPO_effect = model_parts(explain_SPO)

SPO_importance_data <- plot(SPO_effect, show_boxplots = FALSE)$data |> 
  left_join(Influencing_factor_type, by = 'variable')

SPO_contribution_p1 <- SPO_importance_data |> 
  group_by(type) |> 
  summarise(RMSE = mean(dropout_loss.x)) |> 
  ggplot(aes(x = reorder(type,RMSE), y = RMSE, fill = type)) +
  geom_bar(stat = 'identity',width = 0.7) +
  #geom_text(aes(y = 0.02,label = round(RMSE, 3)), hjust = 0, color = "black", size = 6) + 
  scale_fill_manual(values = c('#FF9800','#00C853','#FF0037','#2962FF')) +
  coord_flip() + 
  scale_y_continuous(expand = expansion(mult = c(0,0.05))) +
  labs( y = 'RMSE', x= '') +
  theme_minimal() +
  theme(
    axis.text = element_text(size = 16,colour = 'black'),
    axis.title.x = element_text(size = 20,colour = 'black'),
    axis.ticks.length.x = unit(0.2,"cm"),
    axis.ticks.length.y = unit(0,"cm"),
    panel.grid.major.y = element_blank(), 
    panel.grid.minor.y = element_blank(),
    panel.grid.major.x = element_line(colour = "grey", size = 0.5),  
    panel.grid.minor.x = element_blank(),  
    legend.position = 'none'
  )


SPO_importance_p1 <-  SPO_importance_data |> 
  arrange(dropout_loss.x) |> 
  mutate(variable = fct_inorder(variable)) |> 
  ggplot() +
  geom_bar(aes(variable,dropout_loss.x, fill = type),stat = 'identity',width = 0.6) +
  scale_fill_manual(values = c('#FF9800','#00C853','#FF0037','#2962FF')) +
  coord_flip() + 
  scale_y_continuous(expand = expansion(mult = c(0,0.1))) +
  labs( y = 'RMSE') +
  theme_bw() +
  theme(
    axis.text = element_text(size=16,colour = 'black'),
    axis.title = element_text(size =20),
    axis.ticks.length.x = unit(0.2,"cm"),
    axis.ticks.x = element_line(colour = "black"),
    axis.ticks.length.y = unit(0.2,"cm"),
    axis.ticks.y = element_line(colour = "black"),
    legend.background = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 1.5, fill = NA),
    legend.position = 'none'
  )

#### Arctic Ocean  ----

explain_AO <- DALEXtra::explain_mlr3(model = lrn_ranger_tune,  
                                     data = mps_task_train[mps_task_train$sea == "Arctic Ocean",
                                                           instance1$result_feature_set], 
                                     y = mps_task_train$conc[mps_task_train$sea == "Arctic Ocean"],
                                     label = "Random Forest",
                                     colorize = FALSE)


AO_effect = model_parts(explain_AO)

AO_importance_data <- plot(AO_effect, show_boxplots = FALSE)$data |> 
  left_join(Influencing_factor_type, by = 'variable')

AO_contribution_p1 <- AO_importance_data |> 
  group_by(type) |> 
  summarise(RMSE = mean(dropout_loss.x)) |> 
  ggplot(aes(x = reorder(type,RMSE), y = RMSE, fill = type)) +
  geom_bar(stat = 'identity',width = 0.7) +
  #geom_text(aes(y = 0.02,label = round(RMSE, 3)), hjust = 0, color = "black", size = 6) + 
  scale_fill_manual(values = c('#FF9800','#00C853','#FF0037','#2962FF')) +
  coord_flip() + 
  scale_y_continuous(expand = expansion(mult = c(0,0.05))) +
  labs( y = 'RMSE', x= '') +
  theme_minimal() +
  theme(
    axis.text = element_text(size = 16,colour = 'black'),
    axis.title.x = element_text(size = 20,colour = 'black'),
    axis.ticks.length.x = unit(0.2,"cm"),
    axis.ticks.length.y = unit(0,"cm"),
    panel.grid.major.y = element_blank(),  
    panel.grid.minor.y = element_blank(),
    panel.grid.major.x = element_line(colour = "grey", size = 0.5),  
    panel.grid.minor.x = element_blank(),  
    legend.position = 'none'
  )

AO_importance_p1 <-  AO_importance_data |> 
  arrange(dropout_loss.x) |> 
  mutate(variable = fct_inorder(variable)) |> 
  ggplot() +
  geom_bar(aes(variable,dropout_loss.x, fill = type),stat = 'identity',width = 0.6) +
  scale_fill_manual(values = c('#FF9800','#00C853','#FF0037','#2962FF')) +
  coord_flip() + 
  scale_y_continuous(expand = expansion(mult = c(0,0.1))) +
  labs( y = 'RMSE') +
  theme_bw() +
  theme(
    axis.text = element_text(size=16,colour = 'black'),
    axis.title = element_text(size =20),
    axis.ticks.length.x = unit(0.2,"cm"),
    axis.ticks.x = element_line(colour = "black"),
    axis.ticks.length.y = unit(0.2,"cm"),
    axis.ticks.y = element_line(colour = "black"),
    legend.background = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 1.5, fill = NA),
    legend.position = 'none'
  )

#### Southern Ocean  ----

explain_SO <- DALEXtra::explain_mlr3(model = lrn_ranger_tune,  
                                     data = mps_task_train[mps_task_train$sea == "Southern Ocean",
                                                           instance1$result_feature_set], 
                                     y = mps_task_train$conc[mps_task_train$sea == "Southern Ocean"],
                                     label = "Random Forest",
                                     colorize = FALSE)


SO_effect = model_parts(explain_SO)

SO_importance_data <- plot(SO_effect, show_boxplots = FALSE)$data |> 
  left_join(Influencing_factor_type, by = 'variable')

SO_contribution_p1 <- SO_importance_data |> 
  group_by(type) |> 
  summarise(RMSE = mean(dropout_loss.x)) |> 
  ggplot(aes(x = reorder(type,RMSE), y = RMSE, fill = type)) +
  geom_bar(stat = 'identity',width = 0.7) +
  #geom_text(aes(y = 0.02,label = round(RMSE, 3)), hjust = 0, color = "black", size = 6) + 
  scale_fill_manual(values = c('#FF9800','#00C853','#FF0037','#2962FF')) +
  coord_flip() + 
  scale_y_continuous(expand = expansion(mult = c(0,0.05))) +
  labs( y = 'RMSE', x= '') +
  theme_minimal() +
  theme(
    axis.text = element_text(size = 16,colour = 'black'),
    axis.title.x = element_text(size = 20,colour = 'black'),
    axis.ticks.length.x = unit(0.2,"cm"),
    axis.ticks.length.y = unit(0,"cm"),
    panel.grid.major.y = element_blank(),  # 去掉纵轴的格栅
    panel.grid.minor.y = element_blank(),
    panel.grid.major.x = element_line(colour = "grey", size = 0.5),  # 加粗横轴的格栅
    panel.grid.minor.x = element_blank(),  # 去掉横轴的次要格栅
    legend.position = 'none'
  )

SO_importance_p1 <-  SO_importance_data |> 
  arrange(dropout_loss.x) |> 
  mutate(variable = fct_inorder(variable)) |> 
  ggplot() +
  geom_bar(aes(variable,dropout_loss.x, fill = type),stat = 'identity',width = 0.6) +
  scale_fill_manual(values = c('#FF9800','#00C853','#FF0037','#2962FF')) +
  coord_flip() + 
  scale_y_continuous(expand = expansion(mult = c(0,0.1))) +
  labs( y = 'RMSE') +
  theme_bw() +
  theme(
    axis.text = element_text(size=16,colour = 'black'),
    axis.title = element_text(size =20),
    axis.ticks.length.x = unit(0.2,"cm"),
    axis.ticks.x = element_line(colour = "black"),
    axis.ticks.length.y = unit(0.2,"cm"),
    axis.ticks.y = element_line(colour = "black"),
    legend.background = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 1.5, fill = NA),
    legend.position = 'none'
  )

#### South Atlantic Ocean  ----

explain_SAO <- DALEXtra::explain_mlr3(model = lrn_ranger_tune,  
                                      data = mps_task_train[mps_task_train$sea == "South Atlantic Ocean",
                                                            instance1$result_feature_set], 
                                      y = mps_task_train$conc[mps_task_train$sea == "South Atlantic Ocean"],
                                      label = "Random Forest",
                                      colorize = FALSE)


SAO_effect = model_parts(explain_SAO)

SAO_importance_data <- plot(SAO_effect, show_boxplots = FALSE)$data |> 
  left_join(Influencing_factor_type, by = 'variable')

SAO_contribution_p1 <- SAO_importance_data |> 
  group_by(type) |> 
  summarise(RMSE = mean(dropout_loss.x)) |> 
  ggplot(aes(x = reorder(type,RMSE), y = RMSE, fill = type)) +
  geom_bar(stat = 'identity',width = 0.7) +
  #geom_text(aes(y = 0.02,label = round(RMSE, 3)), hjust = 0, color = "black", size = 6) + 
  scale_fill_manual(values = c('#FF9800','#00C853','#FF0037','#2962FF')) +
  coord_flip() + 
  scale_y_continuous(expand = expansion(mult = c(0,0.05))) +
  labs( y = 'RMSE', x= '') +
  theme_minimal() +
  theme(
    axis.text = element_text(size = 16,colour = 'black'),
    axis.title.x = element_text(size = 20,colour = 'black'),
    axis.ticks.length.x = unit(0.2,"cm"),
    axis.ticks.length.y = unit(0,"cm"),
    panel.grid.major.y = element_blank(),  # 去掉纵轴的格栅
    panel.grid.minor.y = element_blank(),
    panel.grid.major.x = element_line(colour = "grey", size = 0.5),  # 加粗横轴的格栅
    panel.grid.minor.x = element_blank(),  # 去掉横轴的次要格栅
    legend.position = 'none'
  )

SAO_importance_p1 <-  SAO_importance_data |> 
  arrange(dropout_loss.x) |> 
  mutate(variable = fct_inorder(variable)) |> 
  ggplot() +
  geom_bar(aes(variable,dropout_loss.x, fill = type),stat = 'identity',width = 0.6) +
  scale_fill_manual(values = c('#FF9800','#00C853','#FF0037','#2962FF')) +
  coord_flip() + 
  scale_y_continuous(expand = expansion(mult = c(0,0.1))) +
  labs( y = 'RMSE') +
  theme_bw() +
  theme(
    axis.text = element_text(size=16,colour = 'black'),
    axis.title = element_text(size =20),
    axis.ticks.length.x = unit(0.2,"cm"),
    axis.ticks.x = element_line(colour = "black"),
    axis.ticks.length.y = unit(0.2,"cm"),
    axis.ticks.y = element_line(colour = "black"),
    legend.background = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 1.5, fill = NA),
    legend.position = 'none'
  )

#### Mediterranean Region  ----

explain_MR <- DALEXtra::explain_mlr3(model = lrn_ranger_tune,  
                                     data = mps_task_train[mps_task_train$sea == "Mediterranean Region",
                                                           instance1$result_feature_set], 
                                     y = mps_task_train$conc[mps_task_train$sea == "Mediterranean Region"],
                                     label = "Random Forest",
                                     colorize = FALSE)


MR_effect = model_parts(explain_MR)

MR_importance_data <- plot(MR_effect, show_boxplots = FALSE)$data |> 
  left_join(Influencing_factor_type, by = 'variable')

MR_contribution_p1 <- MR_importance_data |> 
  group_by(type) |> 
  summarise(RMSE = mean(dropout_loss.x)) |> 
  ggplot(aes(x = reorder(type,RMSE), y = RMSE, fill = type)) +
  geom_bar(stat = 'identity',width = 0.7) +
  #geom_text(aes(y = 0.02,label = round(RMSE, 3)), hjust = 0, color = "black", size = 6) + 
  scale_fill_manual(values = c('#FF9800','#00C853','#FF0037','#2962FF')) +
  coord_flip() + 
  scale_y_continuous(expand = expansion(mult = c(0,0.05))) +
  labs( y = 'RMSE', x= '') +
  theme_minimal() +
  theme(
    axis.text = element_text(size = 16,colour = 'black'),
    axis.title.x = element_text(size = 20,colour = 'black'),
    axis.ticks.length.x = unit(0.2,"cm"),
    axis.ticks.length.y = unit(0,"cm"),
    panel.grid.major.y = element_blank(),  # 去掉纵轴的格栅
    panel.grid.minor.y = element_blank(),
    panel.grid.major.x = element_line(colour = "grey", size = 0.5),  # 加粗横轴的格栅
    panel.grid.minor.x = element_blank(),  # 去掉横轴的次要格栅
    legend.position = 'none'
  )

MR_importance_p1 <-  MR_importance_data |> 
  arrange(dropout_loss.x) |> 
  mutate(variable = fct_inorder(variable)) |> 
  ggplot() +
  geom_bar(aes(variable,dropout_loss.x, fill = type),stat = 'identity',width = 0.6) +
  scale_fill_manual(values = c('#FF9800','#00C853','#FF0037','#2962FF')) +
  coord_flip() + 
  scale_y_continuous(expand = expansion(mult = c(0,0.1))) +
  labs( y = 'RMSE') +
  theme_bw() +
  theme(
    axis.text = element_text(size=16,colour = 'black'),
    axis.title = element_text(size =20),
    axis.ticks.length.x = unit(0.2,"cm"),
    axis.ticks.x = element_line(colour = "black"),
    axis.ticks.length.y = unit(0.2,"cm"),
    axis.ticks.y = element_line(colour = "black"),
    legend.background = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 1.5, fill = NA),
    legend.position = 'none'
  )

#### Baltic Sea  ----

explain_BS <- DALEXtra::explain_mlr3(model = lrn_ranger_tune,  
                                     data = mps_task_train[mps_task_train$sea == "Baltic Sea",
                                                           instance1$result_feature_set], 
                                     y = mps_task_train$conc[mps_task_train$sea == "Baltic Sea"],
                                     label = "Random Forest",
                                     colorize = FALSE)


BS_effect = model_parts(explain_BS)

BS_importance_data <- plot(BS_effect, show_boxplots = FALSE)$data |> 
  left_join(Influencing_factor_type, by = 'variable')

BS_contribution_p1 <- BS_importance_data |> 
  group_by(type) |> 
  summarise(RMSE = mean(dropout_loss.x)) |> 
  ggplot(aes(x = reorder(type,RMSE), y = RMSE, fill = type)) +
  geom_bar(stat = 'identity',width = 0.7) +
  #geom_text(aes(y = 0.02,label = round(RMSE, 3)), hjust = 0, color = "black", size = 6) + 
  scale_fill_manual(values = c('#FF9800','#00C853','#FF0037','#2962FF')) +
  coord_flip() + 
  scale_y_continuous(expand = expansion(mult = c(0,0.05))) +
  labs( y = 'RMSE', x= '') +
  theme_minimal() +
  theme(
    axis.text = element_text(size = 16,colour = 'black'),
    axis.title.x = element_text(size = 20,colour = 'black'),
    axis.ticks.length.x = unit(0.2,"cm"),
    axis.ticks.length.y = unit(0,"cm"),
    panel.grid.major.y = element_blank(),  # 去掉纵轴的格栅
    panel.grid.minor.y = element_blank(),
    panel.grid.major.x = element_line(colour = "grey", size = 0.5),  # 加粗横轴的格栅
    panel.grid.minor.x = element_blank(),  # 去掉横轴的次要格栅
    legend.position = 'none'
  )


BS_importance_p1 <-  BS_importance_data |> 
  arrange(dropout_loss.x) |> 
  mutate(variable = fct_inorder(variable)) |> 
  ggplot() +
  geom_bar(aes(variable,dropout_loss.x, fill = type),stat = 'identity',width = 0.6) +
  scale_fill_manual(values = c('#FF9800','#00C853','#FF0037','#2962FF')) +
  coord_flip() + 
  scale_y_continuous(expand = expansion(mult = c(0,0.1))) +
  labs( y = 'RMSE') +
  theme_bw() +
  theme(
    axis.text = element_text(size=16,colour = 'black'),
    axis.title = element_text(size =20),
    axis.ticks.length.x = unit(0.2,"cm"),
    axis.ticks.x = element_line(colour = "black"),
    axis.ticks.length.y = unit(0.2,"cm"),
    axis.ticks.y = element_line(colour = "black"),
    legend.background = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 1.5, fill = NA),
    legend.position = 'none'
  )


#### Indian Ocean  ----

explain_IO <- DALEXtra::explain_mlr3(model = lrn_ranger_tune,  
                                     data = mps_task_train[mps_task_train$sea == "Indian Ocean",
                                                           instance1$result_feature_set], 
                                     y = mps_task_train$conc[mps_task_train$sea == "Indian Ocean"],
                                     label = "Random Forest",
                                     colorize = FALSE)


IO_effect = model_parts(explain_IO)

IO_importance_data <- plot(IO_effect, show_boxplots = FALSE)$data |> 
  left_join(Influencing_factor_type, by = 'variable')

IO_contribution_p1 <- IO_importance_data |> 
  group_by(type) |> 
  summarise(RMSE = mean(dropout_loss.x)) |> 
  ggplot(aes(x = reorder(type,RMSE), y = RMSE, fill = type)) +
  geom_bar(stat = 'identity',width = 0.7) +
  #geom_text(aes(y = 0.02,label = round(RMSE, 3)), hjust = 0, color = "black", size = 6) + 
  scale_fill_manual(values = c('#FF9800','#00C853','#FF0037','#2962FF')) +
  coord_flip() + 
  scale_y_continuous(expand = expansion(mult = c(0,0.05))) +
  labs( y = 'RMSE', x= '') +
  theme_minimal() +
  theme(
    axis.text = element_text(size = 16,colour = 'black'),
    axis.title.x = element_text(size = 20,colour = 'black'),
    axis.ticks.length.x = unit(0.2,"cm"),
    axis.ticks.length.y = unit(0,"cm"),
    panel.grid.major.y = element_blank(),  # 去掉纵轴的格栅
    panel.grid.minor.y = element_blank(),
    panel.grid.major.x = element_line(colour = "grey", size = 0.5),  # 加粗横轴的格栅
    panel.grid.minor.x = element_blank(),  # 去掉横轴的次要格栅
    legend.position = 'none'
  )


IO_importance_p1 <-  IO_importance_data |> 
  arrange(dropout_loss.x) |> 
  mutate(variable = fct_inorder(variable)) |> 
  ggplot() +
  geom_bar(aes(variable,dropout_loss.x, fill = type),stat = 'identity',width = 0.6) +
  scale_fill_manual(values = c('#FF9800','#00C853','#FF0037','#2962FF')) +
  coord_flip() + 
  scale_y_continuous(expand = expansion(mult = c(0,0.1))) +
  labs( y = 'RMSE') +
  theme_bw() +
  theme(
    axis.text = element_text(size=16,colour = 'black'),
    axis.title = element_text(size =20),
    axis.ticks.length.x = unit(0.2,"cm"),
    axis.ticks.x = element_line(colour = "black"),
    axis.ticks.length.y = unit(0.2,"cm"),
    axis.ticks.y = element_line(colour = "black"),
    legend.background = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 1.5, fill = NA),
    legend.position = 'none'
  )

#### South China and Easter Archipelagic Seas  ----

explain_SCEAS <- DALEXtra::explain_mlr3(model = lrn_ranger_tune,  
                                        data = mps_task_train[mps_task_train$sea == "South China and Easter Archipelagic Seas",
                                                              instance1$result_feature_set], 
                                        y = mps_task_train$conc[mps_task_train$sea == "South China and Easter Archipelagic Seas"],
                                        label = "Random Forest",
                                        colorize = FALSE)


SCEAS_effect = model_parts(explain_SCEAS)

SCEAS_importance_data <- plot(SCEAS_effect, show_boxplots = FALSE)$data |> 
  left_join(Influencing_factor_type, by = 'variable')

SCEAS_contribution_p1 <- SCEAS_importance_data |> 
  group_by(type) |> 
  summarise(RMSE = mean(dropout_loss.x)) |> 
  ggplot(aes(x = reorder(type,RMSE), y = RMSE, fill = type)) +
  geom_bar(stat = 'identity',width = 0.7) +
  #geom_text(aes(y = 0.02,label = round(RMSE, 3)), hjust = 0, color = "black", size = 6) + 
  scale_fill_manual(values = c('#FF9800','#00C853','#FF0037','#2962FF')) +
  coord_flip() + 
  scale_y_continuous(expand = expansion(mult = c(0,0.05))) +
  labs( y = 'RMSE', x= '') +
  theme_minimal() +
  theme(
    axis.text = element_text(size = 16,colour = 'black'),
    axis.title.x = element_text(size = 20,colour = 'black'),
    axis.ticks.length.x = unit(0.2,"cm"),
    axis.ticks.length.y = unit(0,"cm"),
    panel.grid.major.y = element_blank(),  # 去掉纵轴的格栅
    panel.grid.minor.y = element_blank(),
    panel.grid.major.x = element_line(colour = "grey", size = 0.5),  # 加粗横轴的格栅
    panel.grid.minor.x = element_blank(),  # 去掉横轴的次要格栅
    legend.position = 'none'
  )

SCEAS_importance_p1 <-  SCEAS_importance_data |> 
  arrange(dropout_loss.x) |> 
  mutate(variable = fct_inorder(variable)) |> 
  ggplot() +
  geom_bar(aes(variable,dropout_loss.x, fill = type),stat = 'identity',width = 0.6) +
  scale_fill_manual(values = c('#FF9800','#00C853','#FF0037','#2962FF')) +
  coord_flip() + 
  scale_y_continuous(expand = expansion(mult = c(0,0.1))) +
  labs( y = 'RMSE') +
  theme_bw() +
  theme(
    axis.text = element_text(size=16,colour = 'black'),
    axis.title = element_text(size =20),
    axis.ticks.length.x = unit(0.2,"cm"),
    axis.ticks.x = element_line(colour = "black"),
    axis.ticks.length.y = unit(0.2,"cm"),
    axis.ticks.y = element_line(colour = "black"),
    legend.background = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 1.5, fill = NA),
    legend.position = 'none'
  )

#### North Atlantic Ocean  ----

explain_NAO <- DALEXtra::explain_mlr3(model = lrn_ranger_tune,  
                                      data = mps_task_train[mps_task_train$sea == "North Atlantic Ocean",
                                                            instance1$result_feature_set], 
                                      y = mps_task_train$conc[mps_task_train$sea == "North Atlantic Ocean"],
                                      label = "Random Forest",
                                      colorize = FALSE)


NAO_effect = model_parts(explain_NAO)

NAO_importance_data <- plot(NAO_effect, show_boxplots = FALSE)$data |> 
  left_join(Influencing_factor_type, by = 'variable')

NAO_contribution_p1 <- NAO_importance_data |> 
  group_by(type) |> 
  summarise(RMSE = mean(dropout_loss.x)) |> 
  ggplot(aes(x = reorder(type,RMSE), y = RMSE, fill = type)) +
  geom_bar(stat = 'identity',width = 0.7) +
  #geom_text(aes(y = 0.02,label = round(RMSE, 3)), hjust = 0, color = "black", size = 6) + 
  scale_fill_manual(values = c('#FF9800','#00C853','#FF0037','#2962FF')) +
  coord_flip() + 
  scale_y_continuous(expand = expansion(mult = c(0,0.05))) +
  labs( y = 'RMSE', x= '') +
  theme_minimal() +
  theme(
    axis.text = element_text(size = 16,colour = 'black'),
    axis.title.x = element_text(size = 20,colour = 'black'),
    axis.ticks.length.x = unit(0.2,"cm"),
    axis.ticks.length.y = unit(0,"cm"),
    panel.grid.major.y = element_blank(),  # 去掉纵轴的格栅
    panel.grid.minor.y = element_blank(),
    panel.grid.major.x = element_line(colour = "grey", size = 0.5),  # 加粗横轴的格栅
    panel.grid.minor.x = element_blank(),  # 去掉横轴的次要格栅
    legend.position = 'none'
  )

NAO_importance_p1 <-  NAO_importance_data |> 
  arrange(dropout_loss.x) |> 
  mutate(variable = fct_inorder(variable)) |> 
  ggplot() +
  geom_bar(aes(variable,dropout_loss.x, fill = type),stat = 'identity',width = 0.6) +
  scale_fill_manual(values = c('#FF9800','#00C853','#FF0037','#2962FF')) +
  coord_flip() + 
  scale_y_continuous(expand = expansion(mult = c(0,0.1))) +
  labs( y = 'RMSE') +
  theme_bw() +
  theme(
    axis.text = element_text(size=16,colour = 'black'),
    axis.title = element_text(size =20),
    axis.ticks.length.x = unit(0.2,"cm"),
    axis.ticks.x = element_line(colour = "black"),
    axis.ticks.length.y = unit(0.2,"cm"),
    axis.ticks.y = element_line(colour = "black"),
    legend.background = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 1.5, fill = NA),
    legend.position = 'none'
  )

#### North Pacific Ocean  ----

explain_NPO <- DALEXtra::explain_mlr3(model = lrn_ranger_tune,  
                                      data = mps_task_train[mps_task_train$sea == "North Pacific Ocean",
                                                            instance1$result_feature_set], 
                                      y = mps_task_train$conc[mps_task_train$sea == "North Pacific Ocean"],
                                      label = "Random Forest",
                                      colorize = FALSE)


NPO_effect = model_parts(explain_NPO)

NPO_importance_data <- plot(NPO_effect, show_boxplots = FALSE)$data |> 
  left_join(Influencing_factor_type, by = 'variable')

NPO_contribution_p1 <- NPO_importance_data |> 
  group_by(type) |> 
  summarise(RMSE = mean(dropout_loss.x)) |> 
  ggplot(aes(x = reorder(type,RMSE), y = RMSE, fill = type)) +
  geom_bar(stat = 'identity',width = 0.7) +
  #geom_text(aes(y = 0.02,label = round(RMSE, 3)), hjust = 0, color = "black", size = 6) + 
  scale_fill_manual(values = c('#FF9800','#00C853','#FF0037','#2962FF')) +
  coord_flip() + 
  scale_y_continuous(expand = expansion(mult = c(0,0.05))) +
  labs( y = 'RMSE', x= '') +
  theme_minimal() +
  theme(
    axis.text = element_text(size = 16,colour = 'black'),
    axis.title.x = element_text(size = 20,colour = 'black'),
    axis.ticks.length.x = unit(0.2,"cm"),
    axis.ticks.length.y = unit(0,"cm"),
    panel.grid.major.y = element_blank(),  # 去掉纵轴的格栅
    panel.grid.minor.y = element_blank(),
    panel.grid.major.x = element_line(colour = "grey", size = 0.5),  # 加粗横轴的格栅
    panel.grid.minor.x = element_blank(),  # 去掉横轴的次要格栅
    legend.position = 'none'
  )

NPO_importance_p1 <-  NPO_importance_data |> 
  arrange(dropout_loss.x) |> 
  mutate(variable = fct_inorder(variable)) |> 
  ggplot() +
  geom_bar(aes(variable,dropout_loss.x, fill = type),stat = 'identity',width = 0.6) +
  scale_fill_manual(values = c('#FF9800','#00C853','#FF0037','#2962FF')) +
  coord_flip() + 
  scale_y_continuous(expand = expansion(mult = c(0,0.1))) +
  labs( y = 'RMSE') +
  theme_bw() +
  theme(
    axis.text = element_text(size=16,colour = 'black'),
    axis.title = element_text(size =20),
    axis.ticks.length.x = unit(0.2,"cm"),
    axis.ticks.x = element_line(colour = "black"),
    axis.ticks.length.y = unit(0.2,"cm"),
    axis.ticks.y = element_line(colour = "black"),
    legend.background = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 1.5, fill = NA),
    legend.position = 'none'
  )

## 3.2 Global interpretation----
### 3.2.1 pdp+ice----
rf_profiles <- model_profile(explain_rf)

rf_profiles_plot <-
  plot(rf_profiles, geom = 'profiles') + 
  labs(title = '',subtitle = '') + 
  theme_bw() +
  theme(strip.background = element_blank(),
        strip.placement = 'outside',
        axis.text.x = element_text(size = 10, colour = 'black'),
        axis.text.y = element_text(size = 10, colour = 'black'),
        axis.title.y = element_text(size = 15, colour = 'black'),
        legend.position = 'none')


### 4.2.2 ALE----
rf_ALE <- model_profile(explainer = explain_rf,
                        type       = "accumulated",
                        variables  = instance1$result_feature_set)
rf_ALE_plot <- 
  plot(rf_ALE)+ 
  labs(title = '',subtitle = '') + 
  theme_bw() +
  theme(strip.background = element_blank(),
        strip.placement = 'outside',
        axis.text.x = element_text(size = 10, colour = 'black'),
        axis.text.y = element_text(size = 10, colour = 'black'),
        axis.title.y = element_text(size = 15, colour = 'black'),
        legend.position = 'none')


## 3.3 Choose 4 features to explain----
### 3.3.1 Global----
pd_rf <- model_profile(explainer = explain_rf,
                       type = "partial", N = NULL,
                       variables = c("ph", "shipping",'so','strd'))

al_rf <- model_profile(explainer = explain_rf,
                       type = "accumulated", N = NULL,
                       variables = c("ph", "shipping",'so','strd'))

pd_rf$agr_profiles$`_label_` = "PDP"
al_rf$agr_profiles$`_label_` = "ALE"

PDP_ALE_2_p <-
  plot(pd_rf, al_rf) + 
  labs(title = '',subtitle = '') + 
  theme_bw() +
  theme(strip.background = element_blank(),
        strip.placement = 'outside',
        strip.text = element_text(size = 20),
        axis.text = element_text(size = 15, colour = 'black'),
        axis.title.y = element_text(size = 15, colour = 'black'),
        panel.border = element_rect(color = "black", linewidth = 1.5, fill = NA))

### 4.3.2 Each region----
explain_rf_group <- DALEXtra::explain_mlr3(model = lrn_ranger_tune,  
                                           data = mps_task_train, 
                                           y = mps_task_train$conc,
                                           label = "Random Forest",
                                           colorize = FALSE)

pd_rf_group <- model_profile(explainer = explain_rf_group,
                             type = "partial", groups = 'sea', N = NULL,
                             variables = c("ph", "shipping",'so','strd'))

al_rf_group <- model_profile(explainer = explain_rf_group,
                             type = "accumulated", groups = 'sea', N = NULL,
                             variables = c("ph", "shipping",'so','strd'))


PDP_2_p <- 
  plot(pd_rf_group)+ 
  scale_color_npg() +
  labs(title = '',subtitle = '') + 
  theme_bw() +
  theme(strip.background = element_blank(),
        strip.placement = 'outside',
        strip.text = element_text(size = 20),
        axis.text = element_text(size = 15, colour = 'black'),
        axis.title.y = element_text(size = 15, colour = 'black'),
        panel.border = element_rect(color = "black", linewidth = 1.5, fill = NA))

ALE_2_p <- 
  plot(al_rf_group)+ 
  scale_color_npg() +
  labs(title = '',subtitle = '') + 
  theme_bw() +
  theme(strip.background = element_blank(),
        strip.placement = 'outside',
        strip.text = element_text(size = 20),
        axis.text = element_text(size = 15, colour = 'black'),
        axis.title.y = element_text(size = 15, colour = 'black'),
        panel.border = element_rect(color = "black", linewidth = 1.5, fill = NA))


binwidth_n <- (max(mps_task_train$strd)-min(mps_task_train$strd))/500

mps_task_train |> 
  ggplot(aes(x = strd)) +
  geom_histogram(aes(y = ..density..), binwidth = binwidth_n, fill = 'black', alpha = 0.7) +
  geom_density(color = 'red', size = 0.5) +
  labs(title = 'Data Distribution', x = 'Value', y = 'Density') +
  theme_minimal()


# 4 Model prediction----
## 4.1 MPs concentrations ----
prediction_all = lrn_ranger_tune$predict_newdata(data.table(predict_sea))

predict_sea <- cbind(predict_sea, prediction_all$response, prediction_all$se) 
colnames(predict_sea)[c(45,46)] <- c('response','se')
predict_sea <- predict_sea |> mutate(conc = 10^response-1)


# prediction  log
mps_plot_all_log <- 
  ggplot() + 
  geom_sf(data = world, fill ='lightgray', col = 'lightgray') +
  geom_raster(data = predict_sea, 
              aes(x = lon, y =lat, fill = response)) + 
  scale_fill_gradientn(colors = c("#00BFFF", '#F4A460', '#FF4500',
                                  '#FF0000'),
                       values = rescale(c(0, 
                                          log10(8), 
                                          log10(15),
                                          log10(22), 
                                          log10(max(predict_sea$conc)+1))))+
  theme_bw() +
  theme(
    axis.text = element_text(size=16,colour = 'black'),
    axis.title = element_text(size =20),
    axis.ticks.length.x = unit(0.2,"cm"),
    axis.ticks.x = element_line(colour = "black"),
    axis.ticks.length.y = unit(0.2,"cm"),
    axis.ticks.y = element_line(colour = "black"),
    legend.background = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 1.5, fill = NA)
    #legend.position = 'none'
  ) +
  guides(color = guide_legend(override.aes = list(size = 5)))+
  scale_x_continuous(expand = expansion(mult = c(0,0))) +
  scale_y_continuous(expand = expansion(mult = c(0,0))) +
  labs(fill = 'log10(MPs + 1)')


# prediction， se
mps_plot_all_se <- 
  ggplot() + 
  geom_sf(data = world, fill ='lightgray', col = 'lightgray') +
  geom_raster(data = predict_sea, 
              aes(x = lon, y =lat, fill = se)) + 
  scale_fill_gradientn(colors = c("#0061FF", '#FFFC00', '#FF0000'),
                       values = rescale(c(0, 
                                          0.15,
                                          max(predict_sea$se))))+
  theme_bw() +
  theme(
    axis.text = element_text(size=16,colour = 'black'),
    axis.title = element_text(size =20),
    axis.ticks.length.x = unit(0.2,"cm"),
    axis.ticks.x = element_line(colour = "black"),
    axis.ticks.length.y = unit(0.2,"cm"),
    axis.ticks.y = element_line(colour = "black"),
    legend.background = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 1.5, fill = NA)
    #legend.position = 'none'
  ) +
  guides(color = guide_legend(override.aes = list(size = 5)))+
  scale_x_continuous(expand = expansion(mult = c(0,0))) +
  scale_y_continuous(expand = expansion(mult = c(0,0))) +
  labs(fill = 'se')


# WDPA
poly1 <- st_read(dsn = "./marine_point/p1/WDPA_WDOECM_Jun2023_Public_marine_shp-polygons.shp")
poly2 <- st_read(dsn = "./marine_point/p2/WDPA_WDOECM_Jun2023_Public_marine_shp-polygons.shp")
poly3 <- st_read(dsn = "./marine_point/p3/WDPA_WDOECM_Jun2023_Public_marine_shp-polygons.shp")
poly <- rbind(poly1,poly2,poly3)
rm(poly1,poly2,poly3)


mps_plot_all2 <- 
  ggplot() + 
  geom_sf(data = world, fill ='lightgray', col = 'lightgray') +
  geom_raster(data = predict_sea |> 
                mutate(conc_7 = case_when(
                  conc <= 1 ~ 1,
                  conc <= 2 ~ 2,
                  conc <= 3 ~ 3,
                  conc <= 6 ~ 4,
                  conc <= 9 ~ 5,
                  conc <= 12 ~ 6,
                  conc > 12 ~ 7
                ),conc_7 = factor(conc_7,labels = c('0–1',
                                                    '1–2',
                                                    '2–3',
                                                    '3–6',
                                                    '6–9',
                                                    '9–12',
                                                    '> 12'))), 
              aes(x = lon, y =lat, fill = conc_7)) + 
  scale_fill_manual(values = c('#0000FF','#0080FF','#00FFFF','#80FF80',
                               '#FFFF00','#FF8000','#FF0000'))+
  geom_sf(data = poly, fill = NA, col = 'black', size = 0.2) +
  theme_bw() +
  theme(
    axis.text = element_text(size=16,colour = 'black'),
    axis.title = element_text(size =20),
    axis.ticks.length.x = unit(0.2,"cm"),
    axis.ticks.x = element_line(colour = "black"),
    axis.ticks.length.y = unit(0.2,"cm"),
    axis.ticks.y = element_line(colour = "black"),
    legend.background = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 1.5, fill = NA)
    #legend.position = 'none'
  ) +
  guides(color = guide_legend(override.aes = list(size = 5)))+
  scale_x_continuous(expand = expansion(mult = c(0,0))) +
  scale_y_continuous(expand = expansion(mult = c(0,0))) +
  labs(fill = 'MPs(particle/m3)')



## 4.2 MP stock----
# 6371 km2, grid  
area_calculation <- function(lat,partition){
  lat_low = abs(lat) - partition/2
  area = 2*pi*6371^2*(sin((lat_low + partition)*pi/180) - sin(lat_low*pi/180))/(360/partition)
  return(area)
}
# 5 m
predict_sea <- predict_sea |> 
  mutate(area = map2_dbl(lat,1,~area_calculation(.x,.y)))|> 
  mutate(stock = conc*area*10^6*5)

predict_sea_stock <- predict_sea |> group_by(sea) |> 
  summarise(area_sum = sum(area),
            stock_sum = sum(stock))|> 
  arrange(desc(stock_sum)) 

predict_sea_stock_plot <- 
  predict_sea_stock |> mutate(sea = factor(sea, levels = unique(predict_sea_stock$sea))) |> 
  ggplot() +
  geom_bar(aes(sea, stock_sum/1e14),stat = "identity", 
           fill = '#21908C', width = 0.5)+
  geom_text(aes(sea, y = (stock_sum/1e14 + 0.3), label = round(stock_sum/1e14,3)))+
  annotate('text', x = 6, y = 6, label = '42.792') +
  theme_bw() +
  theme(
    axis.text = element_text(size=16,colour = 'black'),
    axis.title = element_text(size =20),
    axis.ticks.length.x = unit(0,"cm"),
    axis.ticks.x = element_line(colour = "black"),
    axis.ticks.length.y = unit(0.2,"cm"),
    axis.ticks.y = element_line(colour = "black"),
    legend.background = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 1.5, fill = NA),
    legend.position = c(0.8,0.8)
  ) +
  scale_x_discrete(expand = expansion(mult = c(0.1,0.1))) 