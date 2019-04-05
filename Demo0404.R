rm(list=ls())
library(tidyverse)
library(glmnet)
library(pROC)
library(text2vec)
library(slam)
library(xgboost)
library(e1071)
library(caret)
#feature engineering
all = read.table("/Users/chuan/Desktop/WorkingOn/Demo0404/Project2_data.tsv",
                 stringsAsFactors = F,header = T)
all$review = gsub('<.*?>', ' ', all$review)
splits = read.table("/Users/chuan/Desktop/WorkingOn/Demo0404/Project2_splits.csv",
                    header = T)
s = 3 # Here we use the 3rd training/test split. 
train = all[-which(all$new_id%in%splits[,s]),]
test = all[which(all$new_id%in%splits[,s]),]
prep_fun = tolower
tok_fun = word_tokenizer
it_train = itoken(train$review,
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun)
it_test = itoken(test$review,
                 preprocessor = prep_fun, 
                 tokenizer = tok_fun)
stop_words = c("i", "me", "my", "myself", 
               "we", "our", "ours", "ourselves", 
               "you", "your", "yours", 
               "their", "they", "his", "her", 
               "she", "he", "a", "an", "and",
               "is", "was", "are", "were", 
               "him", "himself", "has", "have", 
               "it", "its", "of", "one", "for", 
               "the", "us", "this")
vocab = create_vocabulary(it_train,ngram = c(1L,4L))
pruned_vocab = prune_vocabulary(vocab,
                                term_count_min = 5, 
                                doc_proportion_max = 0.5,
                                doc_proportion_min = 0.001)
bigram_vectorizer = vocab_vectorizer(pruned_vocab)

dtm_train = create_dtm(it_train, bigram_vectorizer)
dtm_test = create_dtm(it_test, bigram_vectorizer)
v.size = dim(dtm_train)[2]
ytrain = train$sentiment
summ = matrix(0, nrow=v.size, ncol=4)
summ[,1] = colapply_simple_triplet_matrix(
  as.simple_triplet_matrix(dtm_train[ytrain==1, ]), mean)
summ[,2] = colapply_simple_triplet_matrix(
  as.simple_triplet_matrix(dtm_train[ytrain==1, ]), var)
summ[,3] = colapply_simple_triplet_matrix(
  as.simple_triplet_matrix(dtm_train[ytrain==0, ]), mean)
summ[,4] = colapply_simple_triplet_matrix(
  as.simple_triplet_matrix(dtm_train[ytrain==0, ]), var)
n1=sum(ytrain); 
n=length(ytrain)
n0= n - n1
myp = (summ[,1] - summ[,3])/
  sqrt(summ[,2]/n1 + summ[,4]/n0)
words = colnames(dtm_train) 
id = order(abs(myp), decreasing=TRUE)[1:2000] 
pos.list = words[id[myp[id]>0]] 
neg.list = words[id[myp[id]<0]]
write(words[id], file="/Users/chuan/Desktop/WorkingOn/Demo0404/myvocab.txt")
####
myvocab = scan(file = "/Users/chuan/Desktop/WorkingOn/Demo0404/myvocab.txt", 
               what = character())
all = read.table("/Users/chuan/Desktop/WorkingOn/Demo0404/Project2_data.tsv",
                 stringsAsFactors = F,header = T)
all$review = gsub('<.*?>', ' ', all$review)
splits = read.table("/Users/chuan/Desktop/WorkingOn/Demo0404/Project2_splits.csv",
                    header = T)
	
s = 3
train = all[-which(all$new_id%in%splits[,s]),]
test = all[which(all$new_id%in%splits[,s]),]
it_train = itoken(train$review,
                  preprocessor = tolower, 
                  tokenizer = word_tokenizer)
it_test = itoken(test$review,
                 preprocessor = tolower, 
                tokenizer = word_tokenizer)
vocab = create_vocabulary(it_train,ngram = c(1L,4L))
vocab = vocab[vocab$term %in% myvocab, ]
bigram_vectorizer = vocab_vectorizer(vocab)
dtm_train = create_dtm(it_train, bigram_vectorizer)
dtm_test = create_dtm(it_test, bigram_vectorizer)
#xgboost
# s = 3 # try the 3rd training/test split. 
train.y = train$sentiment
test.y = test$sentiment
param = list(max_depth = 2, 
             subsample = 0.5, 
             objective='binary:logistic')
ntrees = 500
set.seed(500)
bst = xgb.train(params = param, 
                data = xgb.DMatrix(data = dtm_train, label = train.y),
                nrounds = ntrees, 
                nthread = 2)

dt = xgb.model.dt.tree(model = bst)
words = unique(dt$Feature[dt$Feature != "Leaf"])
words
length(words)
new_feature_train = xgb.create.features(model = bst, dtm_train)
new_feature_train = new_feature_train[, - c(1:ncol(dtm_train))]
new_feature_test = xgb.create.features(model = bst, dtm_test)
new_feature_test = new_feature_test[, - c(1:ncol(dtm_test))]
c(ncol(new_feature_test), ncol(new_feature_train))
#logistic regression with lasso penalty
#find the best lambda using cross-validation
set.seed(123) 
cv.lasso <- cv.glmnet(new_feature_train, train.y, alpha = 1, family = "binomial")
plot(cv.lasso)
#fit the final model on the training data
lasso <- glmnet(new_feature_train, train.y, alpha = 1, family = "binomial",
                lambda = cv.lasso$lambda.min)
#test AUC
pred <- predict(lasso, new_feature_test, type="class")
pROC::roc(test.y,as.numeric(pred))
#logistic regression with ridge penalty
set.seed(123) 
cv.ridge <- cv.glmnet(new_feature_train, train.y, alpha = 0, family = "binomial")
plot(cv.ridge)
#fit the final model on the training data
ridge <- glmnet(new_feature_train, train.y, alpha = 0, family = "binomial",
                lambda = cv.lasso$lambda.min)
#test AUC
pred <- predict(ridge, new_feature_test, type="class")
pROC::roc(test.y,as.numeric(pred))
#naive bayes
#function to convert the word frequencies to yes (presence) and no (absence) labels
convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}
#apply the convert_count function to get final training and testing DTMs
trainNB <- apply(new_feature_train, 2, convert_count)
testNB <- apply(new_feature_test, 2, convert_count)
Bayes <- naiveBayes(trainNB, as.factor(train.y))
#test AUC
pred <- predict(Bayes,testNB)
pROC::roc(test.y,as.numeric(pred))
