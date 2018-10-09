library(tidyverse)
library(tm)
library(SnowballC)
library(lubridate)

# read data
train = read_csv('train.csv')
test = read_csv('test.csv')

train$date = as.Date(sapply(str_split(train$date, ','), str_c, collapse = ''), '%d%B%Y')
test$date = as.Date(sapply(str_split(test$date, ','), str_c, collapse = ''), '%d%B%Y')

# removing dates from other languages
missing = complete.cases(train)
train = train[missing, ]
test$date[which(is.na(test$date))] = median(test$date, na.rm = T)

# combine train and test
test$age = NA
combined = bind_rows(train, test)
combined$gender = factor(combined$gender)
combined$topic = factor(combined$topic)
combined$sign = factor(combined$sign)
combined$date = factor(year(combined$date))

# preparing data for model
documents = Corpus(VectorSource(combined$text))
documents = tm_map(documents, content_transformer(tolower))
documents = tm_map(documents, removeNumbers)
documents = tm_map(documents, removeWords, stopwords('english'))
documents = tm_map(documents, removePunctuation)
documents = tm_map(documents, stripWhitespace)

# for stemming use Snowball package
documents = tm_map(documents, stemDocument)

# create matrix
dtm = DocumentTermMatrix(documents, control = list(weighting = weightTfIdf))
dtm = removeSparseTerms(dtm, 0.97)
tf.idf.matrix = as.matrix(dtm)

# add word features to dataframe
word_features = as.data.frame(tf.idf.matrix)
combined = cbind(combined, word_features)
colnames(combined)[6] = 'yr'
colnames(combined)[5] = 'zodiac'

# split train and test
training = combined[1:nrow(train), -c(1, 2, 7)]
testing = combined[(nrow(train)+1):nrow(combined), -8]

# model
model = lm(age ~ ., data = training)
predictions = predict(model, newdata = testing)

# write results to csv
test$age = predictions
output = test %>% select(user.id, age) %>% group_by(user.id) %>% summarise(age = mean(age))
write.table(output, 'submission.csv', sep = ',', row.names = F)
