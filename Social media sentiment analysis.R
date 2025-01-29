# Install and Load Packages
install.packages(c("tidyverse", "tm", "syuzhet", "wordcloud", "e1071", "caret"))
library(tidyverse); library(tm); library(syuzhet); library(wordcloud); library(e1071); library(caret)

# Load Data
df <- read.csv(choose.files(), stringsAsFactors = FALSE)

df$label <- as.factor(df$label)

# Text Cleaning Function
clean_text <- function(text) {
  text <- tolower(text) %>%
    removePunctuation() %>%
    removeNumbers() %>%
    removeWords(stopwords("en")) %>%
    stripWhitespace()
  return(text)
}

df$text <- sapply(df$text, clean_text)
df$text
# Sentiment Analysis
df$sentiment <- get_sentiment(df$text, method = "syuzhet")
df$sentiment_label <- ifelse(df$sentiment > 0, "Positive", ifelse(df$sentiment < 0, "Negative", "Neutral"))

# Word Cloud
corpus <- Corpus(VectorSource(df$text))
wordcloud(corpus, max.words = 100, colors = brewer.pal(8, "Dark2"))

# Convert to Document-Term Matrix
dtm <- DocumentTermMatrix(corpus)
dtm_df <- as.data.frame(as.matrix(dtm))
dtm_df$label <- df$label  

# Train-Test Split
set.seed(123)
trainIndex <- createDataPartition(dtm_df$label, p = 0.8, list = FALSE)
train_data <- dtm_df[trainIndex, ]
test_data <- dtm_df[-trainIndex, ]

# Train Naive Bayes Model
nb_model <- naiveBayes(label ~ ., data = train_data)
nb_model
# Predict & Evaluate
predictions <- predict(nb_model, test_data)
conf_matrix <- confusionMatrix(predictions, test_data$label)
print(conf_matrix)

# Save Model & Predictions
saveRDS(nb_model, "sentiment_model.rds")
write.csv(data.frame(Actual = test_data$label, Predicted = predictions), "sentiment_predictions.csv", row.names = FALSE)
