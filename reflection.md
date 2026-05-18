# Reflection IAI_Project_1

## Winning model
Following a subset of the training runs, find more in `results/results.csv` (sourced from MLFlow via `src/report.ipynb`):

| Model                    | Training time   |   Accuracy |   Precision |   Recall |   Macro F1 |
|:-------------------------|:----------------|-----------:|------------:|---------:|-----------:|
| Distilbert Fine Tuning   | 24.77 min       |    0.93252 |    0.929394 |  0.93616 |   0.932765 |
| TF_IDF Model [RAW]       | 1.48 min        |    0.88292 |    0.884057 |  0.88144 |   0.882746 |

When looking at both models, they both do a reasonably good job in sentiment analysis. They were both trained on the same dataset consisting of 25k recrods for training and 25k records for testing. With 16x more time, a gain of 5% was seen. Depending on where on the scale between 0% and 100% this 5% jump would be, it might be worth it to invest the additional resources. The jump from 88% to 93% is real, but it does not bring the fine-tuned model into a stage where it could be used in production. 
**Based on this, I would prefer the lower training cost and see the TF_IDF + LogReg model as the winner.**

Based on the given metrics, I would classify both of the models as overfit. When looking at the epoch-over-epoch metrics of the fine-tuning run, the validation loss increases from 0.21 to 0.34 after 5 epochs. The training loss, on the other hand, drops from 0.22 to 0.03 in the same 5 epochs. This indicates a model that is fit too well to the data. Suggested steps could be to decrease the learning rate or select the best-performing epoch for the final model. 
For the TF_IDF model, the training F1 is about 5% higher than the validation F1 (0.932 vs 0.883). In a perfectly fitted model, this value would be 0. As I cannot find any specific threshold, I would conclude this model as being slightly overfit. Suggested steps could be to limit the number of features in the vectorizer. 

## Training and testing data
The dataset for both models was sourced from Hugging Face. The dataset already included a split of data into two sets, one for training and one for testing. Based on the [source](https://ai.stanford.edu/~amaas/data/sentiment/) of the dataset, the data in each set is balanced regarding the count of positive and negative elements. If not checked, it might be an issue that the first 50% of the datasets are all one tag and then the second half the second tag. From my understanding, both TF_IDF and fine-tuning were not affected by this, as they either train on all the data at once (TF_IDF) or use an inbuilt sampler (fine-tuning). 

## Docker
The current state of the repo contains two stages of a machine learning model. The training and the usage of said model. Using Docker, it is possible to ship only one of these two parts - with the dependencies and code required for this stage. The amount of dependencies and especially the size of said dependencies is much larger for the model training compared to serving the model. With containerizing the model deployment and a dedicated `requirements.txt` it is possible to ship without any unused dependencies. 

## The suprise
Initially I planned to clean up the data before feeding it to the vectorizer for training. For this I created two versions of the same data set and trained a model on both. I did not expect a big jump compared to the raw data, however I did not expect the difference to be 0 - as it was. As part of the preprocessing, filler words and punctuation were removed, and the words were lemmatized. 

## The broken 
I ended up investing a lot of time into tracking the model, artifacts and metrics - which is amazing, as I have many artifacts ready now, and I also know how to do this in the future. On the other hand, I should have focused more on the actual task at hand and invested some of this time into actually working with the hyperparameters or other settings that would influence the actual models. 


