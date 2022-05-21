# State Social Operator Detector

## Overview
State-funded social media operators are a hard-to-detect but significant threat to any democracy with free speech, and that threat is growing. In recent years, the extent of these state-funded campaigns has become clear. Russian campaigns undertaken to influence [elections](https://www.brennancenter.org/our-work/analysis-opinion/new-evidence-shows-how-russias-election-interference-has-gotten-more) are most prominent in the news, but other campaigns have been identified, with the intent to [turn South American countries against the US](https://www.nbcnews.com/news/latino/russia-disinformation-ukraine-spreading-spanish-speaking-media-rcna22843), spread disinformation on the [invasion of Ukraine](https://www.forbes.com/sites/petersuciu/2022/03/10/russian-sock-puppets-spreading-misinformation-on-social-media-about-ukraine/), and foment conflict in America's own culture wars by [influencing all sides](https://journals.sagepub.com/doi/10.1177/19401612221082052) as part of an effort to weaken America's hegemonic status.

Iranian and [Chinese](https://www.bbc.com/news/56364952) efforts are also well-funded, though not as widespread or aggressive as those of Russia. Even so, Chinese influence is growing, and often it uses social media to spread specific narratives on [Xinjiang and the Uyghur situation](https://www.lawfareblog.com/understanding-pro-china-propaganda-and-disinformation-tool-set-xinjiang), Hong Kong, COVID-19, and Taiwan as well as sometimes supporting [Russian efforts](https://www.brookings.edu/techstream/china-and-russia-are-joining-forces-to-spread-disinformation/).

We need better tools to combat this disinformation, both for social media administrators as well as the public. As part of an effort towards that, we have created a proof-of-concept tool that can be operated via browser extension to identify likely state-funded social media operators on Twitter through inference performed on tweet content.

The core of the tool is a DistilBERT language transformer model that has been finetuned on 250K samples of known state operator tweets and natural tweets pulled from the Twitter API. It is highly accurate at distinguishing normal users from state operators (99%), but has some limitations due to sampling recency bias. We intend to iteratively improve the model as time goes on. Currently, a demo of the model is hosted at HuggingFace: [state-op-detector](https://huggingface.co/lingwave-admin/state-op-detector)  

This repo contains all the code used to train the model, as well as links to our cleaned dataset, so that you can replicate the process yourself if desired. We also include Chrome and Firefox extensions that can be used to test the model as you browse Twitter. 

## Table of Contents
- Environment & Setup
  - Dependencies
  - Training Hardware
- Repo Structure
  - Data Folder
  - Notebooks Folder
  - Finetuned Folder
- Usage
  - Retraining
  - Inference
- Data
- Training
- Results
- Limitations
- Contributors

## Environment & Setup
### Dependencies
Training was done using `pytorch 1.8.2` and `transfomers 4.18.0`.

Twitter sampling was done with `tweepy 4.8.0`.

Other dependencies are included in the `environment` file.

### Training Hardware
The model was trained on a system with an RTX 3090, and took just under two hours to train with the dataset we've provided. 

## Repo Structure
### 1. Data Folder
- The data files we used are too large for Github, so we provide a Google Drive download link [here](https://drive.google.com/drive/folders/1jkmWl7xXvsiVppXcGf_l1S3FGszHWoUl?usp=sharing) instead. These files should be placed in this folder.

### 2. Notebooks Folder
- `1-Tweet-Data-Prep`: Data extraction for raw CSVs containing state operator tweets, data cleaning, prep, and sequence concatenation.
- `2-Tweet-Named-Entity-Exploration`: Named Entity Recognition for state operator tweets, in order to enhance sampling of normal user tweets (so that the model is exposed to normal user tweets on the same topics frequented by state operators).
- `3-Natural-Tweet-Acquisition`: Normal Twitter user tweet sampling, based on entities identified in previous notebook.
- `4-Dataset-Combination-Split`: Concatenation and sampling of state operator and natural tweets, and randomized separation into train and test sets.
- `5-Transformer-Finetuning`: Finetuning of DistilBERT for tweet classification using the HuggingFace `transformers` library.

### 3. Finetuned Folder
- This folder should contain the trained model. As the files are too large for Github, they can be downloaded [here](https://drive.google.com/drive/folders/1HQrmKupywv7xGigF2Kop_X4PsdBj8aoW?usp=sharing).

## Usage
### Retraining
If you wish to try training the model, make sure the data is in the appropriate folders and then run the notebook `5-Transformer-Finetuning`. Everything needed for training is in this notebook. If you wish to retrain with new or different data, you can choose to modify either the state operator tweet data, the natural tweet data, or both. You can modify this through the appropriate notebook, and then assemble the dataset in `4-Dataset-Combination-Split`.

### Inference
If you wish to try out inference, you can use the model hosted [here](https://huggingface.co/lingwave-admin/state-op-detector). This will always be the latest model. You can also download the model from HuggingFace and run it locally:

```python
from transformers import pipeline

classifier = pipeline(model="lingwave-admin/state-op-detector")

classifier("Washington Gives Ankara An Ultimatum | Trump Makes BigTime Overture To Iran | Authoritarian Spirits Congress The Espionage Act And Punishing WikiLeaks | Analysis Of The European Parliamentary Elections | One Mans Quest To Expose A Fake BBC Video About Syria | China Holds Three Trump Cards In War Against US | Maldives Affirms Fealty To Diego Garcia | The End Of Theresa May | Within The Church People Can Become Truly Free | China Hails Modi Victory This Is Why | ")
```

You can also provide `classifier` a list of strings, each of which contains a tweet sequence like the above.

However, if you wish to retrain the model and test inference locally on your own machine (or a hosted platform), you can use the following code from the `notebooks` folder:

```python
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import TextClassificationPipeline

ft_model = "../finetuned/troll_detect_distilbert"
model = AutoModelForSequenceClassification.from_pretrained(ft_model)
tokenizer = AutoTokenizer.from_pretrained(ft_model)

classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)

classifier("Washington Gives Ankara An Ultimatum | Trump Makes BigTime Overture To Iran | Authoritarian Spirits Congress The Espionage Act And Punishing WikiLeaks | Analysis Of The European Parliamentary Elections | One Mans Quest To Expose A Fake BBC Video About Syria | China Holds Three Trump Cards In War Against US | Maldives Affirms Fealty To Diego Garcia | The End Of Theresa May | Within The Church People Can Become Truly Free | China Hails Modi Victory This Is Why | ")
```
The returned value `LABEL_1` means that the model thinks the user is a state operator. Otherwise, it will return `LABEL_0`.

## Data
The source data for this project was in the form of raw CSVs provided by [Twitter Information Operations](https://transparency.twitter.com/en/reports/information-operations.html). For this project, we used Russian and Chinese datasets from 2019-2021. CSVs should be extracted and placed in a `raw_downloads` folder in the repo's root directory. We have not included the raw data due to size issues. Natural tweets were obtained through `tweepy`, by conducting a search for tweets that mention the named entities extracted from the state operator tweets.

All data underwent the following process:
- Removed retweets
- Cleaned text of special characters, @ mentions, newlines, URLs, etc. This is documented in `tweet_utils`.
- Each tweet was concatenated with the 9 previous tweets from that user to form a 10-tweet sequence in reverse chronological order.
- Tweets were marked as 
- Empty tweets, excessively short tweet sequences, etc. were removed.
- The cleaned data was divided into train and test sets and placed in the `data/` folder.

## Training
The training sequences were further divided into training and validation datasets. We then used the data to finetune the model `distilbert-base-uncased` from HuggingFace. The training process is documented in notebook 5.

## Results

## Limitations
Although this model demonstrates very high accuracy on the validation and test datasets, our data has a recency bias that may have affected some of the results. Specifically, the state operator tweets were mostly from 2019-2021 (with a small proportion scattered in from previous years), whereas the normal user tweets resulted from searches done in May 2022 (and expanded by looking at each users' previous 50 tweets). Thus, the model may be taking recent news topics into account when performing inference.

However, spot checks do seem to show good results even when topics not specific to 2022 or 2019-2021 are mentioned in the tweet strings, so most of the inference likely relies on language style, specific entities, sentiment, etc.

As Twitter releases new datasets of identified state operators, we will test these and see how our model performs. Updated findings will be posted to this readme.

## Contributors
Curt Tigges: data collection & preparation; model training; model deployment
Bryce Meyer: browser extensions