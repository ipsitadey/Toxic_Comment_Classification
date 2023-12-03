# Toxic_Comment_Classification

Social media and online portals are becoming raging platforms for expressing hatred and biases towards certain communities. **Cyber Bullying** has become a serious worldwide issue and needs to be countered. Our goal is to -
> identify hateful and abusive content with high accuracy so that one could safely use these platforms to express their opinion without being disrespectful.

To address these needs, in this study we investigate the ability of **BERT(Bidirectional Encoder Representations from Transformers)**, a pre-trained state-of the art NLP model, at capturing hateful context within social media content by using new finetuning methods based on transfer learning.

## Data Sets
We used publically available kaggle data set

> kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification

## What is BERT?
BERT, which stands for Bidirectional Encoder Representations from Transformers, is a natural language processing (NLP) model developed by Google AI in 2018. It represents a significant advancement in the field of NLP and has had a profound impact on various NLP tasks. Key features and concepts are:

&ensp;&ensp;**Transformer Architecture:** BERT is built on the Transformer architecture, which is a neural network architecture introduced in the paper "Attention Is All You Need" by Vaswani et al. Transformers are known for their ability to capture long-range dependencies in sequences, making them well-suited for NLP tasks.

&ensp;&ensp;**Bidirectional Context:** Unlike previous NLP models that processed text in a left-to-right or right-to-left manner, BERT is designed to consider the context from both directions simultaneously. This bidirectional context modeling allows it to capture more nuanced and context-aware representations of words.

&ensp;&ensp;**Masked Language Modeling:** During pretraining, BERT uses a technique called masked language modeling (MLM). It randomly masks some of the words in a sentence and then learns to predict the masked words based on the surrounding context. This helps BERT understand the relationships between words and their context.

&ensp;&ensp;**Advantages of BERT over other language models:**
* General purpose Bidirectional Contextual language model
* A Transformer Encoder stack trained on Wikipedia and Book Corpus
* Performs better than deep learning models when finetuned for classification task

## Novel processing ideas of BERT
*   Way to “fill in the blank” based on context. e.g: *“She bought a _____ of shoes.”* &rarr; *pair*.<br>While current state-of-the-art OpenAI GPT represent *pair* based on *"she bought a"* but not on *"of shoes"*, BERT uses both previous and next context at the same time.<br>
<img src="images/BERTvsOpenAI.ppm" alt="Adapter BERT" width="600"/>

*   Unique input token embedding<br>
<img src="images/token_embedding.png" alt="Adapter BERT" width="600"/>

*   Train Strategy: Masked LM -Randomly replace 15% of the words with a [MASK] token and try to predict them.
<img src="images/next_pred.png" alt="Adapter BERT" width="600"/>

## Different BERT Fine Tuning Approaches:

&ensp;&ensp;**1. Adapter Module Transfer Learning**
* Add a new modules
* Freeze original params and Train only new ones<br><br>
$` 
\begin{align*}
& Adapter(h_{j}) = W^{D}(non-linearity(W^{E}h_{j}) + b_{E}) + b_{D}\\
\\
& W^{E}: projects\ the\ input\ to\ a\ smaller\ space\\
& W^{D}: projects\ the\ input\ to\ the\ original\ size\\
& b: are\ biases\\
\end{align*}
`$

<a href="https://medium.com/dair-ai/adapters-a-compact-and-extensible-transfer-learning-method-for-nlp-6d18c2399f62"><img src="images/adapter_bert.webp" alt="Adapter BERT" width="600"/></a>

&ensp;&ensp;**2. DistilBERT – distil versioned BERT**
* Smaller, faster, cheaper model.
* Compress BERT by 40% , retaining 97% language understanding capability.
* Token-type embeddings removed.
* Layers reduced by a factor of 2.
* Cross Entropy Loss: $` L = -\sum_{i} t_{i} * log(s_{i}) `$

&ensp;&ensp;**3. Discriminative Fine Tuning**
* Tuning with Varying Learning Rate.
* Different layers capture different types of information.
* Instead of using regular SGD, use L different learning rates in decreasing order while moving to lower layers<br>
&ensp;&ensp;$`
\begin{align*}
& \Theta _{t}^{l} = \Theta _{t-1}^{l} - \eta ^{l}.\bigtriangledown _{\Theta ^{l}}J(\Theta)\\
& \eta ^{l-1} = \eta ^{l}/r
\end{align*}
`$

&ensp;&ensp;**4. BERT with LSTM**
* Feature Based Tuning.
* Extract features by gradual freezing of DistilBERT.
* Classification using LSTM to preserve long distance dependency of words.<\br>

<img src="images/bert_lstm.png" alt="Adapter BERT" width="600"/>


## Experiment Result
Comparative study of different fine-tuned BERT Models:

| Models                                                          | # of Parameters to Train  | Accuracy %  |
| --------------------------------------------------------------- | -------------------------:| -----------:|
| DistilBert + Adaptive Model + Discriminative Learning Rate      |   1,783,298               | 91.77       |
| DistilBert + Gradual Freezing + Discriminative Learning Rate    |   7,680,002               |   87        |
| BERT with Adapter                                               |   3,009,794               |   90        |
| BERT Baseline (bert-base-uncased)                               | 109,483,778               |   91        |

Output Analysis:
* BERT baseline (12 layer) - very high accuracy but computationally expensive.
* DistilBERT (6 layers) + Discriminative Learning rate + Adaptive model - gives same accuracy but is much light weight, faster and has least number of trainable parameters.
* Time - Final tuned model takes 1 hour and yet achieves 91% accuracy unlike baseline(5 hours)<br><br>
![DistilBERT with Adapter](images/distilbert_adapter.png)

## Resources
1. <a href="https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270">BERT Explained: State of the art language model for NLP</a>
2. <a href="https://medium.com/dair-ai/adapters-a-compact-and-extensible-transfer-learning-method-for-nlp-6d18c2399f62">Adapters: A Compact and Extensible Transfer Learning Method for NLP</a>
3. <a href="https://medium.com/huggingface/distilbert-8cf3380435b5">Smaller, faster, cheaper, lighter: Introducing DistilBERT, a distilled version of BERT</a>
4. <a href="https://paperswithcode.com/method/discriminative-fine-tuning">Discriminative Fine-Tuning</a>
