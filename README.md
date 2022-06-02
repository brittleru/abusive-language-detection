# Abusive Language Detection in Social Media
#### Authors: Mocanu Sebastian, Mogoase Ana-Maria Luisa <br> Advisors: Rebedea Traian Eugen, Chiru Costin Gabriel


In theory online platforms are a place for healthy interaction of the users, but in reality,
some people are toxic with each other, or they tend to express their opinion about some subject
in an aggressive way. A way to reduce this behavior in the online environment is to identify 
the abusive text with an abusive language detection model, then delete or report it.

We provide an analysis over the existent state-of-the-art abusive language datasets and deep learning models in order 
to improve the classification of abusive language. By using **[X]** models on each dataset and having 5 datasets
that are different we can gain an enhanced hate-speech detection algorithm.

### Technologies used

- Python 3.8.6 (with the modules from "requirements.txt")
- Tensorflow
- PyTorch
- Django Framework for backend
- Twitter API
- HTML, CSS, JavaScript and Bootstrap for UI


### Setup
With the project on the local machine, you need to create a virtual environment. Then, after 
activating it run the following command in the root directory in order to install the required
dependencies to run the project:
```
python -m pip install --upgrade pip
python -m pip install -r "requirements.txt"
```
If it's the **first time** running the project, you might need to download some `NLTK` data, for a quick
fix you can uncomment the downloading instructions that can be found in `utils.py` file (e.g., 
`nltk.download("stopwords")` line). 

Since the models take quite a lot of memory (some of them 5-8 GB), they can't be stored on GitHub, 
and you need to train in order to generate them. 
You can train each model individually, for each dataset there is a python package which has the 
code for training the models, but you can also train every model at once by running `main.py`, this will 
populate the logs, and the models directories that are needed in order to run the web application.

**Note: do this only if you have a strong PC.** 

Currently, we provide a Google Drive link that has already trained models to download in order to 
run the project. (`TODO:` **INSERT GOOGLE DRIVE LINK HERE**)



After downloading or generation of the models, you can run the application (`TODO:` Add description for 
the framework used).


### Application Demo
```
[INSERT YOUTUBE LINK HERE]
```

## Dataset References
1. Vidgen, Bertie, et al. "Learning from the worst: Dynamically generated datasets to improve online hate detection." 
   arXiv preprint arXiv:2012.15761 (2020).
2. Davidson, Thomas, et al. "Automated hate speech detection and the problem of
   offensive language." Proceedings of the International AAAI Conference on Web and
   Social Media. Vol. 11. No. 1. (2017).
3. Zampieri, Marcos, et al. "Predicting the type and target of offensive posts in social media." arXiv 
   preprint arXiv:1902.09666 (2019).
4. Founta, Antigoni Maria, et al. "Large scale crowdsourcing and characterization of twitter abusive behavior." Twelfth
   International AAAI Conference on Web and Social Media (2018).
5. Waseem, Zeerak, and Dirk Hovy. "Hateful symbols or hateful people? predictive features for hate speech detection on 
   twitter." Proceedings of the NAACL student research workshop (2016).
6. Röttger, Paul, et al. "Hatecheck: Functional tests for hate speech detection models." arXiv preprint arXiv:2012.15606 (2020).
   