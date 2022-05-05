# Abusive Language Detection in Social Media
#### Authors: Mocanu Sebastian, Mogoase Ana-Maria Luisa

In theory online platforms are a place for healthy interaction of the users, but in reality,
some people are toxic with each other, or they tend to express their opinion about some subject
in an aggressive way. A way to reduce this behavior in the online environment is to identify 
the abusive text with an abusive language detection model, then delete or report it.

### Technologies used

- Python 3.8.6 (with the modules from "requirements.txt")
- Django / FastAPI / Flask Framework
- Twitter API
- HTML, CSS, JavaScript


### Setup
With the project on the local machine, you need to create a virtual environment. Then, after 
activating it run the following command in the root directory in order to install the required
dependencies to run the project:
```
python -m pip install -r "requirements.txt"
```
Since the models take quite a lot of memory, they can't be stored here, and you need to train in
order to generate them. After generation, you can run the application (`TODO:` Add description for 
the framework used).


## Dataset References
1. Waseem, Zeerak, and Dirk Hovy. "Hateful symbols or hateful people? predictive
features for hate speech detection on twitter." Proceedings of the NAACL student
research workshop. (2016).
2. Waseem, Zeerak. "Are you a racist or am I seeing things? annotator influence on hate
speech detection on twitter." Proceedings of the first workshop on NLP and
computational social science. (2016).