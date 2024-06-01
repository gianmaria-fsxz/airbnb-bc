The workflow of the project is:

# Basic Explanation
There is some eda in the folder `notebooks`.
We can actually see that data is pretty clean and selfspeaking, so I suppose the important question is what do we wanto to do with it.
Given the fact that the airbnb properties has not su much information, I tried to realize a basic e2e project that uses this data to predict if the property will stay in the airbnb collection in the next months

In order to do this, I tried to productionalize a simple worfklow consisting of
- creation a feature store (inspectable with the command `feast ui`). Feast is one of the loeader frameworks in the feature store paradigm. It comes as an open-source project pretty much integrated with the most important cloud providers. For example it has specific connectors with GCS and BQ.

- basic data preparation, with just feature engineering for the geographical info: I created two new feature as the number of neighbours for every property and the distance in km from the baricenter. This step provides two datasets as output. The first contains the data from the past and the second one the data that the model will predict

- materialization of the feature store. We ingest in the feast app the metadata useful for the fs. for this project i just implemented offline batch feature store, but I also explored the possibility of on_demand features! Given the great amount of data, sometimes having on_demand feature is necessary for streaming data use cases

- training a binary classification model using the data from the past as the learning set in order to predict which properties will leave airbnb. the implementation is really simple and is made with `mlflow`, one of the state of the art mlops framework. It gives us the opportunity to monitor the performances of the model and track every run (both in inference and training)

- last but not least, we retrieve from the mlflow local server the trained model in order to apply it to the test dataset. Before doing this, it's necessary (specially in production) to be sure of the quality of data in input. In order to do so, the software exploit the integration between feast and great expectation, a data validation framework.
In this case I wrote some quality checks that block the execution of the prediction, just for the sake of this little project.

# Installation steps
I used python 3.10.12 version on a linux machine

```sh
python -m venv airbnb-env
```

```sh
source airbnb-env/bin/activate
```

```sh
pip install -r requirements
```

```sh
cd src
```

```sh
python launch_project.py
```

```sh
python train_model.py
```

```sh
python inference.py
```

In order to check things on web browser (mlflow and feast ui), one could comment line 143 of `inference.py`
I wrote that because on my laptop the two clients were struggling a little bit after running for a while
