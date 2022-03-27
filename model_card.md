# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Di Lin created this model. It is a random forest tree model built on census data from https://archive.ics.uci.edu/ml/datasets/census+income.

## Intended Use
This model should be used to predict the salary range based off a handful of attributes. The users are researchers interested in income distributions.

## Training Data
The training data took 80% of the original dataset and performanced on-hot encoding for all the categorical columns.

## Evaluation Data
The evaluation data took the remaining 20% of the original dataset and performanced the same transformation steps as the training data.

## Metrics
The model was evaluated using precision (0.74), recall (0.64) and fbeta(0.69).

## Ethical Considerations
The model is performing differently for people with different education backgrounds. This introduces bias to the inference results.

## Caveats and Recommendations
A more balanced dataset could be sampled to improve the model performance on sub populations.
