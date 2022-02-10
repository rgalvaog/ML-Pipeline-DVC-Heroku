## Model Details

This project was conducted as part of the Udacity MLOps Engineer Nanodegree. The code contained herein was created by Rafael Guerra, with the help from instructor feedback and previously asked questions in the Udacity course forum. After experimenting with the SVM (Support Vector Machines) and the Logistic Regression models from scikit learn, I achieved better results with the latter, although the performance of both models still leave a lot to be desired and certainly has room for improvement.

## Intended Use

For educational purposes only. Not suitable for business cases.

## Training Data

The training data came from US census data that is publicly available. The file itself was downloaded directly from the Udactiy portal. Using scikit learn, I split the data into training and test sets, with training data being comprised of 80% of the data set.

## Evaluation Data

The ramaining 20% of the Census data was used as the test set.

## Metrics
_Please include the metrics used and your model's performance on those metrics._

Metrics Used:

```Precision```
Number of positive predictions that are actually positive.

```Recall```
Number of positive predictions made from all the total positive predictions that should have been made.

```FBeta```
The coefficient of the harmonic mean between precision and recall; usually a measure that combines both if you only want to look at a single metric to understand model performance.

My numbers:

Precision: 0.9
Recall: 0.16
FBeta: 0.24

## Ethical Considerations

The data was anonymized so there are no security concerns. It is important to take into consideration that this is a small dataset used for academic purposes only. Broader generalizations about society at large should not be extrapolated from this.

## Caveats and Recommendations

Overall, the performance of the model was poor. There are many things that could have been done to improve the model, although they were outside of the scope of the project and course as a whole (whose purpose is to understand the pipeline of ML). One improvement could be changing the training and testing partitions such that there is more data available in the test set. The high precision and low recall suggests over-fitting. In addition, more could have been done to clean the data, remove outliers.
