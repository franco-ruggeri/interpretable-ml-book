import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.preprocessing import OneHotEncoder

# Load data
df = (
    pd.read_csv("../data/bike.csv")
    .drop(["yr", "mnth", "weekday"], axis=1)
)
x = df.loc[:, df.columns != "cnt"]
y = df.loc[:, "cnt"]

# One-hot encoding of categorical features
# TODO: useless to drop feature in decision tree, I have to do it in linear regression
categorical_features = ["season", "weathersit", "holiday", "workingday"]
x_categorical = x[categorical_features]
encoder = OneHotEncoder(drop=["WINTER", "GOOD", "NO HOLIDAY", "NO WORKING DAY"])
encoder.fit(X=x_categorical)
categories = [
    encoder.categories_[i][j].lower().replace(" ", "_")
    for i in range(len(encoder.categories_))
    for j in range(len(encoder.categories_[i])) if j != encoder.drop_idx_[i]
]
x_categorical = pd.DataFrame(data=encoder.transform(x_categorical).toarray(), columns=categories)
x = pd.concat([x.drop(categorical_features, axis=1), x_categorical], axis=1)

# Model training
decision_tree = DecisionTreeRegressor(max_depth=2)
decision_tree.fit(X=x, y=y)
figure, axes = plt.subplots()
plot_tree(decision_tree, ax=axes, feature_names=x.columns)
figure.show()
figure.savefig("decision_tree.pdf")

# Explanations
figure, axes = plt.subplots()
feature_importances = pd.Series(decision_tree.feature_importances_, index=x.columns)
feature_importances.plot.bar(ax=axes)
figure.tight_layout()
figure.show()
figure.savefig("feature_importance.pdf")
