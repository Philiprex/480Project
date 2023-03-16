from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import make_pipeline

cat_pipe = make_pipeline(
  OneHotEncoder()
)

num_pipe = make_pipeline(
  StandardScaler()
)

preprocessing = ColumnTransformer([
    ("cat", cat_pipe, make_column_selector(dtype_include=object))
],
  remainder=num_pipe
)

def process(data):
    return preprocessing.fit_transform(data)