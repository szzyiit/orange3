# pip install bayesian-optimization
# - 选择题参数名|中文描述|选项,格式为: [('选项1', 值, 是否默认), ('选项2', 值, 是否默认)]
# - 填空题参数名|中文描述|[默认值]
# 安装： pip install bayesian-optimization
'''
- y_name|目标名称|['sales']
- min_tree|最小树数量|[10]
- max_tree|最大树数量|[250]
- min_min_samples_split|最小分叉的最少节点数|[2]
- max_min_samples_split|最小分叉的最多节点数|[25]
- min_max_features|最小分叉考虑的特征比例|[0.1]
- max_max_features|最小分叉考虑的特征比例|[0.999]
返回值: 
'''

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC

from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours

min_tree = int(in_params['min_tree'])
max_tree = int(in_params['max_tree'])
min_min_samples_split = int(in_params['min_min_samples_split'])
max_min_samples_split = int(in_params['max_min_samples_split'])
min_max_features = float(in_params['min_max_features'])
max_max_features = float(in_params['max_max_features'])

# def get_data():
#     """Synthetic binary classification dataset."""
#     data, targets = make_classification(
#         n_samples=1000,
#         n_features=45,
#         n_informative=12,
#         n_redundant=7,
#         random_state=134985745,
#     )
#     return data, targets
X = in_data.X
y = in_data.Y

def rfc_cv(n_estimators, min_samples_split, max_features, data, targets):
    """Random Forest cross validation.
    This function will instantiate a random forest classifier with parameters
    n_estimators, min_samples_split, and max_features. Combined with data and
    targets this will in turn be used to perform cross validation. The result
    of cross validation is returned.
    Our goal is to find combinations of n_estimators, min_samples_split, and
    max_features that minimzes the log loss.
    """
    estimator = RFC(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        max_features=max_features,
        random_state=2
    )
    cval = cross_val_score(estimator, data, targets,
                           scoring='neg_log_loss', cv=4)
    return cval.mean()

def optimize_rfc(data, targets):
    """Apply Bayesian Optimization to Random Forest parameters."""
    def rfc_crossval(n_estimators, min_samples_split, max_features):
        """Wrapper of RandomForest cross validation.
        Notice how we ensure n_estimators and min_samples_split are casted
        to integer before we pass them along. Moreover, to avoid max_features
        taking values outside the (0, 1) range, we also ensure it is capped
        accordingly.
        """
        return rfc_cv(
            n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=max(min(max_features, 0.999), 1e-3),
            data=data,
            targets=targets,
        )

    optimizer = BayesianOptimization(
        f=rfc_crossval,
        pbounds={
            "n_estimators": (min_tree, max_tree),
            "min_samples_split": (min_min_samples_split, max_min_samples_split),
            "max_features": (min_max_features, max_max_features),
        },
        random_state=1234,
        verbose=2
    )
    optimizer.maximize(n_iter=10)

    print("Final result:", optimizer.max)

print(Colours.green("--- Optimizing Random Forest ---"))
optimize_rfc(X, y)