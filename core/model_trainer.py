from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier


def train_enhanced_model(X, y):
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 配置XGBoost进行二分类
    model = XGBClassifier(
        n_estimators=200,      # 树的数量
        max_depth=7,           # 树的最大深度
        learning_rate=0.1,     # 学习率
        subsample=0.8,         # 训练样本子采样比例
        colsample_bytree=0.8,  # 特征子采样比例
        random_state=42,       # 随机种子
        objective='binary:logistic',  # 二分类目标函数
        eval_metric='logloss'         # 二分类评估指标
    )

    # 使用基本参数训练模型
    model.fit(X_train, y_train)

    return model, X_test, y_test