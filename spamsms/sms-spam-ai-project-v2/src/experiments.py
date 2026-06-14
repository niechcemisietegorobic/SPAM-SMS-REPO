import numpy as np
from scipy.stats import wilcoxon
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


SCORING = {
    'accuracy': 'accuracy',
    'balanced_accuracy': 'balanced_accuracy',
    'precision_spam': 'precision',
    'recall_spam': 'recall',
    'f1_spam': 'f1',
    'roc_auc': 'roc_auc',
}


FEATURE_SETS = {
    'tylko dlugosc': ['dlugosc'],
    'tylko wykrzykniki': ['wykrzykniki'],
    'tylko cyfry': ['cyfry'],
    'dlugosc + slowa': ['dlugosc', 'slowa'],
    'cyfry + wykrzykniki': ['cyfry', 'wykrzykniki'],
    'wszystkie 5 cech': ['dlugosc', 'slowa', 'wielkie', 'cyfry', 'wykrzykniki'],
}

# MajorityClassifier
class MajorityClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        klasy, licznosci = np.unique(y, return_counts=True)
        self.klasa_wiekszosciowa = klasy[np.argmax(licznosci)]
        return self

    def predict(self, X):
        return np.full(len(X), self.klasa_wiekszosciowa)

def _metrics_dict(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'pr_auc': average_precision_score(y_true, y_prob),
        'roc_auc': auc(fpr, tpr),
        'classification_report': classification_report(y_true, y_pred, target_names=['Ham', 'Spam']),
        'confusion_matrix': cm,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'fpr': fpr,
        'tpr': tpr,
    }


def _simple_scores(y_true, y_pred):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'precision_spam': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall_spam': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_spam': float(f1_score(y_true, y_pred, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
    }


# gaussiannb
def run_gnb_experiment(X, y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    result = _metrics_dict(y_test, y_pred, y_prob)
    result.update({
        'name': 'GaussianNB + ręczne cechy',
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'train_size': len(X_train),
        'test_size': len(X_test),
    })
    return result


# tf-idf + complementnb
def run_tfidf_experiment(X_text, y_text, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y_text, test_size=0.2, random_state=random_state, stratify=y_text
    )
    model = Pipeline([
        ('tfidf', TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2)),
        ('clf', ComplementNB()),
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    result = _metrics_dict(y_test, y_pred, y_prob)
    result.update({
        'name': 'TF-IDF + ComplementNB',
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'train_size': len(X_train),
        'test_size': len(X_test),
    })
    return result


# knn + standardscaler
def run_knn_experiment(X, y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=5)),
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    result = _metrics_dict(y_test, y_pred, y_prob)
    result.update({
        'name': 'KNN + StandardScaler + ręczne cechy',
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'train_size': len(X_train),
        'test_size': len(X_test),
    })
    return result


def compare_gnb_inputs(dane, random_state=42):
    """Porownuje GaussianNB dla roznych zestawow tych samych starych cech."""
    y = (dane['etykieta'] == 'spam').astype(int).values
    rows = []
    for name, cols in FEATURE_SETS.items():
        X = dane[cols].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores = _simple_scores(y_test, y_pred)
        scores.update({'name': name, 'features': cols})
        rows.append(scores)
    return rows


def compare_gnb_resampling(X, y, random_state=42):
    """Porownuje GaussianNB bez resamplingu, z RandomOverSampler i ze SMOTE."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    samplers = {
        'bez resamplingu': None,
        'RandomOverSampler': RandomOverSampler(random_state=random_state),
        'SMOTE': SMOTE(random_state=random_state),
    }
    rows = []
    for name, sampler in samplers.items():
        if sampler is None:
            X_train_final, y_train_final = X_train, y_train
        else:
            X_train_final, y_train_final = sampler.fit_resample(X_train, y_train)

        model = GaussianNB()
        model.fit(X_train_final, y_train_final)
        y_pred = model.predict(X_test)
        scores = _simple_scores(y_test, y_pred)
        scores.update({
            'name': name,
            'train_ham': int(np.sum(y_train_final == 0)),
            'train_spam': int(np.sum(y_train_final == 1)),
        })
        rows.append(scores)
    return rows

def compare_decisiontree_and_majority_classifier(X, y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    classifiers = {
        'DecisionTree': DecisionTreeClassifier(random_state=random_state),
        'MajorityClassifier': MajorityClassifier(),
    }
    rows = []
    for name, model in classifiers.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores = _simple_scores(y_test, y_pred)
        scores.update({'name': name})
        rows.append(scores)
    return rows


def cross_validate_all(X, y, X_text, y_text, random_state=42):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=random_state)
    models = {
        'gnb': GaussianNB(),
        'knn': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=5)),
        ]),
        'tfidf': Pipeline([
            ('tfidf', TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2)),
            ('clf', ComplementNB()),
        ]),
    }
    results = {
        'gnb': cross_validate(models['gnb'], X, y, cv=cv, scoring=SCORING),
        'knn': cross_validate(models['knn'], X, y, cv=cv, scoring=SCORING),
        'tfidf': cross_validate(models['tfidf'], X_text, y_text, cv=cv, scoring=SCORING),
    }
    return results


def summarize_cv(cv_results):
    rows = []
    labels = {
        'gnb': 'GaussianNB + ręczne cechy',
        'knn': 'KNN + StandardScaler',
        'tfidf': 'TF-IDF + ComplementNB',
    }
    for key, label in labels.items():
        row = {'model': label}
        for metric in SCORING:
            values = cv_results[key]['test_' + metric]
            row[metric + '_mean'] = float(np.mean(values))
            row[metric + '_std'] = float(np.std(values))
        rows.append(row)
    return rows


def wilcoxon_tests(cv_results):
    f1_gnb = cv_results['gnb']['test_f1_spam']
    f1_knn = cv_results['knn']['test_f1_spam']
    f1_tfidf = cv_results['tfidf']['test_f1_spam']

    comparisons = {
        'GaussianNB vs TF-IDF + ComplementNB': (f1_gnb, f1_tfidf),
        'KNN vs TF-IDF + ComplementNB': (f1_knn, f1_tfidf),
        'GaussianNB vs KNN': (f1_gnb, f1_knn),
    }
    output = {}
    for name, (a, b) in comparisons.items():
        stat, p_value = wilcoxon(a, b)
        output[name] = {
            'statistic': float(stat),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05),
            'mean_a': float(np.mean(a)),
            'mean_b': float(np.mean(b)),
        }
    return output
