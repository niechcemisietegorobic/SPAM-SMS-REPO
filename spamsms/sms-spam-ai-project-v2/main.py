from pathlib import Path

from src.data_preparation import load_sms_data, add_manual_features, get_manual_xy, get_text_xy, dataset_summary
from src.experiments import (
    run_gnb_experiment,
    run_tfidf_experiment,
    run_knn_experiment,
    compare_gnb_inputs,
    compare_gnb_resampling,
    cross_validate_all,
    summarize_cv,
    wilcoxon_tests,
)
from src.visualizations import (
    save_main_gnb_figure,
    save_probability_histogram,
    save_confusion_matrix,
    save_f1_boxplot,
    save_gnb_input_comparison,
    save_resampling_comparison,
)


DATA_PATH = Path('data/spam.csv')
FIGURES_DIR = Path('figures')
FIGURES_DIR.mkdir(exist_ok=True)


def print_single_result(result):
    print('\n' + '=' * 50)
    print(result['name'])
    print('=' * 50)
    print('Accuracy:', round(result['accuracy'], 4))
    print('Balanced accuracy:', round(result['balanced_accuracy'], 4))
    print('PR-AUC:', round(result['pr_auc'], 4))
    print('ROC-AUC:', round(result['roc_auc'], 4))
    print(result['classification_report'])
    print('Macierz pomylek:')
    print(result['confusion_matrix'])
    print('TN:', result['tn'], 'FP:', result['fp'], 'FN:', result['fn'], 'TP:', result['tp'])


def main():
    dane = load_sms_data(DATA_PATH)
    dane = add_manual_features(dane)
    summary = dataset_summary(dane)

    print('Lacznie wiadomosci:', summary['total'])
    print('Ham:', summary['ham'], '(', summary['ham_percent'], '%)')
    print('Spam:', summary['spam'], '(', summary['spam_percent'], '%)')

    X, y = get_manual_xy(dane)
    X_text, y_text = get_text_xy(dane)

    gnb_result = run_gnb_experiment(X, y)
    tfidf_result = run_tfidf_experiment(X_text, y_text)
    knn_result = run_knn_experiment(X, y)

    print_single_result(gnb_result)
    print_single_result(tfidf_result)
    print_single_result(knn_result)

    save_main_gnb_figure(dane, summary, gnb_result, FIGURES_DIR / 'projekt_gnb.png')
    save_probability_histogram(gnb_result, FIGURES_DIR / 'histogram_prob.png')
    save_confusion_matrix(tfidf_result['confusion_matrix'], 'Macierz pomylek - TF-IDF + ComplementNB', 'Greens', FIGURES_DIR / 'macierz_tfidf.png')
    save_confusion_matrix(knn_result['confusion_matrix'], 'Macierz pomylek - KNN + StandardScaler', 'Purples', FIGURES_DIR / 'macierz_knn.png')

    print('\n' + '=' * 50)
    print('DODATKOWY EKSPERYMENT - GaussianNB na roznych inputach')
    print('=' * 50)
    gnb_input_rows = compare_gnb_inputs(dane)
    for row in gnb_input_rows:
        print('\nInput:', row['name'])
        print('Cechy:', row['features'])
        print('Accuracy:', round(row['accuracy'], 4))
        print('Precision spam:', round(row['precision_spam'], 4))
        print('Recall spam:', round(row['recall_spam'], 4))
        print('F1 spam:', round(row['f1_spam'], 4))
    save_gnb_input_comparison(gnb_input_rows, FIGURES_DIR / 'gnb_porownanie_cech.png')

    print('\n' + '=' * 50)
    print('DODATKOWY EKSPERYMENT - RandomOverSampler i SMOTE')
    print('=' * 50)
    resampling_rows = compare_gnb_resampling(X, y)
    for row in resampling_rows:
        print('\nMetoda:', row['name'])
        print('Liczba ham w treningu:', row['train_ham'])
        print('Liczba spam w treningu:', row['train_spam'])
        print('Precision spam:', round(row['precision_spam'], 4))
        print('Recall spam:', round(row['recall_spam'], 4))
        print('F1 spam:', round(row['f1_spam'], 4))
        print('Balanced accuracy:', round(row['balanced_accuracy'], 4))
        print('Macierz pomylek:')
        print(row['confusion_matrix'])
    save_resampling_comparison(resampling_rows, FIGURES_DIR / 'resampling_gnb_porownanie.png')

    print('\n' + '=' * 50)
    print('WALIDACJA KRZYZOWA')
    print('=' * 50)
    cv_results = cross_validate_all(X, y, X_text, y_text)
    cv_summary = summarize_cv(cv_results)
    for row in cv_summary:
        print('\n' + row['model'])
        for metric in ['accuracy', 'balanced_accuracy', 'precision_spam', 'recall_spam', 'f1_spam', 'roc_auc']:
            print(metric, ':', round(row[metric + '_mean'], 4), '+/-', round(row[metric + '_std'], 4))

    save_f1_boxplot(cv_results, FIGURES_DIR / 'porownanie_f1_3modele_boxplot.png')

    print('\n' + '=' * 50)
    print('ANALIZA STATYSTYCZNA - TEST WILCOXONA')
    print('=' * 50)
    tests = wilcoxon_tests(cv_results)
    for name, values in tests.items():
        print('\n' + name)
        print('mean A:', round(values['mean_a'], 4))
        print('mean B:', round(values['mean_b'], 4))
        print('statistic:', round(values['statistic'], 4))
        print('p-value:', values['p_value'])
        print('istotna roznica:', values['significant'])

    print('\n' + '=' * 50)
    print('KONCOWY WNIOSEK')
    print('=' * 50)
    print('Porownano trzy metody rozpoznawania spamu: GaussianNB, KNN oraz TF-IDF + ComplementNB.')
    print('Najlepszy sredni F1-score dla klasy Spam uzyskal model TF-IDF + ComplementNB.')
    print('Dodatkowo sprawdzono rozne inputy dla GaussianNB oraz resampling: RandomOverSampler i SMOTE.')
    print('Gorszy wynik w niektorych wariantach nie jest bledem - celem bylo porownanie wplywu cech i resamplingu.')


if __name__ == '__main__':
    main()
