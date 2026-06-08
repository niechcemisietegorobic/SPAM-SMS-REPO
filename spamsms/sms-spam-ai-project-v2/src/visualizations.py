import matplotlib.pyplot as plt
import numpy as np


def save_main_gnb_figure(dane, summary, gnb_result, output_path='figures/projekt_gnb.png'):
    ham_dane = dane[dane['etykieta'] == 'ham']
    spam_dane = dane[dane['etykieta'] == 'spam']
    cm = gnb_result['confusion_matrix']

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].pie(
        [summary['ham'], summary['spam']],
        labels=['Ham', 'Spam'],
        colors=['steelblue', 'tomato'],
        autopct='%1.1f%%',
        startangle=90,
    )
    axes[0, 0].set_title('Reprezentacja zbioru danych')

    axes[0, 1].scatter(ham_dane['slowa'], ham_dane['dlugosc'], color='steelblue', alpha=0.3, s=10, label='Ham')
    axes[0, 1].scatter(spam_dane['slowa'], spam_dane['dlugosc'], color='tomato', alpha=0.5, s=10, label='Spam')
    axes[0, 1].set_xlabel('Liczba slow')
    axes[0, 1].set_ylabel('Dlugosc wiadomosci')
    axes[0, 1].set_title('Scatter: dlugosc vs slowa')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].imshow(cm, cmap='Blues')
    axes[1, 0].set_xticks([0, 1]); axes[1, 0].set_xticklabels(['Ham', 'Spam'])
    axes[1, 0].set_yticks([0, 1]); axes[1, 0].set_yticklabels(['Ham', 'Spam'])
    axes[1, 0].set_xlabel('Przewidywana klasa')
    axes[1, 0].set_ylabel('Rzeczywista klasa')
    axes[1, 0].set_title('Macierz pomylek')
    for i in range(2):
        for j in range(2):
            kolor = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            axes[1, 0].text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=16, color=kolor)

    axes[1, 1].plot(gnb_result['fpr'], gnb_result['tpr'], color='tomato', lw=2, label='AUC = ' + str(round(gnb_result['roc_auc'], 3)))
    axes[1, 1].plot([0, 1], [0, 1], color='gray', linestyle='--')
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('Krzywa ROC')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_probability_histogram(result, output_path='figures/histogram_prob.png'):
    plt.figure(figsize=(10, 5))
    bins = np.linspace(0, 1, 40)
    y_test = result['y_test']
    y_prob = result['y_prob']
    plt.hist(y_prob[y_test == 0], bins=bins, color='steelblue', alpha=0.7, label='Ham')
    plt.hist(y_prob[y_test == 1], bins=bins, color='tomato', alpha=0.7, label='Spam')
    plt.axvline(0.5, color='black', linestyle='--', label='Prog decyzyjny 0.5')
    plt.xlabel('Prawdopodobienstwo spamu')
    plt.ylabel('Liczba wiadomosci')
    plt.title('Rozklad prawdopodobienstwa - GNB')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_confusion_matrix(cm, title, cmap, output_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap=cmap)
    plt.xticks([0, 1], ['Ham', 'Spam'])
    plt.yticks([0, 1], ['Ham', 'Spam'])
    plt.xlabel('Przewidywana klasa')
    plt.ylabel('Rzeczywista klasa')
    plt.title(title)
    for i in range(2):
        for j in range(2):
            kolor = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=16, color=kolor)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_f1_boxplot(cv_results, output_path='figures/porownanie_f1_3modele_boxplot.png'):
    f1_gnb = cv_results['gnb']['test_f1_spam']
    f1_knn = cv_results['knn']['test_f1_spam']
    f1_tfidf = cv_results['tfidf']['test_f1_spam']

    plt.figure(figsize=(9, 5))
    plt.boxplot(
        [f1_gnb, f1_knn, f1_tfidf],
        tick_labels=['GaussianNB\nreczne cechy', 'KNN\nStandardScaler', 'TF-IDF\nComplementNB'],
    )
    plt.ylabel('F1-score dla klasy Spam')
    plt.title('Porownanie wynikow F1 w walidacji krzyzowej')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_gnb_input_comparison(rows, output_path='figures/gnb_porownanie_cech.png'):
    """Zapisuje wykres porownujacy GaussianNB na roznych zestawach starych cech."""
    names = [r['name'] for r in rows]
    precision = [r['precision_spam'] for r in rows]
    recall = [r['recall_spam'] for r in rows]
    f1 = [r['f1_spam'] for r in rows]

    x = np.arange(len(names))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, precision, width, label='Precision spam')
    plt.bar(x, recall, width, label='Recall spam')
    plt.bar(x + width, f1, width, label='F1 spam')
    plt.xticks(x, names, rotation=30, ha='right')
    plt.ylabel('Sredni wynik / wynik testowy')
    plt.title('GaussianNB - porownanie roznych inputow')
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_resampling_comparison(rows, output_path='figures/resampling_gnb_porownanie.png'):
    """Zapisuje wykres porownujacy brak resamplingu, RandomOverSampler i SMOTE."""
    names = [r['name'] for r in rows]
    precision = [r['precision_spam'] for r in rows]
    recall = [r['recall_spam'] for r in rows]
    f1 = [r['f1_spam'] for r in rows]
    bal_acc = [r['balanced_accuracy'] for r in rows]

    x = np.arange(len(names))
    width = 0.2

    plt.figure(figsize=(10, 6))
    plt.bar(x - 1.5 * width, precision, width, label='Precision spam')
    plt.bar(x - 0.5 * width, recall, width, label='Recall spam')
    plt.bar(x + 0.5 * width, f1, width, label='F1 spam')
    plt.bar(x + 1.5 * width, bal_acc, width, label='Balanced accuracy')
    plt.xticks(x, names)
    plt.ylabel('Wynik')
    plt.title('GaussianNB - porownanie resamplingu')
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
