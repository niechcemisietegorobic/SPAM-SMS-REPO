import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc

# wczytanie danych
dane = pd.read_csv('spam.csv', encoding='latin-1')
dane = dane[['v1', 'v2']].rename(columns={'v1': 'etykieta', 'v2': 'wiadomosc'})

# ile mamy spamu a ile hamu
n_ham = (dane['etykieta'] == 'ham').sum()
n_spam = (dane['etykieta'] == 'spam').sum()
total = len(dane)

print("Lacznie wiadomosci:", total)
print("Ham:", n_ham, " (", round(n_ham/total*100, 1), "%)")
print("Spam:", n_spam, " (", round(n_spam/total*100, 1), "%)")

# ekstrakcja cech numerycznych z tekstu
dane['dlugosc']     = dane['wiadomosc'].apply(len)
dane['slowa']       = dane['wiadomosc'].apply(lambda x: len(x.split()))
dane['wielkie']     = dane['wiadomosc'].apply(lambda x: sum(1 for c in x if c.isupper()))
dane['cyfry']       = dane['wiadomosc'].apply(lambda x: sum(1 for c in x if c.isdigit()))
dane['wykrzykniki'] = dane['wiadomosc'].apply(lambda x: x.count('!'))

# przygotowanie X i y
X = dane[['dlugosc', 'slowa', 'wielkie', 'cyfry', 'wykrzykniki']].values
y = (dane['etykieta'] == 'spam').astype(int).values

# podzial na zbior treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Treningowy:", len(X_train))
print("Testowy:", len(X_test))

# trenowanie modelu GNB
model = GaussianNB()
model.fit(X_train, y_train)

# predykcja
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# wyniki
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", round(acc, 4))
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# macierz pomylek
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print("Macierz pomylek:")
print(cm)
print("TN:", tn, " FP:", fp, " FN:", fn, " TP:", tp)

# -------------------------------------------------------
# wykresy
# -------------------------------------------------------

ham_dane  = dane[dane['etykieta'] == 'ham']
spam_dane = dane[dane['etykieta'] == 'spam']

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# wykres kolowy - reprezentacja zbioru
axes[0, 0].pie(
    [n_ham, n_spam],
    labels=['Ham', 'Spam'],
    colors=['steelblue', 'tomato'],
    autopct='%1.1f%%',
    startangle=90
)
axes[0, 0].set_title('Reprezentacja zbioru danych')

# scatter - dlugosc vs slowa
axes[0, 1].scatter(ham_dane['slowa'],  ham_dane['dlugosc'],  color='steelblue', alpha=0.3, s=10, label='Ham')
axes[0, 1].scatter(spam_dane['slowa'], spam_dane['dlugosc'], color='tomato',    alpha=0.5, s=10, label='Spam')
axes[0, 1].set_xlabel('Liczba slow')
axes[0, 1].set_ylabel('Dlugosc wiadomosci')
axes[0, 1].set_title('Scatter: dlugosc vs slowa')
axes[0, 1].legend()
axes[0, 1].grid(True)

# macierz pomylek - heatmapa
im = axes[1, 0].imshow(cm, cmap='Blues')
axes[1, 0].set_xticks([0, 1]); axes[1, 0].set_xticklabels(['Ham', 'Spam'])
axes[1, 0].set_yticks([0, 1]); axes[1, 0].set_yticklabels(['Ham', 'Spam'])
axes[1, 0].set_xlabel('Przewidywana klasa')
axes[1, 0].set_ylabel('Rzeczywista klasa')
axes[1, 0].set_title('Macierz pomylek')
for i in range(2):
    for j in range(2):
        kolor = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        axes[1, 0].text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=16, color=kolor)

# krzywa ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
axes[1, 1].plot(fpr, tpr, color='tomato', lw=2, label='AUC = ' + str(round(roc_auc, 3)))
axes[1, 1].plot([0, 1], [0, 1], color='gray', linestyle='--')
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].set_title('Krzywa ROC')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('projekt_gnb.png', dpi=150)
plt.show()

# histogram prawdopodobienstwa
plt.figure(figsize=(10, 5))
bins = np.linspace(0, 1, 40)
plt.hist(y_prob[y_test == 0], bins=bins, color='steelblue', alpha=0.7, label='Ham')
plt.hist(y_prob[y_test == 1], bins=bins, color='tomato',    alpha=0.7, label='Spam')
plt.axvline(0.5, color='black', linestyle='--', label='Prog decyzyjny 0.5')
plt.xlabel('Prawdopodobienstwo spamu')
plt.ylabel('Liczba wiadomosci')
plt.title('Rozklad prawdopodobienstwa - GNB')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('histogram_prob.png', dpi=150)
plt.show()

# wnioski
print("=" * 50)
print("WNIOSKI")
print("=" * 50)
print("Zbior jest niezbalansowany: 86.6% ham, 13.4% spam")
print("Accuracy:", round(acc * 100, 1), "%")
print("Precision spam:", round(tp / (tp + fp) * 100, 1), "%")
print("Recall spam:", round(tp / (tp + fn) * 100, 1), "%")
print("AUC-ROC:", round(roc_auc, 3))
print("Najsilniejsza cecha: cyfry (numery tel. w spamie)")
print("FP (ham jako spam):", fp)
print("FN (spam przeoczony):", fn)

# =======================================================
# EKSPERYMENT 2 - TF-IDF + ComplementNB
# =======================================================

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import balanced_accuracy_score, average_precision_score

print("\n" + "=" * 50)
print("EKSPERYMENT 2 - TF-IDF + ComplementNB")
print("=" * 50)

# tutaj bierzemy surowy tekst wiadomości, a nie nasze ręczne cechy
X_text = dane['wiadomosc'].values
y_text = (dane['etykieta'] == 'spam').astype(int).values

# dzielimy dane na treningowe i testowe
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X_text,
    y_text,
    test_size=0.2,
    random_state=42,
    stratify=y_text
)

# tworzymy model tekstowy
model_tfidf = Pipeline([
    ('tfidf', TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2)),
    ('clf', ComplementNB())
])

# uczenie modelu
model_tfidf.fit(X_train_t, y_train_t)

# predykcja
y_pred_t = model_tfidf.predict(X_test_t)
y_prob_t = model_tfidf.predict_proba(X_test_t)[:, 1]

# wyniki
acc_t = accuracy_score(y_test_t, y_pred_t)
bal_acc_t = balanced_accuracy_score(y_test_t, y_pred_t)
pr_auc_t = average_precision_score(y_test_t, y_prob_t)

print("Accuracy:", round(acc_t, 4))
print("Balanced accuracy:", round(bal_acc_t, 4))
print("PR-AUC:", round(pr_auc_t, 4))
print(classification_report(y_test_t, y_pred_t, target_names=['Ham', 'Spam']))

# macierz pomyłek dla eksperymentu 2
cm_t = confusion_matrix(y_test_t, y_pred_t)
tn_t, fp_t, fn_t, tp_t = cm_t.ravel()

print("Macierz pomylek - eksperyment 2:")
print(cm_t)
print("TN:", tn_t, " FP:", fp_t, " FN:", fn_t, " TP:", tp_t)

# wykres macierzy pomyłek dla eksperymentu 2
plt.figure(figsize=(6, 5))
plt.imshow(cm_t, cmap='Greens')
plt.xticks([0, 1], ['Ham', 'Spam'])
plt.yticks([0, 1], ['Ham', 'Spam'])
plt.xlabel('Przewidywana klasa')
plt.ylabel('Rzeczywista klasa')
plt.title('Macierz pomylek - TF-IDF + ComplementNB')

for i in range(2):
    for j in range(2):
        kolor = 'white' if cm_t[i, j] > cm_t.max() / 2 else 'black'
        plt.text(j, i, str(cm_t[i, j]), ha='center', va='center', fontsize=16, color=kolor)

plt.tight_layout()
plt.savefig('macierz_tfidf.png', dpi=150)
plt.show()

# krótkie wnioski z eksperymentu 2
print("=" * 50)
print("WNIOSKI - EKSPERYMENT 2")
print("=" * 50)
print("Drugi eksperyment wykorzystuje tresc wiadomosci, a nie tylko reczne cechy.")
print("TF-IDF zamienia tekst na liczby, pokazujac ktore slowa sa wazne dla klasyfikacji.")
print("ComplementNB jest dobrym modelem do klasyfikacji tekstu i danych niezbalansowanych.")
print("Precision spam:", round(tp_t / (tp_t + fp_t) * 100, 1), "%")
print("Recall spam:", round(tp_t / (tp_t + fn_t) * 100, 1), "%")
print("PR-AUC:", round(pr_auc_t, 3))

# =======================================================
# EKSPERYMENT 3 - KNN + StandardScaler + reczne cechy
# =======================================================

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

print("\n" + "=" * 50)
print("EKSPERYMENT 3 - KNN + StandardScaler + reczne cechy")
print("=" * 50)

# KNN korzysta z tych samych recznych cech co eksperyment 1:
# dlugosc, slowa, wielkie, cyfry, wykrzykniki

X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# KNN jest wrazliwy na skale cech, dlatego normalizujemy dane
scaler = StandardScaler()

X_train_k_scaled = scaler.fit_transform(X_train_k)
X_test_k_scaled = scaler.transform(X_test_k)

# tworzymy model KNN
model_knn = KNeighborsClassifier(n_neighbors=5)

# uczenie modelu
model_knn.fit(X_train_k_scaled, y_train_k)

# predykcja
y_pred_k = model_knn.predict(X_test_k_scaled)
y_prob_k = model_knn.predict_proba(X_test_k_scaled)[:, 1]

# wyniki
acc_k = accuracy_score(y_test_k, y_pred_k)
bal_acc_k = balanced_accuracy_score(y_test_k, y_pred_k)
pr_auc_k = average_precision_score(y_test_k, y_prob_k)

print("Accuracy:", round(acc_k, 4))
print("Balanced accuracy:", round(bal_acc_k, 4))
print("PR-AUC:", round(pr_auc_k, 4))
print(classification_report(y_test_k, y_pred_k, target_names=['Ham', 'Spam']))

# macierz pomylek dla eksperymentu 3
cm_k = confusion_matrix(y_test_k, y_pred_k)
tn_k, fp_k, fn_k, tp_k = cm_k.ravel()

print("Macierz pomylek - eksperyment 3:")
print(cm_k)
print("TN:", tn_k, " FP:", fp_k, " FN:", fn_k, " TP:", tp_k)

# wykres macierzy pomylek dla KNN
plt.figure(figsize=(6, 5))
plt.imshow(cm_k, cmap='Purples')
plt.xticks([0, 1], ['Ham', 'Spam'])
plt.yticks([0, 1], ['Ham', 'Spam'])
plt.xlabel('Przewidywana klasa')
plt.ylabel('Rzeczywista klasa')
plt.title('Macierz pomylek - KNN + StandardScaler')

for i in range(2):
    for j in range(2):
        kolor = 'white' if cm_k[i, j] > cm_k.max() / 2 else 'black'
        plt.text(j, i, str(cm_k[i, j]), ha='center', va='center', fontsize=16, color=kolor)

plt.tight_layout()
plt.savefig('macierz_knn.png', dpi=150)
plt.show()

print("=" * 50)
print("WNIOSKI - EKSPERYMENT 3")
print("=" * 50)
print("Trzeci eksperyment wykorzystuje metode KNN omawiana na laboratorium.")
print("KNN klasyfikuje wiadomosc na podstawie najblizszych podobnych przykladow.")
print("Zastosowano StandardScaler, poniewaz KNN jest wrazliwy na skale cech.")
print("Precision spam:", round(tp_k / (tp_k + fp_k) * 100, 1), "%")
print("Recall spam:", round(tp_k / (tp_k + fn_k) * 100, 1), "%")
print("PR-AUC:", round(pr_auc_k, 3))

# =======================================================
# RZECZ 2 - WALIDACJA KRZYZOWA
# =======================================================

from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

print("\n" + "=" * 50)
print("WALIDACJA KRZYZOWA")
print("=" * 50)

# 5 czesci danych, powtorzone 5 razy
# razem daje 25 testow modelu
cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=42
)

# metryki, ktore bedziemy sprawdzac
metryki = {
    'accuracy': 'accuracy',
    'balanced_accuracy': 'balanced_accuracy',
    'precision_spam': 'precision',
    'recall_spam': 'recall',
    'f1_spam': 'f1',
    'roc_auc': 'roc_auc'
}

# -------------------------------------------------------
# Walidacja krzyzowa dla eksperymentu 1
# reczne cechy + GaussianNB
# -------------------------------------------------------

print("\nEKSPERYMENT 1 - GaussianNB + reczne cechy")

wyniki_gnb = cross_validate(
    GaussianNB(),
    X,
    y,
    cv=cv,
    scoring=metryki
)

for nazwa in metryki:
    wyniki = wyniki_gnb['test_' + nazwa]
    print(nazwa, ":", round(wyniki.mean(), 4), "+/-", round(wyniki.std(), 4))

# -------------------------------------------------------
# Walidacja krzyzowa dla eksperymentu 2
# TF-IDF + ComplementNB
# -------------------------------------------------------

print("\nEKSPERYMENT 2 - TF-IDF + ComplementNB")

model_tfidf_cv = Pipeline([
    ('tfidf', TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2)),
    ('clf', ComplementNB())
])

wyniki_tfidf = cross_validate(
    model_tfidf_cv,
    X_text,
    y_text,
    cv=cv,
    scoring=metryki
)

for nazwa in metryki:
    wyniki = wyniki_tfidf['test_' + nazwa]
    print(nazwa, ":", round(wyniki.mean(), 4), "+/-", round(wyniki.std(), 4))

# -------------------------------------------------------
# Walidacja krzyzowa dla eksperymentu 3
# KNN + StandardScaler + reczne cechy
# -------------------------------------------------------

print("\nEKSPERYMENT 3 - KNN + StandardScaler + reczne cechy")

model_knn_cv = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', KNeighborsClassifier(n_neighbors=5))
])

wyniki_knn = cross_validate(
    model_knn_cv,
    X,
    y,
    cv=cv,
    scoring=metryki
)

for nazwa in metryki:
    wyniki = wyniki_knn['test_' + nazwa]
    print(nazwa, ":", round(wyniki.mean(), 4), "+/-", round(wyniki.std(), 4))

# =======================================================
# RZECZ 3 - ANALIZA STATYSTYCZNA WYNIKOW
# =======================================================

from scipy.stats import wilcoxon

print("\n" + "=" * 50)
print("ANALIZA STATYSTYCZNA - TEST WILCOXONA")
print("=" * 50)

# bierzemy wyniki F1 dla klasy spam z walidacji krzyzowej
f1_gnb = wyniki_gnb['test_f1_spam']
f1_tfidf = wyniki_tfidf['test_f1_spam']
f1_knn = wyniki_knn['test_f1_spam']

# -------------------------------------------------------
# Porownanie 1: GNB vs TF-IDF
# -------------------------------------------------------

stat, p_value = wilcoxon(f1_gnb, f1_tfidf)

print("\nPorownanie 1: GaussianNB vs TF-IDF + ComplementNB")
print("Porownywana metryka: F1-score dla klasy Spam")
print("Sredni F1 - GaussianNB + reczne cechy:", round(f1_gnb.mean(), 4))
print("Sredni F1 - TF-IDF + ComplementNB:", round(f1_tfidf.mean(), 4))
print("Statystyka testu:", round(stat, 4))
print("p-value:", p_value)

if p_value < 0.05:
    print("Wniosek: roznica miedzy GNB i TF-IDF jest statystycznie istotna.")
else:
    print("Wniosek: brak podstaw do stwierdzenia istotnej roznicy miedzy GNB i TF-IDF.")

# -------------------------------------------------------
# Porownanie 2: KNN vs TF-IDF
# -------------------------------------------------------

stat_knn, p_value_knn = wilcoxon(f1_knn, f1_tfidf)

print("\nPorownanie 2: KNN vs TF-IDF + ComplementNB")
print("Porownywana metryka: F1-score dla klasy Spam")
print("Sredni F1 - KNN + StandardScaler:", round(f1_knn.mean(), 4))
print("Sredni F1 - TF-IDF + ComplementNB:", round(f1_tfidf.mean(), 4))
print("Statystyka testu:", round(stat_knn, 4))
print("p-value:", p_value_knn)

if p_value_knn < 0.05:
    print("Wniosek: roznica miedzy KNN i TF-IDF jest statystycznie istotna.")
else:
    print("Wniosek: brak podstaw do stwierdzenia istotnej roznicy miedzy KNN i TF-IDF.")

# -------------------------------------------------------
# Porownanie 3: GNB vs KNN
# -------------------------------------------------------

stat_gnb_knn, p_value_gnb_knn = wilcoxon(f1_gnb, f1_knn)

print("\nPorownanie 3: GaussianNB vs KNN")
print("Porownywana metryka: F1-score dla klasy Spam")
print("Sredni F1 - GaussianNB + reczne cechy:", round(f1_gnb.mean(), 4))
print("Sredni F1 - KNN + StandardScaler:", round(f1_knn.mean(), 4))
print("Statystyka testu:", round(stat_gnb_knn, 4))
print("p-value:", p_value_gnb_knn)

if p_value_gnb_knn < 0.05:
    print("Wniosek: roznica miedzy GNB i KNN jest statystycznie istotna.")
else:
    print("Wniosek: brak podstaw do stwierdzenia istotnej roznicy miedzy GNB i KNN.")

# -------------------------------------------------------
# Wykres porownujacy 3 modele
# -------------------------------------------------------

plt.figure(figsize=(9, 5))

dane_wykres = [f1_gnb, f1_knn, f1_tfidf]
etykiety = [
    'GaussianNB\nreczne cechy',
    'KNN\nStandardScaler',
    'TF-IDF\nComplementNB'
]

plt.boxplot(dane_wykres, tick_labels=etykiety)
plt.ylabel('F1-score dla klasy Spam')
plt.title('Porownanie wynikow F1 w walidacji krzyzowej')
plt.grid(True)

plt.tight_layout()
plt.savefig('porownanie_f1_3modele_boxplot.png', dpi=150)
plt.show()

print("=" * 50)
print("KONCOWY WNIOSEK")
print("=" * 50)
print("W projekcie porownano trzy metody rozpoznawania spamu.")
print("GaussianNB i KNN korzystaja z recznych cech liczbowych.")
print("TF-IDF + ComplementNB korzysta bezposrednio z tresci wiadomosci.")
print("Najlepszy model wybieramy na podstawie sredniego F1-score dla klasy Spam oraz testu Wilcoxona.")

print("DOTARLEM DO SAMEGO KONCA PLIKU")