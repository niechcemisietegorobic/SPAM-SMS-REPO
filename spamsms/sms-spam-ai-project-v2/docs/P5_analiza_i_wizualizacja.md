# P5 — Analiza i wizualizacja wyników

W projekcie wyniki pokazano za pomocą metryk i wykresów.

## Metryki

Użyto:

- accuracy,
- balanced accuracy,
- precision dla klasy spam,
- recall dla klasy spam,
- F1-score dla klasy spam,
- ROC-AUC,
- PR-AUC.

Ponieważ zbiór jest niezbalansowany, najważniejsze są precision, recall i F1-score dla klasy spam.

## Wykresy

W projekcie znajdują się:

- `projekt_gnb.png` — rozkład klas, scatter, macierz pomyłek i ROC,
- `histogram_prob.png` — rozkład prawdopodobieństwa spamu dla GaussianNB,
- `macierz_knn.png` — macierz pomyłek KNN,
- `macierz_tfidf.png` — macierz pomyłek TF-IDF + ComplementNB,
- `porownanie_f1_3modele_boxplot.png` — porównanie F1-score trzech modeli,
- `gnb_porownanie_cech.png` — porównanie różnych inputów dla GaussianNB,
- `resampling_gnb_porownanie.png` — porównanie braku resamplingu, RandomOverSampler i SMOTE.

## Interpretacja

Wyniki nie muszą być najlepsze w każdym wariancie. Ważne jest porównanie, które cechy i metody pomagają, a które nie.
