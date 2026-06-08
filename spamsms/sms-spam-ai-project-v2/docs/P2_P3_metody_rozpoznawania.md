# P2/P3 — Implementacja metod rozpoznawania

W tej części projektu zaimplementowano trzy metody rozpoznawania wiadomości SMS.

## Metoda 1 — GaussianNB + ręczne cechy

Pierwsza metoda wykorzystuje ręczne cechy liczbowe. Po poprawkach część z nich jest liczona przez regex, ale dalej są to te same stare cechy:

- długość wiadomości,
- liczba słów,
- liczba wielkich liter,
- liczba cyfr,
- liczba wykrzykników.

Model: `GaussianNB`.

Ta metoda sprawdza bardziej **strukturę wiadomości**. Regex nie dodaje nowych cech, tylko czytelniej liczy słowa, wielkie litery, cyfry i wykrzykniki.

Najważniejszy fragment kodu:

```python
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## Metoda 2 — KNN + StandardScaler

Druga metoda wykorzystuje klasyfikator `KNeighborsClassifier`, czyli KNN.

KNN był omawiany na laboratorium jako metoda k najbliższych sąsiadów. Model sprawdza, do jakich znanych przykładów najbardziej podobna jest nowa wiadomość.

Przed KNN zastosowano `StandardScaler`, ponieważ KNN działa na odległościach i jest wrażliwy na skalę cech.

Najważniejszy fragment kodu:

```python
model_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', KNeighborsClassifier(n_neighbors=5))
])
model_knn.fit(X_train, y_train)
y_pred = model_knn.predict(X_test)
```

## Metoda 3 — TF-IDF + ComplementNB

Trzecia metoda wykorzystuje bezpośrednio treść wiadomości.

`TfidfVectorizer` zamienia tekst na liczby, a `ComplementNB` klasyfikuje wiadomość jako spam albo ham.

Najważniejszy fragment kodu:

```python
model_tfidf = Pipeline([
    ('tfidf', TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2)),
    ('clf', ComplementNB())
])
model_tfidf.fit(X_train, y_train)
y_pred = model_tfidf.predict(X_test)
```

## Najprostsze porównanie

```text
GaussianNB — sprawdza strukturę wiadomości.
KNN — sprawdza podobieństwo do innych wiadomości.
TF-IDF + ComplementNB — sprawdza treść wiadomości.
```
