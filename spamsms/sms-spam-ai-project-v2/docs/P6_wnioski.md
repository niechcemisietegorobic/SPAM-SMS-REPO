# P6 — Wnioski

Najważniejsze wnioski z projektu:

1. Zbiór SMS Spam Collection jest niezbalansowany.
2. Sama accuracy nie wystarcza do oceny modelu.
3. GaussianNB jest dobrym prostym modelem bazowym.
4. KNN działa dobrze po normalizacji cech za pomocą StandardScaler.
5. TF-IDF + ComplementNB najlepiej wykorzystuje treść wiadomości.
6. Regex ułatwił wytłumaczenie parsowania starych cech.
7. Porównanie inputów dla GaussianNB pokazało, że pojedyncze cechy nie zawsze wystarczają.
8. RandomOverSampler i SMOTE pozwalają sprawdzić wpływ wyrównania klas.
9. Gorszy wynik w części eksperymentów nie jest błędem — celem było porównanie wpływu różnych metod.

Model końcowy według średniego F1-score dla klasy spam: **TF-IDF + ComplementNB**.
