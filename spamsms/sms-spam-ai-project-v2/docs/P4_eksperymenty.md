# P4 — Implementacja eksperymentów

W P4 nie chodzi już tylko o samo napisanie klasyfikatorów, ale o sprawdzenie ich w uporządkowany sposób.

W projekcie wykonano główne eksperymenty:

1. **GaussianNB + ręczne cechy**  
   Model bazowy, który sprawdza, czy spam da się rozpoznawać po prostych cechach liczbowych SMS-a.

2. **KNN + StandardScaler + ręczne cechy**  
   Model z laboratoriów. Sprawdza podobieństwo wiadomości do przykładów ze zbioru treningowego.

3. **TF-IDF + ComplementNB**  
   Model tekstowy. Analizuje słowa występujące w wiadomościach.

Dodatkowo po wskazówkach prowadzącego dodano:

4. **GaussianNB na różnych inputach**  
   Porównano, czy lepiej działa jedna cecha, para cech, czy wszystkie cechy razem.

5. **GaussianNB z RandomOverSampler i SMOTE**  
   Sprawdzono wpływ resamplingu na dane niezbalansowane.

## Walidacja krzyżowa

Do głównego porównania modeli użyto:

```python
RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
```

Czyli każdy model został oceniony 25 razy.

## Dlaczego tak?

Zbiór jest niezbalansowany, więc jeden podział train/test mógłby dać przypadkowo lepszy albo gorszy wynik. Walidacja krzyżowa pozwala sprawdzić, czy wynik jest stabilny.

## Ważne

Jeśli jakiś wariant daje gorszy wynik, to nie jest problem. Celem P4 jest porównanie różnych ustawień i sprawdzenie, co pomaga, a co nie.
