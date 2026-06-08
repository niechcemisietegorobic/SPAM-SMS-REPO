# Dodatkowe wskazówki prowadzącego — co dodano

Po konsultacji z prowadzącym projekt rozszerzono o trzy elementy.

## 1. RandomOverSampler i SMOTE

Zbiór SMS Spam Collection jest niezbalansowany:

- ham: 86.6%,
- spam: 13.4%.

Dlatego dodano sprawdzenie resamplingu.

**RandomOverSampler** powiela próbki klasy mniejszościowej, czyli spamu.

**SMOTE** tworzy nowe sztuczne próbki klasy mniejszościowej na podstawie istniejących przykładów w przestrzeni cech.

Celem nie było koniecznie poprawienie wyniku, tylko sprawdzenie, jak wyrównanie klas wpływa na precision, recall, F1-score i balanced accuracy.

## 2. Regex do parsowania starych cech

Nie dodano nowych cech typu link, e-mail albo telefon.

Zostawiono stare cechy:

- długość wiadomości,
- liczba słów,
- liczba wielkich liter,
- liczba cyfr,
- liczba wykrzykników.

Zmieniono tylko sposób liczenia części z nich na regexy, żeby łatwiej było wytłumaczyć, jak wiadomość jest parsowana.

Przykład:

```python
re.findall(r'\b\w+\b', tekst)  # słowa
re.findall(r'\d', tekst)        # cyfry
re.findall(r'!', tekst)         # wykrzykniki
```

## 3. Porównanie inputów dla GaussianNB

Dodano porównanie różnych zestawów wejściowych dla GaussianNB:

- tylko długość,
- tylko wykrzykniki,
- tylko cyfry,
- długość + słowa,
- cyfry + wykrzykniki,
- wszystkie 5 cech.

Dzięki temu widać, czy modelowi wystarczy jedna cecha, czy potrzebuje większego zestawu informacji.

## Najważniejszy wniosek

Gorszy wynik w niektórych wariantach nie jest błędem. To normalne, bo eksperyment ma pokazać wpływ różnych cech i metod przygotowania danych.
