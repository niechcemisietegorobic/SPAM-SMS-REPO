# Skrypt do prezentacji — SMS Spam

## 1. Cel projektu

Celem projektu było stworzenie systemu rozpoznającego spam w wiadomościach SMS. Jest to problem klasyfikacji binarnej: wiadomość może być normalna, czyli ham, albo niechciana, czyli spam.

## 2. Zbiór danych

Zbiór zawiera 5572 wiadomości. Wiadomości ham stanowią 86.6%, a spam 13.4%. Zbiór jest więc niezbalansowany, dlatego poza accuracy ważne są precision, recall i F1-score dla klasy spam.

## 3. Przygotowanie cech i regex

Z każdej wiadomości wyciągnięto pięć starych cech: długość, liczbę słów, wielkich liter, cyfr i wykrzykników. Po uwagach prowadzącego sposób liczenia części cech zapisano przy pomocy regexów. Nie dodano nowych typów cech — regex tylko ułatwia pokazanie, jak wiadomość jest parsowana.

## 4. Modele

Porównano trzy modele. GaussianNB jest prostym modelem bazowym na ręcznych cechach. KNN sprawdza podobieństwo wiadomości do przykładów treningowych i dlatego używa StandardScaler. TF-IDF + ComplementNB analizuje treść wiadomości, czyli słowa.

## 5. Porównanie inputów dla GaussianNB

Dodano eksperyment, który sprawdza różne wejścia dla GaussianNB: jedną cechę, parę cech i wszystkie pięć cech. Celem było sprawdzenie, czy jedna cecha wystarczy, czy lepszy jest pełny zestaw cech.

## 6. RandomOverSampler i SMOTE

Dodano też resampling. RandomOverSampler powiela przykłady spamu, a SMOTE tworzy sztuczne przykłady klasy mniejszościowej w przestrzeni cech. Wyniki po resamplingu nie muszą być lepsze — badamy wpływ tej metody na wyniki.

## 7. Walidacja i statystyka

Modele porównano w powtarzanej walidacji krzyżowej: 5 foldów i 5 powtórzeń, czyli 25 ocen każdego modelu. Dodatkowo wykonano test Wilcoxona na F1-score dla klasy spam.

## 8. Wnioski

Najlepszy średni F1-score uzyskał model TF-IDF + ComplementNB. Projekt pokazuje też, że wybór cech i sposób przygotowania danych mają duży wpływ na wynik. Gorszy wynik w części wariantów nie jest błędem, tylko elementem porównania.
