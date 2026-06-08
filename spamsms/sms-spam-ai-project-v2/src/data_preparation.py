import re
import pandas as pd


FEATURE_COLUMNS = ['dlugosc', 'slowa', 'wielkie', 'cyfry', 'wykrzykniki']


def load_sms_data(path='data/spam.csv'):
    """Wczytuje SMS Spam Collection i zostawia tylko etykietę oraz treść wiadomości."""
    dane = pd.read_csv(path, encoding='latin-1')
    dane = dane[['v1', 'v2']].rename(columns={'v1': 'etykieta', 'v2': 'wiadomosc'})
    return dane


# =======================================================
# Parsowanie wiadomosci za pomoca regexow
# =======================================================
# Nie dodajemy tutaj nowych cech typu link/e-mail/telefon.
# Regex sluzy tylko do czytelnego policzenia tych samych cech,
# ktore byly w projekcie od poczatku: slowa, wielkie litery,
# cyfry i wykrzykniki.


def policz_slowa_regex(tekst):
    """Liczy slowa w SMS-ie za pomoca wyrazenia regularnego."""
    return len(re.findall(r'\b\w+\b', str(tekst)))


def policz_wielkie_regex(tekst):
    """Liczy wielkie litery w SMS-ie."""
    return len(re.findall(r'[A-Z]', str(tekst)))


def policz_cyfry_regex(tekst):
    """Liczy cyfry w SMS-ie."""
    return len(re.findall(r'\d', str(tekst)))


def policz_wykrzykniki_regex(tekst):
    """Liczy wykrzykniki w SMS-ie."""
    return len(re.findall(r'!', str(tekst)))


def add_manual_features(dane):
    """Dodaje stare ręczne cechy, ale liczone w czytelniejszy sposob regexem."""
    dane = dane.copy()
    dane['dlugosc'] = dane['wiadomosc'].apply(lambda x: len(str(x)))
    dane['slowa'] = dane['wiadomosc'].apply(policz_slowa_regex)
    dane['wielkie'] = dane['wiadomosc'].apply(policz_wielkie_regex)
    dane['cyfry'] = dane['wiadomosc'].apply(policz_cyfry_regex)
    dane['wykrzykniki'] = dane['wiadomosc'].apply(policz_wykrzykniki_regex)
    return dane


def get_manual_xy(dane):
    """Zwraca X/y dla modeli opartych o ręczne cechy."""
    X = dane[FEATURE_COLUMNS].values
    y = (dane['etykieta'] == 'spam').astype(int).values
    return X, y


def get_text_xy(dane):
    """Zwraca X/y dla modeli opartych o treść wiadomości."""
    X_text = dane['wiadomosc'].values
    y_text = (dane['etykieta'] == 'spam').astype(int).values
    return X_text, y_text


def dataset_summary(dane):
    """Zwraca podstawowe liczności klas."""
    n_ham = int((dane['etykieta'] == 'ham').sum())
    n_spam = int((dane['etykieta'] == 'spam').sum())
    total = int(len(dane))
    return {
        'total': total,
        'ham': n_ham,
        'spam': n_spam,
        'ham_percent': round(n_ham / total * 100, 1),
        'spam_percent': round(n_spam / total * 100, 1),
    }
