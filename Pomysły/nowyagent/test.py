import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

def load_and_prepare_data(data_path, target_path, hours_to_predict):
    """ Wczytaj dane z plików CSV i przygotuj je do analizy """
    # Wczytywanie danych
    df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)
    df_target = pd.read_csv(target_path, index_col='timestamp', parse_dates=True)
    
    # Sprawdzenie długości danych
    print(f"Długość df: {len(df)}")
    print(f"Długość df_target: {len(df_target)}")
    
    # Ustalamy minimalną liczbę godzin do prognozy, aby uniknąć błędów
    if hours_to_predict > len(df_target):
        raise ValueError(f"Liczba godzin do prognozy ({hours_to_predict}) jest większa niż liczba próbek w df_target ({len(df_target)}).")
    
    # Tworzenie próbki wejściowej i wyjściowej
    X = []
    y = []

    for i in range(len(df) - hours_to_predict):
        if i + hours_to_predict < len(df_target):
            sample = df.iloc[i:i+hours_to_predict].values  # Pobranie próbek z df
            X.append(sample)
            target_value = df_target.iloc[i+hours_to_predict].values  # Pobranie wartości z df_target
            y.append(target_value)
        else:
            print(f"Brak danych dla indeksu: {i}")

    # Konwersja do numpy array
    X = np.array(X)
    y = np.array(y)
    
    # Sprawdzenie długości danych
    print(f"Długość X: {len(X)}")
    print(f"Długość y: {len(y)}")

    # Upewnij się, że liczba próbek w X i y jest zgodna
    if len(X) != len(y):
        raise ValueError(f"Liczba próbek w X ({len(X)}) i y ({len(y)}) nie jest zgodna.")
    
    return X, y

def train_and_evaluate_model(X, y):
    """ Trenowanie modelu i ocena wyników """
    # Sprawdzenie rozmiaru danych
    print(f"Długość X przed podziałem: {len(X)}")
    print(f"Długość y przed podziałem: {len(y)}")

    if len(X) == 0 or len(y) == 0:
        raise ValueError("Brak danych do trenowania modelu.")
    
    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Sprawdzenie rozmiaru danych po podziale
    print(f"Długość X_train: {len(X_train)}")
    print(f"Długość X_test: {len(X_test)}")
    print(f"Długość y_train: {len(y_train)}")
    print(f"Długość y_test: {len(y_test)}")
    
    # Model regresji liniowej
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Przewidywanie i ewaluacja modelu
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f'MSE: {mse}')
    
    # Zapisanie modelu do pliku
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model

def load_model(model_path):
    """ Wczytaj wytrenowany model z pliku """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def main():
    # Ścieżki do plików CSV
    data_path = 'btc_usdt_1000h.csv'
    target_path = 'btc_usdt_next_100h.csv'
    
    # Liczba godzin do prognozy - zmniejszona do minimalnej wartości
    hours_to_predict = 10  # Ustaw wartość odpowiednią do długości df_target
    
    # Wczytaj i przygotuj dane
    X, y = load_and_prepare_data(data_path, target_path, hours_to_predict)
    
    # Trening i ewaluacja modelu
    model = train_and_evaluate_model(X, y)
    
    # Możesz również załadować model w późniejszym czasie
    # model = load_model('model.pkl')

if __name__ == "__main__":
    main()
