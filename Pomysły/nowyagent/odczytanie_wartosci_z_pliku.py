import pandas as pd

def load_and_print_data(data_path, target_path):
    """ Wczytaj dane z plików CSV i wypisz wszystkie wiersze """
    # Wczytywanie danych
    df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)
    df_target = pd.read_csv(target_path, index_col='timestamp', parse_dates=True)
    
    # Sprawdzenie wczytanych danych
    print(f"Dane z pliku {data_path}:")
    print(df.to_string())  # Wypisz wszystkie wiersze
    print("\nDane z pliku {target_path}:")
    print(df_target.to_string())  # Wypisz wszystkie wiersze

    # Wypisz przykładowe wartości z konkretnego timestamp
    sample_timestamp = df.index[0]  # Przykładowy timestamp (pierwszy wiersz)
    print(f"\nPrzykładowe dane dla timestamp {sample_timestamp}:")
    sample_data = df.loc[sample_timestamp]
    print(sample_data)

# Wywołanie funkcji
load_and_print_data('btc_usdt_1000h.csv', 'btc_usdt_next_100h.csv')
