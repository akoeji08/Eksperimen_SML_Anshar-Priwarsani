import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(input_path: str, output_path: str) -> pd.DataFrame:
    print(f"[INFO] Memuat dataset dari: {input_path}")
    df = pd.read_csv(input_path)
    print(f"[INFO] Ukuran awal dataset: {df.shape}")

    # 1. Menghapus kolom identitas yang tidak relevan untuk pemodelan
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # 2. Menangani Missing Values (Kolom TotalCharges memiliki spasi kosong)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    missing_count = df['TotalCharges'].isnull().sum()
    print(f"[INFO] Jumlah missing values pada TotalCharges: {missing_count}")
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # 3. Menghapus Data Duplikat
    dup_count = df.duplicated().sum()
    print(f"[INFO] Jumlah data duplikat yang dihapus: {dup_count}")
    df = df.drop_duplicates().reset_index(drop=True)

    # 4. Deteksi dan Penanganan Outlier (Metode IQR Capping pada fitur numerik)
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower_bound, upper_bound)

    # 5. Encoding Data Kategorikal
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # 6. Standarisasi Fitur Numerik
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Simpan dataset yang sudah bersih dan siap dilatih
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Preprocessing selesai! Ukuran akhir: {df.shape}")
    print(f"[INFO] Dataset tersimpan di: {output_path}")
    return df

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_file = os.path.join(base_dir, "telco_churn_raw", "telco_churn.csv")
    clean_file = os.path.join(base_dir, "preprocessing", "telco_churn_preprocessing", "telco_churn_clean.csv")
    preprocess_data(raw_file, clean_file)
