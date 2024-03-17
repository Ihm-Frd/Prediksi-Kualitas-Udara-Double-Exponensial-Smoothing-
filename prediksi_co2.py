import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

def cover_page():
    st.markdown('<h1 style="font-size: 38px;">Forecasting Kualitas Udara Menggunakan Algoritma Double Exponential Smoothing</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="font-size: 1.2em;">Dibuat Oleh : Nama lu bre </h2>', unsafe_allow_html=True)

    
    st.markdown('<h2 style="font-size: 1.5em;">Deskripsi Aplikasi</h2>', unsafe_allow_html=True)
    st.write("Aplikasi ini dirancang untuk melakukan forecasting atau peramalan sederhana menggunakan metode **Double Exponential Smoothing**. Anda dapat dengan mudah memprediksi nilai berdasarkan data historis dengan menggunakan algoritma peramalan yang efektif.")
    
    st.markdown('<h2 style="font-size: 1.5em;"> Metode Double Exponential Smoothing</h2>', unsafe_allow_html=True)
    st.write("Double Exponential Smoothing adalah metode peramalan yang digunakan untuk dataset yang menunjukkan tren serta pola musiman. Metode ini menggabungkan dua komponen, yaitu level (tingkat) dan trend (kecenderungan), untuk memperkirakan nilai di masa depan. Dengan memahami metode ini, Anda dapat membuat prediksi yang lebih baik dan akurat.")



# Fungsi untuk menghitung skor presentase prediksi akurasi model
def calculate_accuracy_percentage(df, model, year):
    last_date = df.index.max()
    start_date = last_date - pd.DateOffset(years=year)
    actual_values = df['CO2'].loc[df.index >= start_date]

    # Melakukan prediksi menggunakan model
    pred = model.forecast(len(actual_values))  # Menyesuaikan panjang hasil prediksi

    # Pastikan kedua variabel memiliki jumlah data yang sama
    if len(pred) != len(actual_values):
        # Lakukan penyesuaian untuk memastikan jumlah baris sama
        if len(pred) > len(actual_values):
            pred = pred[:len(actual_values)]
        else:
            actual_values = actual_values[:len(pred)]

    rmse = np.sqrt(mean_squared_error(actual_values, pred))
    accuracy_percentage = 100 * (1 - rmse / actual_values.mean())
    return accuracy_percentage

# Fungsi untuk halaman aplikasi prediksi
def app_page():
    model = pickle.load(open('prediksi_co2.sav', 'rb'))

    df = pd.read_excel("CO2 dataset.xlsx")
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    df.set_index(['Year'], inplace=True)

    st.title('Forecasting Kualitas Udara')
    year = st.slider("Tentukan Tahun", 1, 30, step=1)

    if st.button("Prediksikan"):
        st.markdown("<div style='font-size: larger;' class='highlight'>Hasil Prediksi:</div>", unsafe_allow_html=True)

        # Menghitung dan menampilkan skor presentase prediksi akurasi model
        accuracy_percentage = calculate_accuracy_percentage(df, model, year)
        # st.subheader("Evaluasi Model")
        st.write(f"Skor Presentase Prediksi Akurasi: {accuracy_percentage:.2f}%")

        # Melakukan prediksi menggunakan model
        pred = model.forecast(year)
        pred = pd.DataFrame(pred, columns=['CO2'])

        col1, col2 = st.columns([2, 3])

        with col1:
            st.dataframe(pred)

        with col2:
            fig, ax = plt.subplots()
            df['CO2'].plot(style='--', color='gray', legend=True, label='Data Yang DiKetahui')
            pred['CO2'].plot(color='b', legend=True, label='Hasil Prediksi')
            plt.title('Grafik Prediksi CO2')
            plt.xlabel('Tahun')
            plt.ylabel('CO2')

            # Simpan plot sebagai file gambar
            temp_file = "temp_plot.png"
            plt.savefig(temp_file)

            # Tampilkan plot sebagai gambar menggunakan st.image()
            st.image(temp_file)

            # Hapus file gambar setelah ditampilkan
            import os
            os.remove(temp_file)

# Fungsi utama untuk memilih halaman
def main():
    page = st.sidebar.radio("Pilih Halaman", ("Halaman Utama", "Aplikasi"))

    if page == "Halaman Utama":
        cover_page()
    elif page == "Aplikasi":
        app_page()

if __name__ == "__main__":
    main()
