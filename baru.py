import streamlit as st
import pandas as pd
import numpy as np
import holidays
from datetime import datetime
import sqlite3
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import openpyxl
import io
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import plotly.express as px
import re

# Koneksi ke database SQLite
def create_connection():
    conn = sqlite3.connect("forecasting_results.db")
    return conn

def create_table():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_name TEXT,
            month TEXT,
            sales REAL,
            quantity REAL,
            method TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Create the table when the application starts
create_table()

# Judul aplikasi 
st.title("📈 Aplikasi")

def process_date_column(data):
    if 'Tanggal' in data.columns:
        # Mengubah kolom Tanggal menjadi datetime
        data['Tanggal'] = pd.to_datetime(data['Tanggal'], errors='coerce')
        # Hapus entri yang tidak valid
        data = data.dropna(subset=['Tanggal'])
    return data

# Fungsi untuk mendapatkan tanggal event
def get_event_dates(year):
    id_holidays = holidays.Indonesia(years=year)
    event_dates = {
        'Idul Fitri': [],
        'Idul Adha': [],
        'Tahun Baru': [],
    }
    for date, name in id_holidays.items():
        if 'Idul Fitri' in name:
            event_dates['Idul Fitri'].append(date)
        elif 'Idul Adha' in name:
            event_dates['Idul Adha'].append(date)
        elif 'Tahun Baru' in name:
            event_dates['Tahun Baru'].append(date)
    return event_dates

# Mendapatkan event untuk tahun ini
current_year = datetime.now().year
event_dates = get_event_dates(current_year)

def add_event_column(data):
    year = data['Tanggal'].dt.year.unique()[0]
    event_dates = get_event_dates(year)
    data['Event'] = data['Tanggal'].apply(lambda x: any(x.date() in event for event in event_dates.values()))
    return data

def detect_sales_spikes(sales_data, threshold=1.5):
    mean_sales = sales_data.mean()
    std_sales = sales_data.std()
    upper_limit = mean_sales + (threshold * std_sales)
    spikes = sales_data[sales_data > upper_limit]
    return spikes

def select_forecasting_method(product_data, steps=3, method='ARIMA'):
    if isinstance(product_data, pd.DataFrame):
        product_data = product_data.squeeze()

    n = len(product_data)

    if n == 0:
        return [0] * steps, "No Data"

    if method == 'Naive':
        return [product_data.iloc[-1]] * steps, "Naive"

    elif method == 'Moving Average':
        if n < 3:
            return [product_data.mean()] * steps, "Average"
        forecast = product_data.rolling(window=3).mean().iloc[-1]
        return [forecast] * steps, "Moving Average"

    elif method == 'Exponential Smoothing':
        if n < 3:
            return [product_data.mean()] * steps, "Average"
        model = ExponentialSmoothing(product_data, trend='add', seasonal='add', seasonal_periods=12) 
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast, "Exponential Smoothing"

    elif method == 'Prophet':
        if n < 5:
            return [product_data.mean()] * steps, "Average"
        
        prophet_data = pd.DataFrame({
            'ds': product_data.index.to_timestamp(),
            'y': product_data.values
        })
        
        prophet_model = Prophet()
        prophet_model.add_seasonality(name='event', period=30, fourier_order=5)
        prophet_model.fit(prophet_data)
        future = prophet_model.make_future_dataframe(periods=steps, freq='M')
        forecast = prophet_model.predict(future)['yhat'].values[-steps:]
        return forecast, "Prophet"

    elif method in ['Random Forest', 'XGBoost']:
        if n < 5:
            return [product_data.mean()] * steps, "Average"
        
        df = pd.DataFrame(product_data)
        df['Month'] = np.arange(len(df))
        df['Event'] = df['Event'].astype(int)
        X = df[['Month', 'Event']]
        y = df['Penjualan']
        
        if method == 'Random Forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = XGBRegressor(n_estimators=100, random_state=42)
        
        model.fit(X, y)
        
        future_months = np.array([[len(df) + i, 1] for i in range(1, steps + 1)])
        forecast = model.predict(future_months)
        return forecast, method

    elif method == 'SARIMA':
        p, d, q = 1, 1, 1
        P, D, Q, s = 1, 1, 1, 12

        result = adfuller(product_data)
        is_stationary = result[1] <= 0.05

        if is_stationary:
            model = SARIMAX(product_data, order=(p, d, q), seasonal_order=(P, D, Q, s))
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=steps)
            return forecast, "SARIMA"
        else:
            return [product_data.mean()] * steps, "Average"

    else:
        result = adfuller(product_data)
        is_stationary = result[1] <= 0.05

        if is_stationary:
            model = ARIMA(product_data, order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=steps)
            return forecast, "ARIMA"
        else:
            return [product_data.mean()] * steps, "Average"

def display_forecast_with_events(forecast_df):
    if not forecast_df.empty:
        st.write("Hasil Peramalan Transaksi Produk dengan Event:")
        forecast_df['Penjualan'] = pd.to_numeric(forecast_df['Penjualan'], errors='coerce')
        forecast_df['Penjualan'] = forecast_df['Penjualan'].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else '0.00')
        forecast_df['Tanggal'] = pd.to_datetime(forecast_df['Tanggal'], errors='coerce')
        
        if forecast_df['Tanggal'].isnull().any():
            st.warning("Ada nilai yang tidak dapat dikonversi menjadi tanggal. Pastikan format tanggal benar.")
        
        forecast_df['Event'] = forecast_df['Event'].apply(lambda x: 'Ada Event' if x else 'Tidak Ada Event')
        st.dataframe(forecast_df)

def analyze_event_impact(monthly_data, event_dates):
    event_impact = {}
    
    for product in monthly_data['Nama Barang'].unique():
        product_data = monthly_data[monthly_data['Nama Barang'] == product]
        product_data['Bulan'] = product_data['Tanggal'].dt.to_period('M')
        
        for event_name, dates in event_dates.items():
            for event_date in dates:
                # Check if the event date is in the product's data
                if event_date in product_data['Tanggal'].values:
                    # Get the month before and after the event
                    event_month = event_date.to_period('M')
                    previous_month = event_month - 1
                    next_month = event_month + 1
                    
                    # Calculate sales for the previous and next month
                    previous_sales = product_data[product_data['Bulan'] == previous_month]['Penjualan'].sum()
                    event_sales = product_data[product_data['Bulan'] == event_month]['Penjualan'].sum()
                    next_sales = product_data[product_data['Bulan'] == next_month]['Penjualan'].sum()
                    
                    # Check if there was an increase
                    if event_sales > previous_sales:
                        if product not in event_impact:
                            event_impact[product] = {'increase': True, 'event_name': event_name}
                    if next_sales > event_sales:
                        if product not in event_impact:
                            event_impact[product] = {'increase': True, 'event_name': event_name}
    
    return event_impact

def chat(contexts, history, question):
    API_KEY = "AIzaSyAPUF_xOkqVUj7aWX_bXO_8cV6R9-xpQ4Y"  
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        api_key=API_KEY
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Anda adalah asisten yang sangat membantu dan berpengetahuan luas dalam bidang peramalan produk. Anda dapat menggunakan data yang diberikan untuk menjawab pertanyaan tentang penjualan, kuantitas, dan tren produk. Pastikan untuk memberikan jawaban yang jelas dan informatif, serta menyertakan analisis jika diperlukan.",
            ),
            ("human", "Berikut adalah data yang relevan: {contexts}\n\nGunakan riwayat chat ini untuk menghasilkan jawaban yang relevan berdasarkan percakapan terbaru: {history}\n\nPertanyaan pengguna: {question}\n\nSilakan berikan jawaban yang komprehensif dan jika perlu, sertakan rekomendasi atau wawasan tambahan berdasarkan data yang ada."),
        ]
    )
    
    chain = prompt | llm
    completion = chain.invoke(
        {
            "contexts": contexts,
            "history": history,
            "question": question,
        }
    )

    answer = completion.content
    input_tokens = completion.usage_metadata['input_tokens']
    completion_tokens = completion.usage_metadata['output_tokens']

    result = {}
    result["answer"] = answer
    result["input_tokens"] = input_tokens
    result["completion_tokens"] = completion_tokens
    
    return result

# Cek apakah pengguna sudah login
PASSWORD = "admin1234"  # Ganti dengan password yang diinginkan
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Form untuk memasukkan password
if not st.session_state.logged_in:
    password_input = st.text_input("Masukkan Password untuk Mengakses Aplikasi", type="password")

    if st.button("Login"):
        if password_input == PASSWORD:
            st.session_state.logged_in = True
            st.success("Login berhasil!")
        else:
            st.error("Password salah. Silakan coba lagi.")
else:
    # Inisialisasi DataFrame untuk menyimpan data
    if 'data' not in st.session_state:
        # Coba memuat data dari file CSV jika ada
        try:
            st.session_state.data = pd.read_csv("data_akuntansi.csv")
            st.success("Data berhasil dimuat dari file CSV.")
        except FileNotFoundError:
            st.session_state.data = pd.DataFrame()

    # Menu untuk memilih fungsi
    menu = st.sidebar.selectbox("Pilih Fungsi", [
        "Unggah Data",
        #"Akuntansi Utang",
        #"Akuntansi Piutang",
        "Penganggaran dan Peramalan",
        "Statistik"
    ])

    # Fungsi Unggah Data
    if menu == "Unggah Data":
        st.subheader("Unggah Data dari File")
        uploaded_file = st.file_uploader("Pilih file CSV atau Excel", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                new_data = pd.read_csv(uploaded_file)
            else:
                new_data = pd.read_excel(uploaded_file)
        
            # Validasi kolom
            required_columns = []  # Ganti dengan kolom yang diperlukan
            has_date_column = 'Tanggal' in new_data.columns
        
            if all(col in new_data.columns for col in required_columns):
                new_data = process_date_column(new_data)
                
                if not has_date_column:
                    st.warning("Kolom 'Tanggal' tidak ditemukan. Data akan tetap dimasukkan tanpa kolom 'Tanggal'.")
        
                # Menambahkan kolom event
                new_data = add_event_column(new_data)
        
                # Menggabungkan data yang diunggah dengan data yang sudah ada
                st.session_state.data = pd.concat([st.session_state.data, new_data], ignore_index=True)
                st.success("Data berhasil diunggah dan ditambahkan.")
    
    # Fungsi Akuntansi Utang
    #elif menu == "Akuntansi Utang":
        #st.subheader("Akuntansi Utang")
        #vendor = st.text_input("Pelanggan")
        #barang = st.text_input("Nama Barang")
        #amount = st.number_input("Jumlah Pembayaran", min_value=0.0)
        #quantity = st.number_input("Kuantitas", min_value=1)
        #due_date = st.date_input("Tanggal Jatuh Tempo", datetime.today())
        #city = st.text_input("Kota Pengiriman Pelanggan")
        
        #if st.button("Simpan Pembayaran"):
            #if vendor and barang and amount > 0 and quantity > 0:
                #new_row = pd.DataFrame({
                    #"Tipe Transaksi": ["Pengeluaran"],
                    #"Pelanggan": [vendor],
                    #"Nama Barang": [barang],
                    #"Penjualan": [-amount],
                    #"Kuantitas": [quantity],
                    #"Tanggal": [due_date],
                    #"Kota Pengiriman Pelanggan": [city]
                #})
                #st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
                #st.success(f"Pembayaran sebesar {amount} kepada {vendor} untuk {barang} dijadwalkan pada {due_date}.")
            #else:
                #st.error("Silakan masukkan semua informasi yang diperlukan.")

    # Fungsi Akuntansi Piutang
    #elif menu == "Akuntansi Piutang":
        #st.subheader("Akuntansi Piutang")
        #customer = st.text_input("Pelanggan")
        #barang1 = st.text_input("Nama Barang")
        #amount = st.number_input("Jumlah Penerimaan", min_value=0.0)
        #quantity = st.number_input("Kuantitas", min_value=1)
        #invoice_date = st.date_input(" Tanggal Faktur", datetime.today())
        #city1 = st.text_input("Kota Pengiriman Pelanggan")
        
        #if st.button("Simpan Penerimaan"):
            #if customer and barang1 and amount > 0 and quantity > 0:
                #new_row = pd.DataFrame({
                    #"Tipe Transaksi": ["Pemasukan"],
                    #"Pelanggan": [customer],
                    #"Nama Barang": [barang1],
                    #"Penjualan": [amount],
                    #"Kuantitas": [quantity],
                    #"Tanggal": [invoice_date],
                    #"Kota Pengiriman Pelanggan": [city1]
                #})
                #st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
                #st.success(f"Penerimaan sebesar {amount} dari {customer} untuk {barang1} pada {invoice_date}.")
            #else:
                #st.error("Silakan masukkan semua informasi yang diperlukan.")

    # Fungsi Penganggaran dan Peramalan
    elif menu == "Penganggaran dan Peramalan":
        st.subheader("Penganggaran dan Peramalan")
    
        if not st.session_state.data.empty:
            # Menghitung rata-rata pemasukan dan pengeluaran bulanan
            monthly_data = st.session_state.data.copy()
    
            # Pastikan kolom 'Tanggal' dalam format datetime
            monthly_data['Tanggal'] = pd.to_datetime(monthly_data['Tanggal'], errors='coerce')
    
            # Hapus entri yang tidak valid 
            monthly_data = monthly_data.dropna(subset=['Tanggal'])
    
            # Menambahkan kolom Bulan untuk analisis bulanan
            monthly_data['Bulan'] = monthly_data['Tanggal'].dt.to_period('M')
    
            # Mengelompokkan data berdasarkan Bulan dan Tipe Transaksi
            monthly_summary = monthly_data.groupby(['Bulan']).agg({
                'Penjualan': 'sum',
                'Kuantitas': 'sum'
            }).unstack(fill_value=0)
    
            st.write("Ringkasan Bulanan:")
            st.dataframe(monthly_summary)
    
            # Input jumlah bulan untuk peramalan
            forecast_months = st.number_input("Jumlah Bulan untuk Peramalan", min_value=1, max_value=12, value=3)
    
            # Pilih metode peramalan
            forecasting_method = st.selectbox("Pilih Metode Peramalan", ["ARIMA", "Exponential Smoothing", "Moving Average", "Naive", "Prophet", "Average"])
    
            # Forecasting for each product
            # Forecasting for each product
            forecast_results = []
            # Inside your forecasting loop
            event_impact = analyze_event_impact(monthly_data, event_dates)
            
            for product in monthly_data['Nama Barang'].unique():
                product_sales_data = monthly_data[monthly_data['Nama Barang'] == product].groupby('Bulan')['Penjualan'].sum()
                product_quantity_data = monthly_data[monthly_data['Nama Barang'] == product].groupby('Bulan')['Kuantitas'].sum()
                
                if product_sales_data.nunique() <= 1 or product_quantity_data.nunique() <= 1:
                    continue  # Skip to the next product
                
                try:
                    sales_forecast, method_used_sales = select_forecasting_method(product_sales_data, steps=forecast_months, method=forecasting_method)
                    quantity_forecast, method_used_quantity = select_forecasting_method(product_quantity_data, steps=forecast_months, method=forecasting_method)
                    
                    for month_offset in range(forecast_months):
                        forecast_date = (monthly_data['Bulan'].max() + month_offset + 1).to_timestamp().date()
                        event_occurred = any(forecast_date in event for event in event_dates.values())
                        
                        forecast_value_sales = sales_forecast[month_offset]
                        forecast_value_quantity = quantity_forecast[month_offset]
                        
                        # Check if the product has shown an increase during events
                        if product in event_impact and event_occurred:
                            forecast_value_sales *= 1.1  # Increase by 10% if the product has shown an increase during events
                            forecast_value_quantity *= 1.1  # Increase by 10% on quantity
                        
                        forecast_results.append({
                            'Nama Barang': product,
                            'Tanggal': forecast_date,
                            'Kuantitas': int(forecast_value_quantity),
                            'Penjualan': forecast_value_sales,
                            'Event': event_occurred
                        })
                except ValueError:
                    continue  # Skip this product if there is an error during forecasting
            # Convert the results into a DataFrame
            forecast_df = pd.DataFrame(forecast_results)
            display_forecast_with_events(forecast_df)  # Menampilkan hasil forecasting dengan event
            
            # Simpan dalam Excel dengan format rapi
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                forecast_df.to_excel(writer, index=False, sheet_name="Hasil Peramalan Nama Barang")
                workbook = writer.book
                worksheet = writer.sheets["Hasil Peramalan Nama Barang"]
                worksheet.set_column('A:A', 20)  # Atur lebar kolom untuk 'Bulan'
                worksheet.set_column('B:B', 15)  # Atur lebar kolom untuk 'Kuantitas'
                worksheet.set_column('C:C', 15)  # Atur lebar kolom untuk 'Penjualan'
                worksheet.set_column('D:D', 20)  # Atur Lebar Kolom untuk 'Event'
            
            # Menyediakan tombol download untuk file Excel
            st.download_button(
                label="Download Hasil Peramalan Produk (Excel)",
                data=excel_buffer.getvalue(),
                file_name="hasil_peramalan_produk.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            
            # Plotting with Plotly
            fig = go.Figure()
            for product in forecast_df['Nama Barang'].unique():
                product_data = forecast_df[forecast_df['Nama Barang'] == product]
                fig.add_trace(go.Scatter(
                    x=product_data['Tanggal'],
                    y=product_data['Penjualan'],
                    mode='lines+markers',
                    name=f'Penjualan {product}',
                    text=product_data['Kuantitas'],  # Show quantity on hover
                    hoverinfo='text+y',  # Show info on hover
                    line=dict(width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=product_data['Tanggal'],
                    y=product_data['Kuantitas'],
                    mode='lines+markers',
                    name=f'Kuantitas {product}',
                    text=product_data['Kuantitas'],  # Show quantity on hover
                    hoverinfo='text+y',  # Show info on hover line=dict(dash='dash', width=2)
                ))
            
            fig.update_layout(
                title='Peramalan Penjualan dan Kuantitas Produk',
                xaxis_title='Tanggal',
                yaxis_title='Jumlah',
                hovermode='closest'
            )
            
            st.plotly_chart(fig)  # Display interactive chart in Streamlit

            # Wawasan Berdasarkan Hasil Peramalan
            insights_product = []
            for product in forecast_df['Nama Barang'].unique():
                product_data = forecast_df[forecast_df['Nama Barang'] == product]
                product_data['Penjualan'] = pd.to_numeric(product_data['Penjualan'].str.replace(',', ''), errors='coerce')
                average_forecast_sales = product_data['Penjualan'].mean()
                trend = product_data['Penjualan'].iloc[-1] - product_data['Penjualan'].iloc[-2]  # Perubahan dari bulan terakhir ke bulan sebelumnya

                if trend > 0:
                    insights_product.append({
                        'Nama Barang': product,
                        'Rata-rata Penjualan': average_forecast_sales,
                        'Tren': 'Meningkat',
                        'Rekomendasi': 'Siapkan stok tambahan dan pertimbangkan promosi.'
                    })
                elif trend < 0:
                    insights_product.append({
                        'Nama Barang': product,
                        'Rata-rata Penjualan': average_forecast_sales,
                        'Tren': 'Menurun',
                        'Rekomendasi': 'Evaluasi strategi pemasaran dan pertimbangkan diskon.'
                    })
                else:
                    insights_product.append({
                        'Nama Barang': product,
                        'Rata-rata Penjualan': average_forecast_sales,
                        'Tren': 'Stabil',
                        'Rekomendasi': 'Pertahankan strategi pemasaran saat ini.'
                    })

            # Buat DataFrame untuk wawasan produk
            insights_product_df = pd.DataFrame(insights_product)
            st.write("Wawasan Berdasarkan Hasil Peramalan Produk:")
            st.dataframe(insights_product_df)

            # Simpan dalam Excel dengan format rapi
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                insights_product_df.to_excel(writer, index=False, sheet_name="Wawasan Produk")

                # Dapatkan workbook dan worksheet
                workbook = writer.book
                worksheet = writer.sheets["Wawasan Produk"]

                # Atur lebar kolom untuk menghindari ####
                worksheet.set_column('A:A', 20)  # Atur lebar kolom untuk 'Nama Barang'
                worksheet.set_column('B:B', 20)  # Atur lebar kolom untuk 'Rata-rata Penjualan'
                worksheet.set_column('C:C', 15)  # Atur lebar kolom untuk 'Tren'
                worksheet.set_column('D:D', 50)  # Atur lebar kolom untuk 'Rekomendasi'

            # Menyediakan tombol download untuk file Excel
            st.download_button(
                label="Download Wawasan Peramalan Produk (Excel)",
                data=excel_buffer.getvalue(),
                file_name="wawasan_peramalan_produk .xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            insights_quantity_product = []

            # Analisis untuk Produk
            for product in forecast_df['Nama Barang'].unique():
                product_data = forecast_df[forecast_df['Nama Barang'] == product]
                average_forecast_quantity = product_data['Kuantitas'].mean()
                trend_quantity = product_data['Kuantitas'].iloc[-1] - product_data['Kuantitas'].iloc[-2]  # Perubahan dari bulan terakhir ke bulan sebelumnya

                if trend_quantity > 0:
                    insights_quantity_product.append({
                        'Nama Barang': product,
                        'Rata-rata Kuantitas': average_forecast_quantity,
                        'Tren': 'Meningkat',
                        'Rekomendasi': 'Siapkan stok tambahan untuk memenuhi permintaan yang meningkat.'
                    })
                elif trend_quantity < 0:
                    insights_quantity_product.append({
                        'Nama Barang': product,
                        'Rata-rata Kuantitas': average_forecast_quantity,
                        'Tren': 'Menurun',
                        'Rekomendasi': 'Evaluasi alasan penurunan dan pertimbangkan untuk mengurangi stok.'
                    })
                else:
                    insights_quantity_product.append({
                        'Nama Barang': product,
                        'Rata-rata Kuantitas': average_forecast_quantity,
                        'Tren': 'Stabil',
                        'Rekomendasi': 'Pertahankan tingkat stok saat ini.'
                    })

            insights_quantity_product_df = pd.DataFrame(insights_quantity_product)
            st.write("Wawasan Berdasarkan Hasil Peramalan Kuantitas Produk:")
            st.dataframe(insights_quantity_product_df)

            # Simpan dalam Excel dengan format rapi
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                insights_quantity_product_df.to_excel(writer, index=False, sheet_name="Wawasan Kuantitas")

                # Dapatkan workbook dan worksheet
                workbook = writer.book
                worksheet = writer.sheets["Wawasan Kuantitas"]

                # Atur lebar kolom untuk menghindari ####
                worksheet.set_column('A:A', 20)  # Atur lebar kolom untuk 'Nama Barang'
                worksheet.set_column('B:B', 20)  # Atur lebar kolom untuk 'Rata-rata Kuantitas'
                worksheet.set_column('C:C', 15)  # Atur lebar kolom untuk 'Tren'
                worksheet.set_column('D:D', 50)  # Atur lebar kolom untuk 'Rekomendasi'

            # Menyediakan tombol download untuk file Excel
            st.download_button(
                label="Download Wawasan Peramalan Kuantitas Produk (Excel)",
                data=excel_buffer.getvalue(),
                file_name="wawasan_peramalan_kuantitas_produk.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            # Forecasting for each customer
            customer_forecast_results = []

            for customer in monthly_data['Pelanggan'].unique():
                customer_data = monthly_data[monthly_data['Pelanggan'] == customer].groupby('Bulan')['Penjualan'].sum()
                customer_quantity = monthly_data[monthly_data['Pelanggan'] == customer].groupby('Bulan')['Kuantitas'].sum()

                # Check if customer_data has enough variation
                if customer_data.nunique() <= 1 or customer_quantity.nunique() <= 1:
                    continue  # Skip to the next customer

                try:
                    # Forecast sales and quantity
                    customer_forecast, method_used_customer = select_forecasting_method(customer_data, steps=forecast_months, method=forecasting_method)
                    quantity_forecast, method_used_quantity = select_forecasting_method(customer_quantity, steps=forecast_months, method=forecasting_method)

                    # Store the results in a list
                    for month_offset in range(forecast_months):
                        customer_forecast_results.append({
                            'Pelanggan': customer,
                            'Tanggal': (monthly_data['Bulan'].max() + month_offset + 1).to_timestamp(),
                            'Kuantitas': int(quantity_forecast[month_offset]),
                            'Penjualan': customer_forecast[month_offset]
                        })
                except ValueError:
                    continue  # Skip this customer if there is an error during forecasting

            # Convert the results into a DataFrame
            customer_forecast_df = pd.DataFrame(customer_forecast_results)

            if not customer_forecast_df.empty:
                st.write("Hasil Peramalan Transaksi Pelanggan:")
                customer_forecast_df['Penjualan'] = customer_forecast_df['Penjualan'].apply(lambda x: f"{x:,.2f}")
                customer_forecast_df['Tanggal'] = customer_forecast_df['Tanggal'].dt.strftime('%d-%m-%y ')  # Format tanggal
                st.dataframe(customer_forecast_df)

                # Simpan dalam Excel dengan format rapi
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                    customer_forecast_df.to_excel(writer, index=False, sheet_name="Hasil Peramalan Pelanggan")

                    # Dapatkan workbook dan worksheet
                    workbook = writer.book
                    worksheet = writer.sheets["Hasil Peramalan Pelanggan"]

                    # Atur lebar kolom untuk menghindari ####
                    worksheet.set_column('A:A', 25)  # Atur lebar kolom untuk 'Pelanggan'
                    worksheet.set_column('B:B', 15)  # Atur lebar kolom untuk 'Kuantitas'
                    worksheet.set_column('C:C', 15)  # Atur lebar kolom untuk 'Penjualan'

                # Menyediakan tombol download untuk file Excel
                st.download_button(
                    label="Download Hasil Peramalan Pelanggan (Excel)",
                    data=excel_buffer.getvalue(),
                    file_name="hasil_peramalan_pelanggan.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

                # Visualization using Plotly
                fig = go.Figure()
                for customer in customer_forecast_df['Pelanggan'].unique():
                    customer_data = customer_forecast_df[customer_forecast_df['Pelanggan'] == customer]
                    fig.add_trace(go.Scatter(
                        x=customer_data['Tanggal'],
                        y=customer_data['Penjualan'],
                        mode='lines+markers',
                        name=f'Peramalan Penjualan {customer}',
                        text=customer_data['Kuantitas'],  # Show quantity on hover
                        hoverinfo='text+y',  # Show info on hover
                        line=dict(width=2)
                    ))

                fig.update_layout(
                    title='Peramalan Penjualan Pelanggan',
                    xaxis_title='Tanggal',
                    yaxis_title='Penjualan',
                    hovermode='closest'
                )

                st.plotly_chart(fig)  # Display interactive chart in Streamlit

            # Menghitung rata-rata penjualan dan kuantitas untuk setiap pelanggan
            average_customer_forecast_results = []

            for customer in customer_forecast_df['Pelanggan'].unique():
                customer_data = customer_forecast_df[customer_forecast_df['Pelanggan'] == customer]

                if not customer_data.empty:
                    # Menghapus tanda koma dan mengonversi ke float
                    customer_data['Penjualan'] = pd.to_numeric(customer_data['Penjualan'].str.replace(',', ''), errors='coerce')
                    customer_data['Kuantitas'] = pd.to_numeric(customer_data['Kuantitas'], errors='coerce')

                    # Hapus baris yang memiliki NaN setelah konversi
                    customer_data = customer_data.dropna(subset=['Penjualan', 'Kuantitas'])

                    average_sales = customer_data['Penjualan'].mean()
                    average_quantity = customer_data['Kuantitas'].mean()

                    average_customer_forecast_results.append({
                        'Pelanggan': customer,
                        'Rata-rata Penjualan': average_sales,
                        'Rata-rata Kuantitas': int(average_quantity)
                    })

            # Convert the results into a DataFrame
            average_customer_forecast_df = pd.DataFrame(average_customer_forecast_results)

            # Tampilkan tabel rata-rata pelanggan
            if not average_customer_forecast_df.empty:
                st.write("Rata-rata Hasil Peramalan Pelanggan:")
                average_customer_forecast_df['Rata-rata Penjualan'] = average_customer_forecast_df['Rata-rata Penjualan'].apply(lambda x: f"{x:,.2f}")
                st.dataframe(average_customer_forecast_df)

                # Simpan dalam Excel dengan format rapi
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                    average_customer_forecast_df.to_excel(writer, index=False, sheet_name="Rata-rata Pelanggan")

                    # Dapatkan workbook dan worksheet
                    workbook = writer.book
                    worksheet = writer.sheets["Rata-rata Pelanggan"]

                    # Atur lebar kolom untuk menghindari ####
                    worksheet.set_column('A:A', 25)  # Atur lebar kolom untuk 'Pelanggan'
                    worksheet.set_column('B:B', 20)  # Atur lebar kolom untuk 'Rata-rata Penjualan'
                    worksheet.set_column('C:C', 20)  # Atur lebar kolom untuk 'Rata-rata Kuantitas'

                # Menyediakan tombol download untuk file Excel
                st.download_button(
                    label="Download Rata-rata Pelanggan (Excel)",
                    data=excel_buffer.getvalue(),
                    file_name="rata_rata_pelanggan.xlsx", 
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

                # Wawasan Berdasarkan Hasil Peramalan Pelanggan
                insights_customer = []
                for customer in customer_forecast_df['Pelanggan'].unique():
                    customer_data = customer_forecast_df[customer_forecast_df['Pelanggan'] == customer]
                    customer_data['Penjualan'] = pd.to_numeric(customer_data['Penjualan'].str.replace(',', ''), errors='coerce')
                    average_forecast_customer_sales = customer_data['Penjualan'].mean()
                    trend_customer = customer_data['Penjualan'].iloc[-1] - customer_data['Penjualan'].iloc[-2]  # Perubahan dari bulan terakhir ke bulan sebelumnya

                    if trend_customer > 0:
                        insights_customer.append({
                            'Pelanggan': customer,
                            'Rata-rata Penjualan': average_forecast_customer_sales,
                            'Tren': 'Meningkat',
                            'Rekomendasi': 'Tingkatkan komunikasi dan penawaran khusus.'
                        })
                    elif trend_customer < 0:
                        insights_customer.append({
                            'Pelanggan': customer,
                            'Rata-rata Penjualan': average_forecast_customer_sales,
                            'Tren': 'Menurun',
                            'Rekomendasi': 'Lakukan survei kepuasan pelanggan.'
                        })
                    else:
                        insights_customer.append({
                            'Pelanggan': customer,
                            'Rata-rata Penjualan': average_forecast_customer_sales,
                            'Tren': 'Stabil',
                            'Rekomendasi': 'Pertahankan hubungan baik dan tawarkan produk baru.'
                        })

                # Buat DataFrame untuk wawasan pelanggan
                insights_customer_df = pd.DataFrame(insights_customer)
                st.write("Wawasan Berdasarkan Hasil Peramalan Pelanggan:")
                st.dataframe(insights_customer_df)

                # Simpan dalam Excel dengan format rapi
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                    insights_customer_df.to_excel(writer, index=False, sheet_name="Wawasan Peramalan Penjualan")

                    # Dapatkan workbook dan worksheet
                    workbook = writer.book
                    worksheet = writer.sheets["Wawasan Peramalan Penjualan"]

                    # Atur lebar kolom untuk menghindari ####
                    worksheet.set_column('A:A', 20)  # Atur lebar kolom untuk 'Pelanggan'
                    worksheet.set_column('B:B', 20)  # Atur lebar kolom untuk 'Rata-rata Penjualan'
                    worksheet.set_column('C:C', 20)  # Atur lebar kolom untuk 'Tren'
                    worksheet.set_column('D:D', 50)  # Atur lebar kolom untuk 'Rekomendasi'

                # Menyediakan tombol download untuk file Excel
                st.download_button(
                    label="Download Wawasan Peramalan Penjualan (Excel)",
                    data=excel_buffer.getvalue(),
                    file_name="wawasan_peramalan_pelanggan_penjualan.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

                insights_quantity_customer = []

                # Analisis untuk Pelanggan
                for customer in customer_forecast_df['Pelanggan'].unique():
                    customer_data = customer_forecast_df[customer_forecast_df['Pelanggan'] == customer]
                    average_forecast_customer_quantity = customer_data['Kuantitas'].mean()
                    trend_customer_quantity = customer_data['Kuantitas'].iloc[-1] - customer_data['Kuantitas'].iloc[-2]  # Perubahan dari bulan terakhir ke bulan sebelumnya

                    if trend_customer_quantity > 0:
                        insights_quantity_customer.append({
                            'Pelanggan': customer,
                            'Rata-rata Kuantitas': int(average_forecast_customer_quantity),
                            'Tren': 'Meningkat',
                            'Rekomendasi': 'Tingkatkan persediaan untuk memenuhi permintaan yang meningkat.'
                        })
                    elif trend_customer_quantity < 0:
                        insights_quantity_customer.append({
                            'Pelanggan': customer,
                            'Rata-rata Kuantitas': int(average_forecast_customer_quantity),
                            'Tren': 'Menurun',
                            'Rekomendasi': 'Evaluasi alasan penurunan dan pertimbangkan untuk mengurangi persediaan.'
                        })
                    else:
                        insights_quantity_customer.append({
                            'Pelanggan': customer,
                            'Rata-rata Kuantitas': int(average_forecast_customer_quantity),
                            'Tren': 'Stabil',
                            'Rekomendasi': 'Pertahankan tingkat persediaan saat ini.'
                        })

                insights_quantity_customer_df = pd.DataFrame(insights_quantity_customer)
                st.write("Wawasan Berdasarkan Hasil Peramalan Kuantitas Pelanggan:")
                st.dataframe(insights_quantity_customer_df)

                # Simpan dalam Excel dengan format rapi
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                    insights_quantity_customer_df.to_excel(writer, index=False, sheet_name="Wawasan Peramalan Kuantitas")

                    # Dapatkan workbook dan worksheet
                    workbook = writer.book
                    worksheet = writer.sheets["Wawasan Peramalan Kuantitas"]

                    # Atur lebar kolom untuk menghindari ####
                    worksheet.set_column('A:A', 20)  # Atur lebar kolom untuk 'Pelanggan'
                    worksheet.set_column('B:B', 20)  # Atur lebar kolom untuk 'Rata-rata Kuantitas'
                    worksheet.set_column('C:C', 20)  # Atur lebar kolom untuk 'Tren'
                    worksheet.set_column('D:D', 50)  # Atur lebar kolom untuk 'Rekomendasi'

                # Menyediakan tombol download untuk file Excel
                st.download_button(
                    label="Download Wawasan Peramalan Kuantitas (Excel)",
                    data=excel_buffer.getvalue(),
                    file_name="wawasan_peramalan_pelanggan_kuantitas.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

                # Menghitung prediksi pelanggan untuk setiap produk
                predicted_customers = []  # Initialize list for storing customer predictions

                for product in forecast_df['Nama Barang'].unique():
                    # Ambil data penjualan dan kuantitas untuk produk ini
                    product_forecast = forecast_df[forecast_df['Nama Barang'] == product]

                    # Menghapus tanda koma dan mengonversi ke float
                    product_forecast['Penjualan'] = product_forecast['Penjualan'].str.replace(',', '').astype(float)
                    product_forecast['Kuantitas'] = product_forecast['Kuantitas'].astype(float)  # Pastikan kuantitas juga dalam format float

                    # Ambil rata-rata penjualan dan kuantitas dari hasil peramalan
                    average_sales = product_forecast['Penjualan'].mean()
                    average_quantity = product_forecast['Kuantitas'].mean()

                    # Tentukan pelanggan yang mungkin membeli produk ini
                    potential_customers = monthly_data[monthly_data['Nama Barang'] == product]['Pelanggan'].unique()

                    # Simpan hasil prediksi dalam list
                    for customer in potential_customers:
                        # Calculate predicted sales and quantity based on historical data
                        customer_sales_data = monthly_data[(monthly_data['Nama Barang'] == product) & (monthly_data['Pelanggan'] == customer)]

                        if not customer_sales_data.empty:
                            predicted_sales = customer_sales_data['Penjualan'].mean()  # Use historical average
                            predicted_quantity = customer_sales_data['Kuantitas'].mean()  # Use historical average

                            predicted_customers.append({
                                'Nama Barang': product,
                                'Pelanggan': customer,
                                'Kuantitas Diprediksi': int(predicted_quantity),
                                'Penjualan Diprediksi': predicted_sales
                            })

                # Convert the results into a DataFrame
                predicted_customers_df = pd.DataFrame(predicted_customers)

                # Tampilkan tabel prediksi pelanggan
                if not predicted_customers_df.empty:
                    st.write("Prediksi Pelanggan untuk Bulan Depan:")
                    predicted_customers_df['Penjualan Diprediksi'] = predicted_customers_df['Penjualan Diprediksi'].apply(lambda x: f"{x:,.2f}")
                    st.dataframe(predicted_customers_df)

                    # Simpan dalam Excel dengan format rapi
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                        predicted_customers_df.to_excel(writer, index=False, sheet_name="Prediksi Pelanggan")

                        # Dapatkan workbook dan worksheet
                        workbook = writer.book
                        worksheet = writer.sheets["Prediksi Pelanggan"]

                        # Atur lebar kolom untuk menghindari ####
                        worksheet.set_column('A:A', 20)  # Atur lebar kolom untuk 'Nama Barang'
                        worksheet.set_column('B:B', 20)  # Atur lebar kolom untuk 'Pelanggan'
                        worksheet.set_column('C:C', 20)  # Atur lebar kolom untuk 'Kuantitas Diprediksi'
                        worksheet.set_column('D:D', 20)  # Atur lebar kolom untuk 'Penjualan Diprediksi'

                    # Menyediakan tombol download untuk file Excel
                    st.download_button(
                        label="Download Prediksi Pelanggan (Excel)",
                        data=excel_buffer.getvalue(),
                        file_name="prediksi_pelanggan.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheet.spreadsheetml.sheet",
                    )

                    # Meminta pengguna untuk memilih produk untuk grafik
                    unique_products = predicted_customers_df['Nama Barang'].unique()
                    selected_products = st.multiselect("Pilih Nama Barang untuk Grafik", unique_products)

                    # Jika tidak ada produk yang dipilih, tampilkan pesan
                    if not selected_products:
                        st.warning("Silakan pilih setidaknya satu nama barang untuk melihat grafik.")
                    else:
                        # Filter data berdasarkan produk yang dipilih
                        filtered_predictions = predicted_customers_df[predicted_customers_df['Nama Barang'].isin(selected_products)]

                        # Plotting dengan Plotly
                        fig = px.line(
                            filtered_predictions,
                            x='Pelanggan',
                            y='Kuantitas Diprediksi',
                            color='Nama Barang',
                            markers=True,
                            hover_data=['Pelanggan', 'Kuantitas Diprediksi']
                        )

                        fig.update_layout(
                            title='Prediksi Kuantitas Berdasarkan Produk dan Pelanggan',
                            xaxis_title='Pelanggan',
                            yaxis_title='Kuantitas Diprediksi',
                            xaxis=dict(tickangle=45),
                        )

                        # Menampilkan grafik di Streamlit
                        st.plotly_chart(fig, use_container_width=True)

                    threshold = st.number_input("Masukkan Ambang Batas untuk Deteksi Lonjakan", min_value=1.0, value=1.5)

                    if st.button("Deteksi Lonjakan Penjualan"):
                        if 'data' in st.session_state and not st.session_state.data.empty:
                            sales_data = st.session_state.data.groupby('Tanggal')['Penjualan'].sum()
                            spikes = detect_sales_spikes(sales_data, threshold)
                                
                            if not spikes.empty:
                                st.write("Lonjakan Penjualan Terdeteksi:")
                                st.dataframe(spikes)
                                    
                                plt.figure(figsize=(12, 6))
                                plt.plot(sales_data.index, sales_data, label='Penjualan', color='blue')
                                plt.scatter(spikes.index, spikes, color='red', label='Lonjakan Penjualan', marker='o')
                                plt.title('Analisis Penjualan dan Deteksi Lonjakan')
                                plt.xlabel('Tanggal')
                                plt.ylabel('Penjualan')
                                plt.legend()
                                st.pyplot(plt)
                            else:
                                st.write("Tidak ada lonjakan penjualan yang terdeteksi.")
                        else:
                            st.warning("Data penjualan tidak tersedia untuk analisis.")

                    # Menyimpan DataFrame hasil peramalan ke dalam session state
                    st.session_state.forecast_df = forecast_df
                    st.session_state.insights_product_df = insights_product_df
                    st.session_state.insights_quantity_product_df = insights_quantity_product_df
                    st.session_state.customer_forecast_df = customer_forecast_df
                    st.session_state.insights_customer_df = insights_customer_df
                    st.session_state.insights_quantity_customer_df = insights_quantity_customer_df
                    st.session_state.predicted_customers_df = predicted_customers_df
                    st.session_state.average_customer_forecast_df = average_customer_forecast_df
                    
                    # Menambahkan chatbot di sidebar
                    st.header("Chatbot")
                    st.write("Tanyakan tentang prediksi penjualan dan pelanggan.")
                    
                    # Inisialisasi chat history jika belum ada
                    if "messages" not in st.session_state:
                        st.session_state.messages = []  # Inisialisasi sebagai list kosong
                    
                    # Inisialisasi grafik jika belum ada
                    if "graphs" not in st.session_state:
                        st.session_state.graphs = []  # Inisialisasi sebagai list kosong untuk menyimpan grafik
                    
                    # Menampilkan pesan chat dari history pada rerun aplikasi
                    for i, message in enumerate(st.session_state.messages):
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])
                    
                    # Menampilkan semua grafik yang telah disimpan
                    for graph in st.session_state.graphs:
                        st.plotly_chart(graph, use_container_width=True)
                    
                    # In the chatbot section
                    if prompt := st.text_input("Tanya saya tentang prediksi:"):
                        # Get chat history if not Null
                        messages_history = st.session_state.get("messages", [])
                        history = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in messages_history]) or " "
                    
                        # Display user message in chat message container
                        st.chat_message("user").markdown(prompt)
                        # Add user message to chat history
                        st.session_state.messages.append({"role": "user", "content": prompt})
                    
                        # Prepare contexts from session state
                        contexts = {
                            "forecast": st.session_state.forecast_df.to_string(index=False),
                            "insights_product": st.session_state.insights_product_df.to_string(index=False),
                            "insights_quantity_product": st.session_state.insights_quantity_product_df.to_string(index=False),
                            "customer_forecast": st.session_state .customer_forecast_df.to_string(index=False),
                            "insights_customer": st.session_state.insights_customer_df.to_string(index=False),
                            "insights_quantity_customer": st.session_state.insights_quantity_customer_df.to_string(index=False),
                            "predicted_customers": st.session_state.predicted_customers_df.to_string(index=False), 
                            "average_customer_forecast_df": st.session_state.average_customer_forecast_df.to_string(index=False)
                        }
                    
                        # Call the chat function
                        response = chat(contexts, history , prompt)
                        answer = response["answer"]
                    
                        # Display assistant response in chat message container
                        with st.chat_message("assistant"):
                            st.markdown(answer)

                            # Filter data untuk grafik berdasarkan pertanyaan
                            if "tren" in prompt.lower() or "grafik" in prompt.lower():
                                # Ekstrak nama barang dari pertanyaan
                                match = re.search(r'tren\s+(.+)', prompt, re.IGNORECASE)
                                if match:
                                    nama_barang = match.group(1).strip()
                                    # Filter data berdasarkan nama barang
                                    trend_data = predicted_customers_df[predicted_customers_df['Nama Barang'].str.contains(nama_barang, case=False, na=False)]
                                    
                                    if not trend_data.empty:
                                        # Plotting dengan Plotly
                                        fig = px.line(
                                            trend_data,
                                            x='Pelanggan',
                                            y='Kuantitas Diprediksi',
                                            markers=True,
                                            title=f'Tren Kuantitas Diprediksi untuk {nama_barang}',
                                            labels={'Kuantitas Diprediksi': 'Kuantitas Diprediksi', 'Pelanggan': 'Pelanggan'}
                                        )
                                        fig.update_layout(
                                            xaxis_title='Pelanggan',
                                            yaxis_title='Kuantitas Diprediksi',
                                            xaxis=dict(tickangle=45),
                                            showlegend=True,
                                        )
                                        # Menambahkan lebih banyak grid
                                        fig.update_xaxes(showgrid=True, gridcolor='LightGray', gridwidth=1)
                                        fig.update_yaxes(showgrid=True, gridcolor='LightGray', gridwidth=1)
                            
                                        st.markdown(f"Berikut adalah grafik tren kuantitas diprediksi untuk {nama_barang}:")
                                        st.plotly_chart(fig, use_container_width=True)
                                        st.session_state.graphs.append(fig)  # Simpan grafik ke dalam state
                                    else:
                                        st.markdown(f"Tidak ada data untuk barang '{nama_barang}'.")
                            
                            elif "pelanggan" in prompt.lower() and "menurun" in prompt.lower():
                                match = re.search(r'pelanggan\s+yang\s+mengalami\s+penurunan', prompt, re.IGNORECASE)
                                if match:
                                    # Ambil data pelanggan yang mengalami penurunan
                                    trend_data = insights_quantity_customer_df[
                                        insights_quantity_customer_df['Rata-rata Kuantitas'] < 100
                                    ]
                                    
                                    if not trend_data.empty:
                                        fig = px.bar(
                                            trend_data,
                                            x='Pelanggan',
                                            y='Rata-rata Kuantitas',
                                            title='Pelanggan yang Mengalami Penurunan',
                                            labels={'Pelanggan': 'Pelanggan', 'Rata-rata Kuantitas': 'Rata-rata Kuantitas'},
                                            text='Rata-rata Kuantitas'
                                        )
                                        fig.update_layout(
                                            xaxis_title='Pelanggan',
                                            yaxis_title='Rata-rata Kuamtitas',
                                            xaxis=dict(tickangle=45),
                                            showlegend=False,
                                        )
                                    
                                        st.markdown("Berikut adalah grafik untuk pelanggan yang mengalami penurunan:")
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.markdown("Tidak ada pelanggan yang mengalami penurunan dengan Rata-rata Kuantitas di bawah 100.") 
                                            
                            # Cek jika pertanyaan berkaitan dengan pelanggan yang mengalami kenaikan
                            if "pelanggan" in prompt.lower() and "naik" in prompt.lower():
                                # Ambil data pelanggan yang mengalami kenaikan
                                trend_data = average_customer_forecast_df[average_customer_forecast_df['Rata-rata Penjualan'] > 0]
                            
                                if not trend_data.empty:
                                    fig = px.bar(
                                        trend_data,
                                        x='Pelanggan',
                                        y='Rata-rata Penjualan',
                                        title='Pelanggan yang Mengalami Kenaikan',
                                        labels={'Pelanggan': 'Pelanggan', 'Rata-rata Penjualan': 'Rata-rata Penjualan'},
                                        text='Rata-rata Penjualan'
                                    )
                                    fig.update_layout(
                                        xaxis_title='Pelanggan',
                                        yaxis_title='Rata-rata Penjualan',
                                        xaxis=dict(tickangle=45),
                                        showlegend=False,
                                    )
                            
                                    st.markdown("Berikut adalah grafik untuk pelanggan yang mengalami kenaikan:")
                                    st.plotly_chart(fig, use_container_width=True)
                                else: 
                                    st.markdown("Tidak ada pelanggan yang mengalami kenaikan.")
                            
                            # Cek jika pertanyaan berkaitan dengan nama barang yang mengalami penurunan
                            elif "barang" in prompt.lower() and "menurun" in prompt.lower():
                                match = re.search(r'barang\ yang\ mengalami\ penurunan', prompt, re.IGNORECASE)
                                if match:
                                    # Ambil data barang yang mengalami penurunan
                                    trend_data = average_customer_forecast_df[average_customer_forecast_df['Rata-rata Penjualan'] < 0]
                            
                                    if not trend_data.empty:
                                        fig = px.bar(
                                            trend_data,
                                            x='Pelanggan',
                                            y='Rata-rata Penjualan',
                                            title='Barang yang Mengalami Penurunan',
                                            labels={'Pelanggan': 'Pelanggan', 'Rata-rata Penjualan': 'Rata-rata Penjualan'},
                                            text='Rata-rata Penjualan'
                                        )
                                        fig.update_layout(
                                            xaxis_title='Pelanggan',
                                            yaxis_title='Rata-rata Penjualan',
                                            xaxis=dict(tickangle=45),
                                            showlegend=False,
                                        )
                            
                                        st.markdown("Berikut adalah grafik untuk barang yang mengalami penurunan:")
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.markdown("Tidak ada barang yang mengalami penurunan.")
                            
                            # Cek jika pertanyaan berkaitan dengan nama barang yang mengalami kenaikan
                            elif "barang" in prompt.lower() and "naik" in prompt.lower():
                                match = re.search(r'barang\ yang\ mengalami\ kenaikan', prompt, re.IGNORECASE)
                                if match:
                                    # Ambil data barang yang mengalami kenaikan
                                    trend_data = average_customer_forecast_df[average_customer_forecast_df['Rata-rata Penjualan'] > 0]
                            
                                    if not trend_data.empty:
                                        fig = px.bar(
                                            trend_data,
                                            x='Pelanggan',
                                            y='Rata-rata Penjualan',
                                            title='Barang yang Mengalami Kenaikan',
                                            labels={'Pelanggan': 'Pelanggan', 'Rata-rata Penjualan': 'Rata-rata Penjualan'},
                                            text='Rata-rata Penjualan'
                                        )
                                        fig.update_layout(
                                            xaxis_title='Pelanggan',
                                            yaxis_title='Rata-rata Penjualan',
                                            xaxis=dict(tickangle=45),
                                            showlegend=False,
                                        )
                            
                                        st.markdown("Berikut adalah grafik untuk barang yang mengalami kenaikan:")
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.markdown("Tidak ada barang yang mengalami kenaikan.")
                    
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": answer})

    elif menu == "Statistik":
        st.subheader("Statistik Perkembangan Pembelian Pelanggan")
    
        if not st.session_state.data.empty:
            df = st.session_state.data.copy()
    
            # Format tanggal
            df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    
            # Kolom penting
            df = df[['Tanggal', 'Pelanggan', 'Nama Barang', 'Kuantitas', 'Kota Pengiriman Pelanggan']]
            df.rename(columns={
                'Kota Pengiriman Pelanggan': 'Kota',
                'Pelanggan': 'Customer'
            }, inplace=True)
    
            # Hitung total kuantitas harian
            grouped = df.groupby(['Kota', 'Customer', 'Tanggal'])['Kuantitas'].sum().reset_index()
    
            # Sidebar filter untuk memilih kota (menggunakan lower case untuk menghindari duplikasi)
            kota_terpilih = st.selectbox("Pilih Kota", sorted(grouped['Kota'].str.lower().unique()))
            df_kota = grouped[grouped['Kota'].str.lower() == kota_terpilih]
    
            # Menambahkan kolom perkembangan
            df_kota.sort_values(by=['Customer', 'Tanggal'], inplace=True)
            df_kota['Perkembangan'] = df_kota.groupby('Customer')['Kuantitas'].diff().fillna(0)
    
            # Menentukan status perkembangan
            df_kota['Status Perkembangan'] = df_kota['Perkembangan'].apply(
                lambda x: 'Naik' if x > 0 else ('Turun' if x < 0 else 'Tetap')
            )
    
            # Menampilkan semua pelanggan dan perkembangan mereka di kota terpilih
            st.write(f"Pelanggan di Kota {kota_terpilih.capitalize()}")
            st.dataframe(df_kota[['Customer', 'Tanggal', 'Kuantitas', 'Status Perkembangan']], use_container_width=True)
    
            # Membuat grafik untuk semua pelanggan di kota terpilih
            st.write("Grafik Perkembangan Kuantitas Pelanggan")
            fig_all = px.line(df_kota, x='Tanggal', y='Kuantitas', color='Customer', markers=True,
                              title=f'Perkembangan Kuantitas Pelanggan di {kota_terpilih.capitalize()}')
            fig_all.update_layout(xaxis_title="Tanggal", yaxis_title="Kuantitas", title_x=0.5)
            st.plotly_chart(fig_all)
    
            # Sidebar filter untuk memilih customer
            if not df_kota.empty:
                customer_terpilih = st.selectbox("Pilih Customer untuk Detail", sorted(df_kota['Customer'].unique()))
                df_cust = df_kota[df_kota['Customer'] == customer_terpilih]
    
                st.write(f"Grafik Perubahan Kuantitas - {customer_terpilih} di {kota_terpilih.capitalize()}")
    
                # Membuat grafik interaktif untuk customer yang dipilih
                fig_cust = px.line(df_cust, x='Tanggal', y='Kuantitas', markers=True,
                                   title=f'Perkembangan Kuantitas {customer_terpilih}')
                fig_cust.update_layout(xaxis_title="Tanggal", yaxis_title="Kuantitas", title_x=0.5)
                st.plotly_chart(fig_cust)
    
                st.markdown("---")
                st.write("Data Rinci untuk Customer Terpilih")
                st.dataframe(df_cust[['Tanggal', 'Kuantitas', 'Status Perkembangan']], use_container_width=True)
    
        else:
            st.warning("Tidak ada data yang dimuat. Silakan pastikan data sudah tersedia di menu awal.")


    # Menambahkan opsi untuk mereset seluruh chat history
    if st.sidebar.button("Reset Chat History"):
        st.session_state.messages = []  # Clear the chat history
        st.success("Chat history telah direset. Anda dapat memulai percakapan baru.")

    # Menampilkan data yang telah dimasukkan
    if st.button("Tampilkan Data"):
        sorted_data = st.session_state.data.sort_values(by=["Tanggal", "Pelanggan", "Nama Barang", "Kuantitas", "Penjualan", "Kota Pengiriman Pelanggan"])
        st.write(sorted_data)

    # Menyimpan data ke file CSV
    if st.button("Simpan Data ke CSV"):
        st.session_state.data.to_csv("data_akuntansi.csv", index=False)
        st.success("Data telah disimpan ke file CSV.")

    # Menambahkan fitur untuk menghapus data
    if st.button("Hapus Data Terakhir"):
        if not st.session_state.data.empty:
            st.session_state.data = st.session_state.data[:-1]
            st.success("Data terakhir telah dihapus.")
        else:
            st.warning("Tidak ada data untuk dihapus.")

    # Menambahkan fitur untuk mengedit data
    if st.button("Edit Data"):
        edit_index = st.number_input("Masukkan Indeks Data yang Ingin Diedit", min_value=0, max_value=len(st.session_state.data)-1)
        if edit_index is not None and edit_index < len(st.session_state.data):
            edited_row = st.session_state.data.iloc[edit_index]
            new_vendor = st.text_input("Pelanggan", value=edited_row["Pelanggan"])
            new_barang = st.text_input("Nama Barang", value=edited_row["Nama Barang"])
            new_amount = st.number_input("Penjualan", value=edited_row["Penjualan"])
            new_quantity = st.number_input("Kuantitas", value=edited_row["Kuantitas"])
            new_date = st.date_input("Tanggal", value=edited_row["Tanggal"])
            
            if st.button("Simpan Perubahan"):
                st.session_state.data.at[ edit_index, "Pelanggan"] = new_vendor
                st.session_state.data.at[edit_index, "Nama Barang"] = new_barang
                st.session_state.data.at[edit_index, "Penjualan"] = new_amount
                st.session_state.data.at[edit_index, "Kuantitas"] = new_quantity
                st.session_state.data.at[edit_index, "Tanggal"] = new_date
                st.success("Data telah diperbarui.")

    # Menambahkan fitur untuk menampilkan ringkasan laporan
    if st.button("Tampilkan Ringkasan Laporan"):
        if not st.session_state.data.empty:
            summary = st.session_state.data.groupby("Tipe Transaksi").agg({"Penjualan": "sum", "Kuantitas": "sum"}).reset_index()
            st.write("Ringkasan Laporan:")
            st.dataframe(summary)
        else:
            st.warning("Tidak ada data untuk ditampilkan.")

    # Opsi untuk menghapus data
    st.sidebar.subheader("Hapus Data")
    if st.sidebar.button("Hapus Semua Data"):
        st.session_state.data = pd.DataFrame(columns=["Pelanggan", "Tanggal", "Tipe Transaksi", "Nama Barang", "Kuantitas", "Penjualan"])
        st.success("Semua data telah dihapus.")

    # Panggil fungsi untuk menyimpan data ke database
    if st.button("Simpan Data ke Database"):
        save_data_to_database(st.session_state.data)
        st.success("Data telah disimpan ke database.")

    # Menambahkan fitur untuk menampilkan data terbaru
    if st.button("Tampilkan Data Terbaru"):
        if not st.session_state.data.empty:
            latest_data = st.session_state.data.tail(5)
            st.write("Data Terbaru:")
            st.dataframe(latest_data)
        else:
            st.warning("Tidak ada data untuk ditampilkan.")

    # Menambahkan opsi untuk menghapus entri tertentu
    st.sidebar.subheader("Hapus Entri Tertentu")
    if not st.session_state.data.empty:
        entry_index = st.sidebar.number_input("Pilih Indeks Entri untuk Dihapus", min_value=0, max_value=len(st.session_state.data)-1, step=1)
        if st.sidebar.button("Hapus Entri"):
            st.session_state.data = st.session_state.data.drop(entry_index).reset_index(drop=True)
            st.success("Entri telah dihapus.")

    # Menambahkan opsi untuk logout
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.success("Anda telah logout.")
