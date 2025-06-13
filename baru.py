import streamlit as st
import pandas as pd
import numpy as np
import holidays
from datetime import datetime
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
import os
import sys
from sqlalchemy import create_engine


#engine = create_engine('sqlite:///data_unggahan.db')

# Judul aplikasi 
st.title("📈 Aplikasi")

# SQLAlchemy Engine
#engine = create_engine(f"sqlite:///{DATABASE_PATH}")

        
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
        'Natal': [], 
    }
    for date, name in id_holidays.items():
        if 'Idul Fitri' in name:
            event_dates['Idul Fitri'].append(date)
        elif 'Idul Adha' in name:
            event_dates['Idul Adha'].append(date)
        elif 'Tahun Baru' in name:
            event_dates['Tahun Baru'].append(date)
        elif 'Natal' in name:
            event_dates['Natal'].append(date)
    return event_dates
    
    # Mendapatkan event untuk tahun ini
current_year = datetime.now().year
event_dates = get_event_dates(current_year)
    
def add_event_column(data):
    year = data['Tanggal'].dt.year.unique()[0]
    event_dates = get_event_dates(year)
    data['Event'] = data['Tanggal'].apply(lambda x: any(x.date() in event for event in event_dates.values()))
    return data
    
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

# Path permanen untuk database dan pickle
#DATABASE_PATH = get_app_data_path("data_unggahan.db")
#DATA_FILE = get_app_data_path("saved_data.pkl")
    
     # Lokasi file penyimpanan data
#DATA_FILE = "saved_data.pkl"
#engine = create_engine(f"sqlite:///{DATABASE_PATH}")

# Ambil data dari database

#f "database_url" not in st.secrets:
    #st.stop()
#DATABASE_URL = st,secrets["database_url"]
#engine = create_engine(DATABASE_URL)

#def save_data_to_db(df, table_name):
    #df.to_sql(table_name, engine, if_exists="append", index=False)

#def load_data_from_db(table_name):
    #query = f"SELECT * FROM {table_name}"
    #try:
        #df = pd.read_sql(query, engine)
        #return df
    #except Exception as e:
        #st.error(f"Gagal memuat tabel {table_name}: {e}")
        #return pd.DataFrame()

        
    # Fungsi untuk load data dari file
#def load_data():
    #if os.path.exists(DATA_FILE):
        #return pd.read_pickle(DATA_FILE)
    #return None
        
    # Fungsi untuk save data ke file
#def save_data(data):
    #data.to_pickle(DATA_FILE)

# Ganti dengan URL database Anda
#DATABASE_URL = st.secrets["database_url"]  # Simpan kredensial di Streamlit Secrets
#engine = create_engine(DATABASE_URL)

# Fungsi untuk menyimpan data ke database
#def save_data_to_db(df, table_name):
    #df.to_sql(table_name, engine, if_exists="append", index=False)

# Fungsi untuk memuat data dari database
#def load_data_from_db(table_name):
    #query = f"SELECT * FROM {table_name}"
    #try:
        #df = pd.read_sql(query, engine)
        #return df
    #except Exception as e:
        #st.error(f"Gagal memuat tabel {table_name}: {e}")
        #return pd.DataFrame()
DATA_FILE = "saved_data_baru.pkl"
        
    # Fungsi untuk load data dari file
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_pickle(DATA_FILE)
    return None
        
    # Fungsi untuk save data ke file
def save_data(data):
    data.to_pickle(DATA_FILE)

    # Cek apakah pengguna sudah login
PASSWORD = "admin1234"  # Ganti dengan password yang diinginkan
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    
     # Inisialisasi session state
if "data" not in st.session_state:
    st.session_state.data = load_data()
    
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
    st.success("Sudah login.")
    #st.title("Selamat Datang di Aplikasi")
        # Inisialisasi DataFrame untuk menyimpan data
    #if 'data' not in st.session_state:
            # Coba memuat data dari file CSV jika ada
        #try:
            #st.session_state.data = pd.read_csv("data_unggahan.csv")
            #st.success("Data berhasil dimuat dari file CSV.")
        #except FileNotFoundError:
            #st.session_state.data = pd.DataFrame()
    
        # Menu untuk memilih fungsi
    menu = st.sidebar.selectbox("Pilih Fungsi", [
        "Unggah Data",
        "Penganggaran dan Peramalan",
        "Statistik Customer",
        "Statistik Sales Kuantitas",
        "Statistik Sales Omsetnya",
        "Statistik E-commerce"
    ])

    if menu == "Unggah Data":
        st.subheader("Unggah Data dari File")
        
        # Tampilkan data yang sudah ada
        if st.session_state.data is not None:
            st.write("Data yang sudah diunggah (tersimpan):")
            st.dataframe(st.session_state.data)
        
        uploaded_files = st.file_uploader(
            "Pilih file CSV atau Excel (bisa lebih dari satu)",
            type=["csv", "xlsx"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            all_data = []
            
            for uploaded_file in uploaded_files:
                if uploaded_file.name.endswith('.csv'):
                    new_data = pd.read_csv(uploaded_file)
                else:
                    new_data = pd.read_excel(uploaded_file)
    
                # Jika kolom 'Tanggal' ada, konversi ke tipe datetime
                if 'Tanggal' in new_data.columns:
                    new_data['Tanggal'] = pd.to_datetime(new_data['Tanggal'])
                else:
                    st.warning(f"Kolom 'Tanggal' tidak ditemukan di {uploaded_file.name}. Data akan tetap diunggah tanpa kolom 'Tanggal'.")

                # Ambil nama tabel dari nama file
                table_name = uploaded_file.name.rsplit('.', 1)[0].lower()
        # Simpan data ke database
                save_data_to_db(new_data, table_name)
                st.success(f"Data dari {uploaded_file.name} berhasil disimpan ke tabel `{table_name}`.")
    
                # Tentukan nama tabel dari nama file
                #table_name = uploaded_file.name.split('.')[0].lower()  # nama tabel dari nama file
                
                # Simpan data ke database
                #new_data.to_sql(table_name, engine, if_exists="append", index=False)
                #all_data.append(new_data)
                #st.success(f"Data dari {uploaded_file.name} berhasil diunggah dan disimpan ke database ({table_name}).")
    
            if all_data:
                if st.session_state.data is not None:
                    st.session_state.data = pd.concat([st.session_state.data] + all_data, ignore_index=True)
                else:
                    st.session_state.data = pd.concat(all_data, ignore_index=True)
                
                save_data(st.session_state.data)
                st.success("Data berhasil diperbarui dan disimpan!")
                st.write("Data saat ini:")
                st.dataframe(st.session_state.data)

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
            forecast_months = st.number_input("Jumlah Bulan untuk Peramalan", min_value=1, max_value=12, value=3, key="forecast_months_input")
                
                # Pilih metode peramalan
            forecasting_method = st.selectbox("Pilih Metode Peramalan", ["ARIMA", "SARIMA", "Exponential Smoothing", "Moving Average", "Naive", "Prophet", "Average"], key="forecasting_method_input")
                
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
                customer_quantity = monthly_data[monthly_data['Pelanggan'] == customer].groupby('Bulan')['Kuantitas'].sum()  # Corrected to use customer
        
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
                customer_forecast_df_display = customer_forecast_df.copy()
                customer_forecast_df_display['Penjualan'] = customer_forecast_df_display['Penjualan'].apply(lambda x: f"{x:,.2f}")
                customer_forecast_df_display['Tanggal'] = customer_forecast_df_display['Tanggal'].dt.strftime('%d-%m-%y')  # Format tanggal
                st.dataframe(customer_forecast_df_display)
            
                # Ensure 'Tanggal' is in datetime format
            selected_customer = st.selectbox("Pilih Pelanggan untuk Melihat Grafik Peramalan:", customer_forecast_df['Pelanggan'].unique())
            selected_customer_data = customer_forecast_df[customer_forecast_df['Pelanggan'] == selected_customer]
                
                # Convert 'Tanggal' to datetime format
            selected_customer_data['Tanggal'] = pd.to_datetime(selected_customer_data['Tanggal'], errors='coerce')
                
                # Check for NaT values and handle them if necessary
            if selected_customer_data['Tanggal'].isnull().any():
                st.warning("Ada nilai yang tidak dapat dikonversi menjadi tanggal. Pastikan format tanggal benar.")
                
                # Ensure the data is sorted by 'Tanggal'
            selected_customer_data = selected_customer_data.sort_values(by='Tanggal')
                
                # Plot grafik
            fig_forecast = px.line(selected_customer_data, x='Tanggal', y='Kuantitas', markers=True,
                                       title=f'Perkembangan Kuantitas untuk {selected_customer}')
                
                # Update x-axis to show month and year
            fig_forecast.update_xaxes(
                tickvals=selected_customer_data['Tanggal'],
                ticktext=[date.strftime('%b %Y') for date in selected_customer_data['Tanggal'] if pd.notnull(date)],
                title_text="Bulan dan Tahun"
            )
                
            fig_forecast.update_layout(yaxis_title="Kuantitas", title_x=0.5)
            st.plotly_chart(fig_forecast)
    
                # Menghitung rata-rata penjualan dan kuantitas untuk setiap pelanggan
            average_customer_forecast_results = []
    
            for customer in customer_forecast_df['Pelanggan'].unique():
                customer_data = customer_forecast_df[customer_forecast_df['Pelanggan'] == customer]
    
                if not customer_data.empty:
                        # Menghapus tanda koma dan mengonversi ke float
                    customer_data['Penjualan'] = pd.to_numeric(customer_data['Penjualan'].astype(str).str.replace(',', ''), errors='coerce')
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
                    customer_data['Penjualan'] = pd.to_numeric(customer_data['Penjualan'].astype(str).str.replace(',', ''), errors='coerce')
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
    
                        #if not selected_products:
                            #st.warning("Silahkan pilih setidaknya satu nama barang untuk melihat grafik")
    
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
    
    elif menu == "Statistik Customer":
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
        
                # Hitung total kuantitas bulanan
            grouped = df.groupby(['Kota', 'Customer', pd.Grouper(key='Tanggal', freq='M')])['Kuantitas'].sum().reset_index()
        
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
        
                # Mengatur sumbu x untuk menampilkan semua bulan
            fig_all.update_xaxes(
                tickvals=df_kota['Tanggal'],  # Menggunakan semua tanggal yang ada
                ticktext=[date.strftime('%b %Y') for date in df_kota['Tanggal']],  # Format bulan dan tahun
                title_text="Bulan dan Tahun"  # Judul sumbu x
            )
                
            fig_all.update_layout(yaxis_title="Kuantitas", title_x=0.5)
            st.plotly_chart(fig_all)
        
                # Sidebar filter untuk memilih customer
            if not df_kota.empty:
                customer_terpilih = st.selectbox("Pilih Customer untuk Detail", sorted(df_kota['Customer'].unique()))
                df_cust = df_kota[df_kota['Customer'] == customer_terpilih]
        
                st.write(f"Grafik Perubahan Kuantitas - {customer_terpilih} di {kota_terpilih.capitalize()}")
        
                    # Membuat grafik interaktif untuk customer yang dipilih
                fig_cust = px.line(df_cust, x='Tanggal', y='Kuantitas', markers=True,
                                    title=f'Perkembangan Kuantitas {customer_terpilih}')
                    
                    # Mengatur sumbu x untuk menampilkan semua bulan
                fig_cust.update_xaxes(
                    tickvals=df_cust['Tanggal'],  # Menggunakan semua tanggal yang ada
                    ticktext=[date.strftime('%b %Y') for date in df_cust['Tanggal']],  # Format bulan dan tahun
                    title_text="Bulan dan Tahun"  # Judul sumbu x
                )
                    
                fig_cust.update_layout(yaxis_title="Kuantitas", title_x=0.5)
                st.plotly_chart(fig_cust)
        
                st.markdown("---")
                st.write("Data Rinci untuk Customer Terpilih")
                st.dataframe(df_cust[['Tanggal', 'Kuantitas', 'Status Perkembangan']], use_container_width=True)
        
        else:
            st.warning("Tidak ada data yang dimuat. Silakan pastikan data sudah tersedia di menu awal.")
                
        #sudah benar ini
    elif menu == "Statistik Sales Kuantitas":
        st.subheader("Statistik Perkembangan Sales Bulanan")            
        if not st.session_state.data.empty:
                #Pastikan Tanggal format datetime
            st.session_state.data['Tanggal'] = pd.to_datetime(st.session_state.data['Tanggal'], errors='coerce')
            st.session_state.data = st.session_state.data.dropna(subset=['Tanggal'])
        
                # Ekstrak Bulan
            st.session_state.data['Bulan'] = st.session_state.data['Tanggal'].dt.to_period('M').astype(str)
        
                # Agregasi total kuantitas dan penjualan per bulan, tenaga penjual, dan departemen
            monthly_sales = st.session_state.data.groupby(['Bulan', 'Nama Tenaga Penjual', 'Nama Departemen'])[['Kuantitas', 'Penjualan']].sum().reset_index()
        
                # Konversi Bulan ke datetime untuk pengurutan
            monthly_sales['Bulan'] = pd.to_datetime(monthly_sales['Bulan'])
            monthly_sales = monthly_sales.sort_values(by='Bulan')
            monthly_sales['Bulan'] = monthly_sales['Bulan'].dt.strftime('%Y-%m')
        
                # Tampilkan DataFrame ringkasan
            st.write("Ringkasan Penjualan Bulanan:")
            st.dataframe(monthly_sales)
        
                # Dropdown untuk memilih Tenaga Penjual
            semua_nama_tenaga_penjual = monthly_sales['Nama Tenaga Penjual'].unique()
            selected_nama_tenaga_penjual = st.selectbox("Pilih Nama Tenaga Penjual", semua_nama_tenaga_penjual)
        
                # Filter data berdasarkan tenaga penjual yang dipilih
            filtered_sales_by_salesperson = monthly_sales[monthly_sales['Nama Tenaga Penjual'] == selected_nama_tenaga_penjual]
        
                # Tampilkan statistik bulanan untuk tenaga penjual yang dipilih
            st.write(f"Statistik Penjualan Bulanan untuk {selected_nama_tenaga_penjual}:")
            st.dataframe(filtered_sales_by_salesperson)
        
                # Dropdown untuk memilih Barang berdasarkan tenaga penjual yang dipilih
            semua_nama_barang = st.session_state.data[st.session_state.data['Nama Tenaga Penjual'] == selected_nama_tenaga_penjual]['Nama Barang'].unique()
            selected_nama_barang = st.selectbox("Pilih Nama Barang", semua_nama_barang)
        
                # Filter data berdasarkan barang yang dipilih
            filtered_sales = st.session_state.data[(st.session_state.data['Nama Tenaga Penjual'] == selected_nama_tenaga_penjual) & 
                                                    (st.session_state.data['Nama Barang'] == selected_nama_barang)]
    
            monthly_product_sales = filtered_sales.groupby('Bulan')[['Kuantitas']].sum().reset_index()
    
                # Plot grafik Kuantitas untuk barang yang dipilih oleh tenaga penjual yang dipilih
            fig = px.line(monthly_product_sales, x='Bulan', y='Kuantitas', 
                            title=f'Tren Kuantitas Penjualan Bulanan untuk {selected_nama_barang} oleh {selected_nama_tenaga_penjual}',
                            markers=True,
                            labels={'Bulan': 'Bulan', 'Kuantitas': 'Total Kuantitas', })
    
            st.plotly_chart(fig, key='fig_salesperson_product') 
    
                # Tampilkan data rinci untuk barang yang dipilih
            st.write(f"Data Rinci untuk {selected_nama_barang} oleh {selected_nama_tenaga_penjual}:")
            st.dataframe(filtered_sales)
        else:
            st.warning("Tidak ada data yang dimuat. Silakan pastikan data sudah tersedia di menu awal.")   
                
    elif menu == "Statistik Sales Omsetnya":
        st.subheader("Statistik Perkembangan Sales Bulanan Omsetnya")
        
        if not st.session_state.data.empty:
            st.session_state.data['Tanggal'] = pd.to_datetime(st.session_state.data['Tanggal'], errors='coerce')
            st.session_state.data = st.session_state.data.dropna(subset=['Tanggal'])
        
                # Ekstrak Bulan
            st.session_state.data['Bulan'] = st.session_state.data['Tanggal'].dt.to_period('M').astype(str)
    
                # Agregasi total omset per bulan dan tenaga penjual
            monthly_revenue = st.session_state.data.groupby(['Bulan', 'Nama Tenaga Penjual'])[['Penjualan']].sum().reset_index()
        
                # Mengurutkan data berdasarkan Bulan dan Penjualan (dari tertinggi ke terendah)
            monthly_revenue = monthly_revenue.sort_values(by=['Bulan', 'Penjualan'], ascending=[True, False])
        
                # Tampilkan DataFrame ringkasan
            st.write("Ringkasan Omset Bulanan:")
            st.dataframe(monthly_revenue)
        
            # Dropdown untuk memilih Tenaga Penjual
            semua_nama_tenaga_penjual = monthly_revenue['Nama Tenaga Penjual'].unique()
            selected_nama_tenaga_penjual = st.selectbox("Pilih Nama Tenaga Penjual", semua_nama_tenaga_penjual)
        
                # Filter data berdasarkan tenaga penjual yang dipilih
            filtered_revenue_by_salesperson = monthly_revenue[monthly_revenue['Nama Tenaga Penjual'] == selected_nama_tenaga_penjual]
        
                # Tampilkan statistik bulanan untuk tenaga penjual yang dipilih
            st.write(f"Statistik Omset Bulanan untuk {selected_nama_tenaga_penjual}:")
            st.dataframe(filtered_revenue_by_salesperson)
                
                # Dropdown untuk memilih Barang berdasarkan tenaga penjual yang dipilih
            semua_nama_barang = st.session_state.data[st.session_state.data['Nama Tenaga Penjual'] == selected_nama_tenaga_penjual]['Nama Barang'].unique()
            selected_nama_barang = st.selectbox("Pilih Nama Barang", semua_nama_barang)
                
                # Filter data berdasarkan barang yang dipilih
            filtered_sales = st.session_state.data[(st.session_state.data['Nama Tenaga Penjual'] == selected_nama_tenaga_penjual) & 
                                                    (st.session_state.data['Nama Barang'] == selected_nama_barang)]
        
            monthly_product_revenue = filtered_sales.groupby('Bulan')[['Penjualan']].sum().reset_index()
        
                # Plot grafik Omset untuk barang yang dipilih oleh tenaga penjual yang dipilih
            fig = px.line(monthly_product_revenue, x='Bulan', y='Penjualan', 
                          title=f'Tren Omset Penjualan Bulanan untuk {selected_nama_barang} oleh {selected_nama_tenaga_penjual}',
                          markers=True,
                          labels={'Bulan': 'Bulan', 'Penjualan': 'Total Omset'})
        
            st.plotly_chart(fig, key='fig_salesperson_product_revenue') 
        
                # Tampilkan data rinci untuk barang yang dipilih
            st.write(f"Data Rinci untuk {selected_nama_barang} oleh {selected_nama_tenaga_penjual}:")
            st.dataframe(filtered_sales[['Tanggal', 'Penjualan', 'Kuantitas']])  # Display relevant columns
                
            overall_monthly_revenue = st.session_state.data.groupby(['Bulan', 'Nama Tenaga Penjual'])[['Penjualan']].sum().reset_index()
                
                # Tampilkan tabel omset keseluruhan
            st.write("Tabel Omset Keseluruhan per Bulan untuk Setiap Tenaga Penjual:")
            st.dataframe(overall_monthly_revenue)
        
                # Plot grafik omset keseluruhan
            overall_fig = px.line(overall_monthly_revenue, x='Bulan', y='Penjualan', color='Nama Tenaga Penjual',
                                    title='Tren Omset Keseluruhan per Bulan untuk Setiap Tenaga Penjual ',
                                    markers=True,
                                    labels={'Bulan': 'Bulan', 'Penjualan': 'Total Omset'})
                
            st.plotly_chart(overall_fig, key='fig_overall_revenue') 
        
                # Dropdown untuk memilih Tenaga Penjual untuk omset keseluruhan
            selected_nama_tenaga_penjual_overall = st.selectbox("Pilih Nama Tenaga Penjual untuk Omset Keseluruhan", semua_nama_tenaga_penjual)
        
                # Filter data untuk omset keseluruhan berdasarkan tenaga penjual yang dipilih
            overall_filtered_revenue = overall_monthly_revenue[overall_monthly_revenue['Nama Tenaga Penjual'] == selected_nama_tenaga_penjual_overall]
        
                # Plot grafik omset keseluruhan untuk tenaga penjual yang dipilih
            overall_selected_fig = px.line(overall_filtered_revenue, x='Bulan', y='Penjualan',
                                            title=f'Tren Omset Keseluruhan Bulanan untuk {selected_nama_tenaga_penjual_overall}',
                                            markers=True,
                                            labels={'Bulan': 'Bulan', 'Penjualan': 'Total Omset'})
        
            st.plotly_chart(overall_selected_fig, key='fig_selected_overall_revenue')
    
            st.write(f"Tabel Omset Keseluruhan untuk {selected_nama_tenaga_penjual_overall}:")
            st.dataframe(overall_filtered_revenue)
        else:
            st.warning("Tidak ada data yang dimuat. Silakan Pastikan data sudah tersedia di menu awal.")
    
    elif menu == "Statistik E-commerce":
        st.subheader("Statistik E-commerce")
        
        if not st.session_state.data.empty:
            ecommerce_data = st.session_state.data.copy()
        
            required_cols = ['Nama Barang Lazada', 'Nama Barang Shopee', 'Nama Barang Tokopedia', 'Kota', 'Harga', 'Berapa Terjual']
            missing_cols = [col for col in required_cols if col not in ecommerce_data.columns]
            if missing_cols:
                st.error(f"Data e-commerce tidak memiliki kolom yang diperlukan: {missing_cols}")
        
            st.write("Masukkan kata kunci untuk cari nama barang:")
            selected_keyword = st.text_input("Kata Kunci").strip().lower()
        
            if not selected_keyword:
                st.warning("Silakan masukkan kata kunci untuk menampilkan data.")
            else:
                keywords = selected_keyword.split()
        
                    # Filter baris yang mengandung semua keyword di salah satu kolom nama barang
                mask_lazada = ecommerce_data['Nama Barang Lazada'].apply(
                    lambda x: all(k in x.lower() for k in keywords) if isinstance(x, str) else False
                )
                mask_shopee = ecommerce_data['Nama Barang Shopee'].apply(
                    lambda x: all(k in x.lower() for k in keywords) if isinstance(x, str) else False
                )
                mask_tokopedia = ecommerce_data['Nama Barang Tokopedia'].apply(
                    lambda x: all(k in x.lower() for k in keywords) if isinstance(x, str) else False
                )
        
                mask_contains = mask_lazada | mask_shopee | mask_tokopedia
                filtered_data = ecommerce_data[mask_contains]
    
                if not filtered_data.empty:
                    filtered_data = filtered_data[required_cols]
        
                        # Konversi kolom Harga ke float, mengganti non-numeric dengan NaN
                    filtered_data['Harga'] = (
                        filtered_data['Harga']
                        .astype(str)
                        .str.replace(r'[^0-9]', '', regex=True)  # Remove non-numeric characters
                    )
        
                        # Convert to numeric, coercing errors to NaN
                    filtered_data['Harga'] = pd.to_numeric(filtered_data['Harga'], errors='coerce')
        
                        # Fill NaN values with 0 or drop rows with NaN in 'Harga'
                    filtered_data['Harga'] = filtered_data['Harga'].fillna(0)  # Fill NaN with 0
        
                    st.write(f"Data yang mengandung kata kunci '{selected_keyword}':")
                    st.dataframe(filtered_data)
        
                        # Calculate min and max prices
                    min_harga = filtered_data['Harga'].min()
                    max_harga = filtered_data['Harga'].max()
                    st.success(f"Harga Terendah: Rp{min_harga:,.0f}")
                    st.success(f"Harga Tertinggi: Rp{max_harga:,.0f}")
        
                    labels = []
                    sizes = []
                    colors = []
        
                    if filtered_data['Nama Barang Tokopedia'].dropna().any():
                        labels.append('Tokopedia')
                        sizes.append(filtered_data['Nama Barang Tokopedia'].notna().sum())
                        colors.append('#40B349')
        
                    if filtered_data['Nama Barang Lazada'].dropna().any():
                        labels.append('Lazada')
                        sizes.append(filtered_data['Nama Barang Lazada'].notna().sum())
                        colors.append('#F00000')
        
                    if filtered_data['Nama Barang Shopee'].dropna().any():
                        labels.append('Shopee')
                        sizes.append(filtered_data['Nama Barang Shopee'].notna().sum())
                        colors.append('#FF5722')
        
                        # Pie chart estetik
                    fig, ax = plt.subplots()
                    wedges, texts, autotexts = ax.pie(
                        sizes,
                        labels=labels,
                        colors=colors,
                        autopct='%1.1f%%',
                        startangle=90,
                        wedgeprops=dict(width=0.4, edgecolor='white'),
                        textprops={'fontsize': 12, 'color': 'black'}
                    )
        
                    ax.legend(
                        wedges,
                        labels,
                        title="Platform",
                        loc="center left",
                        bbox_to_anchor=(1, 0, 0.5, 1),
                        fontsize=12
                    )
        
                    ax.set_title("Produk per E-commerce", fontsize=14, weight='bold')
                    ax.axis('equal')
                    st.pyplot(fig)
        
                        # Ensure all plot outputs update in place
                    placeholder_tokopedia = st.empty()
                    placeholder_lazada = st.empty()
                    placeholder_shopee = st.empty()
        
                        # Tokopedia
                    df_tokopedia = filtered_data[['Nama Barang Tokopedia', 'Harga']].dropna()
                    df_tokopedia = df_tokopedia.rename(columns={'Nama Barang Tokopedia': 'Nama Barang'})
                    if not df_tokopedia.empty:
                        df_tokopedia = df_tokopedia.sort_values(by='Harga').reset_index(drop=True)
                        df_tokopedia['Nama Singkat'] = df_tokopedia['Nama Barang'].str.slice(0, 20) + '...'  # disingkat untuk sumbu X
        
                        fig_tokopedia = px.line(
                            df_tokopedia,
                            x='Nama Singkat',
                            y='Harga',
                            markers=True,
                            title='Tren Harga Produk Tokopedia',
                            hover_name='Nama Barang',
                            labels={'Nama Singkat': 'Produk', 'Harga': 'Harga (Rp)'},
                            color_discrete_sequence=['#40B349']
                        )
                        fig_tokopedia.update_layout(xaxis_title="Nama Produk", yaxis_title="Harga (Rp)")
                        placeholder_tokopedia.plotly_chart(fig_tokopedia)
        
                            # Display min and max prices for Tokopedia
                        min_tokopedia = df_tokopedia['Harga'].min()
                        max_tokopedia = df_tokopedia['Harga'].max()
                        st.write(f"Harga Terendah Tokopedia: Rp{min_tokopedia:,.0f}, Harga Tertinggi Tokopedia: Rp{max_tokopedia:,.0f}")
        
                    else:
                        placeholder_tokopedia.info("Tidak ada data Tokopedia untuk kata kunci ini.")
        
                        # Lazada
                    df_lazada = filtered_data[['Nama Barang Lazada', 'Harga']].dropna()
                    df_lazada = df_lazada.rename(columns={'Nama Barang Lazada': 'Nama Barang'})
                    if not df_lazada.empty:
                        df_lazada = df_lazada.sort_values(by='Harga').reset_index(drop=True)
                        df_lazada['Nama Singkat'] = df_lazada['Nama Barang'].str.slice(0, 20) + '...'
        
                        fig_lazada = px.line(
                            df_lazada,
                            x='Nama Singkat',
                            y='Harga',
                            markers=True,
                            title='Tren Harga Produk Lazada',
                            hover_name='Nama Barang',
                            labels={'Nama Singkat': 'Produk', 'Harga': 'Harga (Rp)'},
                            color_discrete_sequence=['#F00000']
                        )
                        fig_lazada.update_layout(xaxis_title="Nama Produk", yaxis_title="Harga (Rp)")
                        placeholder_lazada.plotly_chart(fig_lazada)
    
                            # Display min and max prices for Lazada
                        min_lazada = df_lazada['Harga'].min()
                        max_lazada = df_lazada['Harga'].max()
                        st.write(f"Harga Terendah Lazada: Rp{min_lazada:,.0f}, Harga Tertinggi Lazada: Rp{max_lazada:,.0f}")
    
                    else:
                        placeholder_lazada.info("Tidak ada data Lazada untuk kata kunci ini.")
        
                        # Shopee
                    df_shopee = filtered_data[['Nama Barang Shopee', 'Harga']].dropna()
                    df_shopee = df_shopee.rename(columns={'Nama Barang Shopee': 'Nama Barang'})
                    if not df_shopee.empty:
                        df_shopee = df_shopee.sort_values(by='Harga').reset_index(drop=True)
                        df_shopee['Nama Singkat'] = df_shopee['Nama Barang'].str.slice(0, 20) + '...'
        
                        fig_shopee = px.line(
                            df_shopee,
                            x='Nama Singkat',
                            y='Harga',
                            markers=True,
                            title='Tren Harga Produk Shopee',
                            hover_name='Nama Barang',
                            labels={'Nama Singkat': 'Produk', 'Harga': 'Harga (Rp)'},
                            color_discrete_sequence=['#FF5722']
                        )
                        fig_shopee.update_layout(xaxis_title="Nama Produk", yaxis_title="Harga (Rp)")
                        placeholder_shopee.plotly_chart(fig_shopee)
        
                            # Display min and max prices for Shopee
                        min_shopee = df_shopee['Harga'].min()
                        max_shopee = df_shopee['Harga'].max()
                        st.write(f"Harga Terendah Shopee: Rp{min_shopee:,.0f}, Harga Tertinggi Shopee: Rp{max_shopee:,.0f}")
        
                    else:
                        placeholder_shopee.info("Tidak ada data Shopee untuk kata kunci ini.")
                        
                        #YANG BENER
                        # Pastikan kolom string 
                    filtered_data['Kota'] = filtered_data['Kota'].astype(str)
                    filtered_data['Berapa Terjual'] = filtered_data['Berapa Terjual'].astype(str)
                        
                    filtered_data['Jumlah Terjual'] = filtered_data['Berapa Terjual'].apply(
                        lambda x: (
                            float(re.findall(r'[\d,]+', str(x))[0].replace(',', '.')) * 1000
                            if 'rb' in str(x).lower()
                            else float(re.findall(r'\d+', str(x))[0])
                            if re.findall(r'\d+', str(x))
                            else 0.0
                        )
                    )
    
                        # Ubah NaN jadi 0 agar kota tetap dihitung
                    filtered_data['Jumlah Terjual'] = filtered_data['Jumlah Terjual'].fillna(0)
                        
                        # Groupby tetap semua kota
                    city_sales = filtered_data.groupby('Kota')['Jumlah Terjual'].sum().reset_index()
                        
                        # Pastikan urut berdasarkan jumlah terjual
                    city_sales = city_sales.sort_values(by='Jumlah Terjual', ascending=False)
                        
                    st.write("Jumlah Terjual berdasarkan kota")
                    st.dataframe(city_sales)
                        
                        # Membuat grafik bar yang jelas dengan plotly express
                    fig = px.bar(
                        city_sales,
                        x='Kota',
                        y='Jumlah Terjual',
                        title=f"Total Produk Terjual per Kota untuk '{selected_keyword}'",
                        labels={'Jumlah Terjual': 'Jumlah Produk Terjual', 'Kota': 'Kota'},
                        text='Jumlah Terjual',
                        color='Jumlah Terjual',
                        color_continuous_scale=px.colors.sequential.Blues,
                        height=500,  # Lebih tinggi agar lebih lega
                        width=900   # Lebar lebar agar tidak sempit
                    )
                        
                        # Memperjelas tampilan angka pada bar
                    fig.update_traces(
                        texttemplate='%{text:.0f}',
                        textposition='outside',
                        marker_line_width=1.5,  # Garis batas bar
                        marker_line_color='rgba(0,0,0,0.6)'  # Warna garis batas bar
                    )
                        
                        # Set layout untuk memperjelas axis dan tema
                    fig.update_layout(
                        xaxis_title='Kota',
                        yaxis_title='Jumlah Produk Terjual',
                        xaxis_tickangle=-45,
                        yaxis=dict(range=[0, city_sales['Jumlah Terjual'].max() + city_sales['Jumlah Terjual'].max()*0.1]),
                        template='plotly_white',
                        font=dict(size=14, color='black'),  # Ukuran dan warna font keseluruhan
                        margin=dict(t=60, b=150, l=70, r=40),  # Margin untuk label x yang berputar
                        coloraxis_colorbar=dict(
                            title="Jumlah Terjual",
                            ticks="outside",
                            ticklen=5,
                            outlinewidth=1
                        )
                    )
                        
                        # Tampilkan grafik pada Streamlit
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"Kata kunci '{selected_keyword}' tidak ditemukan dalam data.")
    
        # Menambahkan opsi untuk mereset seluruh chat history
    if st.sidebar.button("Reset Chat History"):
        st.session_state.messages = []  # Clear the chat history
        st.success("Chat history telah direset. Anda dapat memulai percakapan baru.")
    
        # Panggil fungsi untuk menyimpan data ke database
    if st.button("Simpan Data ke Database"):
        save_data_to_database(st.session_state.data)
        st.success("Data telah disimpan ke database.")
    
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

    #import streamlit.web.cli as stcli
    #import sys
    
    #if __name__ == '__main__':
        #sys.argv = ["streamlit", "run", "app.py"]  # Ganti "app.py" dengan nama file Anda jika berbeda
        #sys.exit(stcli.main())
