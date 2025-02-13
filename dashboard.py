#library
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly_express as px

from statsmodels.tsa.stattools import adfuller
from scipy.stats import boxcox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import kstest_normal
from scipy.stats import shapiro
from statsmodels.tsa import stattools as tsa

#streamlit
import streamlit as st

def header():
    st.title(":bar_chart: Time Series Forecasting using ARIMA")
    st.write("This is a web application that uses the ARIMA model to forecast time series data.")
    st.link_button("Hendri Agustono on LinkedIn", "https://linkedin.com/in/hendri-agustono")
    st.link_button("Bhima Fairul Rifqi on LinkedIn", "https://linkedin.com/in/fairulrifqi962")
    st.divider()

def upload():
    def plot(df):
        if not isinstance(df, pd.Series):
            if isinstance(df, pd.DataFrame) and len(df.columns) == 1:
                df = df.iloc[:, 0]
            else:
                st.error("Input must be a DataFrame with one column.")
                return  # Exit if input is invalid

        date = df.index
        fig = go.Figure(data=go.Scatter(x=date, y=df, mode='lines+markers'))
        
        mean = df.mean()

        fig.add_shape(
            type="line",
            x0=date.min(),
            y0=mean,
            x1=date.max(),
            y1=mean,
            line=dict(color="red", width=2, dash="dash"),
        )

        fig.update_layout(
            title="Plot time series",
            xaxis_title="Index/Date",
            yaxis_title="Value"
        )
        st.plotly_chart(fig, use_container_width=True)

    def acf_plot(transformed_data):
        # Calculate the ACF
        acf, confint = tsa.acf(transformed_data, alpha=0.01, nlags=20)

        # Significance level (used for dashed lines) - it looks like the original uses a different level than 0.05
        sig_level = 0.05  # Adjust as needed to visually match the original plot

        # Calculate confidence interval bounds based on the approximate formula
        confint_approx = np.array([[-1.96 / np.sqrt(len(transformed_data))] * (len(acf)), [1.96 / np.sqrt(len(transformed_data))] * len(acf)]) * (1-sig_level/2)


        # Create Plotly figure
        fig = go.Figure()

        # ACF bars
        fig.add_trace(go.Bar(
            x=np.arange(1, len(acf)),  # Start at 1 to match the original plot
            y=acf[1:],  # Start at 1 to match the original plot
            marker_color='green'  # Match the black bars of the original plot
        ))

        # Significance lines (dashed)
        fig.add_hline(y=confint_approx[1,0], line_width=1, line_dash="dash", line_color="white") # manually set the confidence intervals
        fig.add_hline(y=confint_approx[0,0], line_width=1, line_dash="dash", line_color="white")



        # Customize layout
        fig.update_layout(
            title="ACF Plot",
            xaxis_title="Lag",
            yaxis_title="ACF Value",
            template="simple_white" # Set to a simpler template to match the original style more closely
        )
        st.plotly_chart(fig, use_container_width=True)
        
    def pacf_plot(transformed_data):
        pacf, confint = tsa.pacf(transformed_data, alpha=0.01, nlags=20)

        sig_level = 0.05
        confint_approx = np.array([[-1.96 / np.sqrt(len(transformed_data))] * (len(pacf)), [1.96 / np.sqrt(len(transformed_data))] * len(pacf)])* (1-sig_level/2)

        fig = go.Figure()

        # PACF bars
        fig.add_trace(go.Bar(
            x=np.arange(1, len(pacf)),  # Start at lag 1
            y=pacf[1:],  # Start at lag 1
            marker_color='blue'
        ))

        # Significance lines (dashed)

        fig.add_hline(y=confint_approx[1,0], line_width=1, line_dash="dash", line_color="white")
        fig.add_hline(y=confint_approx[0,0], line_width=1, line_dash="dash", line_color="white")


        # Customize layout
        fig.update_layout(
            title="PACF Plot",
            xaxis_title="Lag",
            yaxis_title="PACF Value",
            template="simple_white"  # Use a simple white template
        )

        st.plotly_chart(fig, use_container_width=True)

    def arima_x(df, data):
        def arima_manual(transformed_data):
            model = ARIMA(transformed_data, order=(2,1,0))
            model_fit = model.fit()
            st.write(model_fit.summary())

        def adf(df, data):
            # adfuller
            adf_test = adfuller(df, regression="ct")
            st.write('ADF Statistic: %f' % adf_test[0])
            st.write('p-value: %f' % adf_test[1])
            # cek stastioner rataan boxcox
            transformed_data, lamda = boxcox(data[df.columns[0]])
            st.write('lamda boxcox before transform: ', lamda)

            if lamda == 0:
                transform =  np.log(data[df.columns[0]])
            else:
                transform =  (data[df.columns[0]]**lamda - 1) / lamda
            # boxcox once again
            transformed_data, lamda = boxcox(transform)
            transformed_one = transformed_data
            st.write('lamda boxcox after transform: ', lamda)
            st.divider()
            st.title("Data After Transformation")
            transformed_data = pd.DataFrame(transformed_data)
            st.dataframe(transformed_data, use_container_width=True)
            # adf
            adf_test = adfuller(transformed_data, regression="ct")
            st.write('ADF Statistic: %f' % adf_test[0])
            st.write('p-value: %f' % adf_test[1])

            if adf_test[1] > 0.05:
                transformed_data = transformed_data.diff()
                transformed_data.dropna(inplace=True)
                st.title("Data After Differencing")
                st.dataframe(transformed_data, use_container_width=True)
                adf_test = adfuller(transformed_data, regression="ct")
                st.write('ADF Statistic: %f' % adf_test[0])
                st.write('p-value: %f' % adf_test[1])

            st.write("Mean:",transformed_data.mean())
            plot(transformed_data[0])
            acf_plot(transformed_data[0])
            pacf_plot(transformed_data[0])
            arima_manual(transformed_one)
        adf(df, data)

    dataframe = st.file_uploader(
       "Select excel or csv file", accept_multiple_files= False, type=['xlsx', 'csv']
    )
    if dataframe is not None:
        st.title("Preview Data")
        st.divider()
        if dataframe.type == 'text/csv':
            df = pd.read_csv(dataframe)
            st.dataframe(df, use_container_width=True)
        else:
            df = pd.read_excel(dataframe)
            st.dataframe(df, use_container_width=True)

        st.write("Select one column for ARIMA:")
        arima_df = list(df.select_dtypes(exclude=['object']))
        #dataframe output
        arima = []
        st.write("Dataframe:")
        for col in arima_df:
            if st.checkbox(str(col), value=True):
                arima.append(col)
        st.divider()
        st.dataframe(df[arima], use_container_width=True)
        # plot
        plot(df[arima])

        # uji statisik \\ button trigger
        if "clicked" not in st.session_state:
            st.session_state["clicked"] = False

        def onSearch(opt):
            st.session_state["clicked"] = True

        def drawBtn():
            option= ...
            st.button("Diagnostic", on_click= onSearch, args= [option])
        drawBtn()

        if st.session_state["clicked"]:
            arima_x(df[arima], df)

header()
upload()