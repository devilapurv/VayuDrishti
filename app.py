import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title='VayuDrishti | Predictive Air Quality & Environmental Intelligence', page_icon='🌫️', layout='wide', initial_sidebar_state='expanded')

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
html,body,[class*="css"],.stApp{font-family:'DM Sans',sans-serif}
.stApp{background:radial-gradient(ellipse at 75% 0%,#173452 0%,#090e1c 39%,#080c16 85%);color:#eaf3ff}
[data-testid="stSidebar"]{background:#0c1425;border-right:1px solid #20344c}
[data-testid="stSidebar"] *{color:#dbe9fa}
h1,h2,h3{font-family:'Space Grotesk',sans-serif;letter-spacing:-.035em}
h1{font-weight:700!important}
.hero{background:linear-gradient(110deg,rgba(12,30,51,.96),rgba(14,60,79,.58));border:1px solid #28536b;border-radius:23px;padding:33px 38px;margin:10px 0 22px;box-shadow:0 16px 50px #0005}
.eyebrow{color:#58e5d2;font-size:12px;letter-spacing:3px;font-weight:700}
.hero h1{font-size:clamp(35px,5vw,61px);line-height:1.08;margin:14px 0;color:#f5fbff}
.hero p{color:#a9c5d7;font-size:16px;max-width:850px}
.pill{display:inline-block;border:1px solid #2b6a77;border-radius:30px;color:#8af1de;padding:7px 13px;font-size:12px;margin-top:8px}
[data-testid="stMetric"]{background:#101e30;border:1px solid #244057;border-radius:17px;padding:15px 18px}
[data-testid="stMetricLabel"]{color:#9ab4c8}
[data-testid="stMetricValue"]{color:#ecfaff}
.stButton>button[kind="primary"]{background:linear-gradient(90deg,#14b8a6,#168cb3);border:0}
[data-testid="stDataFrame"]{border-radius:12px;overflow:hidden}
.block-container{padding-top:1.4rem;padding-bottom:4rem}
.small-note{color:#92aabe;font-size:13px;margin:12px 0}
</style>''', unsafe_allow_html=True)

PLOT_BG='#101a2a'; GRID='#27364b'
def plot_style(fig, height=390):
    fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(color='#c9d9ea'),height=height,margin=dict(l=12,r=15,t=48,b=22),legend=dict(orientation='h',y=-.25),hovermode='x unified')
    fig.update_xaxes(showgrid=False,zeroline=False)
    fig.update_yaxes(gridcolor=GRID,zeroline=False)
    return fig

def category(x):
    if pd.isna(x): return 'Unknown'
    if x<=50:return 'Good'
    if x<=100:return 'Satisfactory'
    if x<=200:return 'Moderate'
    if x<=300:return 'Poor'
    if x<=400:return 'Very Poor'
    return 'Severe'

def hero(kicker,title,desc):
    st.markdown(f'<div class="hero"><div class="eyebrow">{kicker}</div><h1>{title}</h1><p>{desc}</p><span class="pill">DELHI • HISTORICAL MODEL • 2015–2020</span></div>',unsafe_allow_html=True)

@st.cache_resource
def models():
    from xgboost import XGBRegressor
    from sklearn.pipeline import make_pipeline

    model_dir = BASE / 'models'
    with open(model_dir / 'airguard_features.json', encoding='utf-8') as f:
        feats = json.load(f)

    imputer = joblib.load(model_dir / 'airguard_imputer.pkl')
    xgb_model = XGBRegressor()
    xgb_model.load_model(str(model_dir / 'airguard_xgboost.json'))
    forecast_model = make_pipeline(imputer, xgb_model)

    anomaly_model = joblib.load(model_dir / 'airguard_anomaly_pipeline.pkl')
    return forecast_model, anomaly_model, feats

@st.cache_data
def stored():
    f=pd.read_csv(BASE/'data/airguard_forecasts.csv',parse_dates=['Date'])
    f['Forecast_Date']=f['Date']+pd.Timedelta(days=1)
    a=pd.read_csv(BASE/'data/airguard_anomalies.csv',parse_dates=['Date'])
    return f,a

try:
    forecast_model,anomaly_model,FEATURES=models()
except Exception as e:
    st.error('Could not load trained models. Place airguard_xgboost.json and airguard_imputer.pkl inside the models folder. If the anomaly pipeline fails to load, check scikit-learn version compatibility.')
    st.exception(e)
    st.stop()
f,a=stored()

st.sidebar.markdown('## ◈ VAYUDRISHTI')
st.sidebar.caption('PREDICTIVE AIR QUALITY & ENVIRONMENTAL INTELLIGENCE')
page=st.sidebar.radio('EXPLORE', ['Command Center','Forecast Lab','Anomaly Radar','Scenario Studio','Seasonal Intelligence','Explainable AI'], label_visibility='collapsed')
st.sidebar.divider()
st.sidebar.caption('OPTIONAL • FULL HISTORICAL INPUT DATA')
upload=st.sidebar.file_uploader('Upload city_day.csv',type=['csv'],help='Enables real historical feature selection and monthly pollution patterns.')
@st.cache_data
def load_history(blob):
    from io import BytesIO
    d=pd.read_csv(BytesIO(blob),parse_dates=['Date'])
    d=d[d['City'].eq('Delhi')].sort_values('Date').reset_index(drop=True)
    for lag in [1,2,3]:d[f'AQI_lag{lag}']=d['AQI'].shift(lag)
    d['AQI_rolling7']=d['AQI'].shift(1).rolling(7).mean()
    d['Month']=d['Date'].dt.month;d['DayOfWeek']=d['Date'].dt.dayofweek
    return d
hist=None
try:
    if upload: hist=load_history(upload.getvalue())
    elif (BASE/'data/city_day.csv').exists(): hist=load_history((BASE/'data/city_day.csv').read_bytes())
except Exception as e: st.sidebar.warning(f'Historical file could not be read: {e}')
st.sidebar.markdown('<div class="small-note">Predict Pollution. Decode Patterns. Protect Tomorrow. • Historical research prototype, not live data. Use only trusted model files.</div>',unsafe_allow_html=True)

# Manual input / historical selection: model needs all 13 features.
def input_panel(prefix):
    defaults={'PM2.5':95.0,'PM10':190.0,'NO2':35.0,'CO':1.4,'SO2':12.0,'O3':30.0,'AQI':220.0,'AQI_lag1':210.0,'AQI_lag2':205.0,'AQI_lag3':215.0,'AQI_rolling7':212.0,'Month':11,'DayOfWeek':2}
    if hist is not None:
        choices=hist.dropna(subset=['AQI']).copy()
        idx=st.selectbox('Historical observation date',options=list(choices.index),format_func=lambda i: str(choices.loc[i,'Date'].date()),index=len(choices)-1,key=prefix+'_date')
        sample=choices.loc[idx,FEATURES]
        defaults.update({k:float(v) for k,v in sample.items() if pd.notna(v)})
        st.caption('Historical measurements are used as model inputs; lag features come from previous calendar days.')
    else:
        st.info('Upload city_day.csv in the sidebar to populate historical observations. Otherwise, use manual illustrative inputs below.')
    with st.expander('Adjust all 13 model inputs',expanded=(hist is None)):
        cols=st.columns(3)
        for i,key in enumerate(FEATURES):
            with cols[i%3]:
                if key in ['Month','DayOfWeek']:
                    defaults[key]=st.number_input(key,min_value=1 if key=='Month' else 0,max_value=12 if key=='Month' else 6,value=int(defaults[key]),step=1,key=prefix+key)
                else:
                    defaults[key]=st.number_input(key,min_value=0.,value=float(defaults[key]),step=1.,format='%.2f',key=prefix+key)
    return pd.DataFrame([defaults])[FEATURES]

if page=='Command Center':
    hero('MISSION CONTROL / 01','Breathe data. See tomorrow.','A historical Delhi air-quality intelligence platform combining real trained machine learning, anomaly discovery, and interpretable forecasts.')
    actual=f['Target_AQI'];pred=f['Predicted_AQI']
    mae=np.mean(np.abs(actual-pred));rmse=np.sqrt(np.mean((actual-pred)**2));r2=1-np.sum((actual-pred)**2)/np.sum((actual-actual.mean())**2)
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Forecast MAE',f'{mae:.2f} AQI')
    c2.metric('Forecast RMSE',f'{rmse:.2f}')
    c3.metric('Model R²',f'{r2:.2f}')
    c4.metric('Historical anomalies',str(int((a['Anomaly']==-1).sum())))
    st.subheader('Forecast signal / actual vs predicted')
    span=st.select_slider('Display last N test observations',options=[30,60,120,240,len(f)],value=120)
    part=f.tail(span)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=part.Forecast_Date,y=part.Target_AQI,name='Actual AQI',line=dict(color='#5eead4',width=2)))
    fig.add_trace(go.Scatter(x=part.Forecast_Date,y=part.Predicted_AQI,name='Predicted AQI',line=dict(color='#a7a2ff',width=2)))
    st.plotly_chart(plot_style(fig,430),use_container_width=True)
    st.caption('All dates represent the target forecast day, not the source observation day. Test period is historical; no live predictions are shown.')
    st.subheader('Prediction quality')
    q1,q2=st.columns(2)
    with q1:
        fig=px.scatter(f,x='Target_AQI',y='Predicted_AQI',opacity=.65,labels={'Target_AQI':'Actual AQI','Predicted_AQI':'Predicted AQI'},color_discrete_sequence=['#57d9d0'])
        mn=float(min(f.Target_AQI.min(),f.Predicted_AQI.min()));mx=float(max(f.Target_AQI.max(),f.Predicted_AQI.max()))
        fig.add_trace(go.Scatter(x=[mn,mx],y=[mn,mx],mode='lines',name='Perfect forecast',line=dict(color='#e3a4c7',dash='dash')))
        st.plotly_chart(plot_style(fig),use_container_width=True)
    with q2:
        counts=f.Predicted_AQI.map(category).value_counts()
        fig=px.bar(x=counts.index,y=counts.values,labels={'x':'Predicted category','y':'Observations'},color=counts.values,color_continuous_scale=['#226d83','#59ddcf'])
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(plot_style(fig),use_container_width=True)

elif page=='Forecast Lab':
    hero('PREDICTIVE LAB / 02','Tomorrow, modeled.','Explore one-day-ahead AQI predictions using the exact XGBoost pipeline trained in Google Colab.')
    x=input_panel('forecast_')
    if st.button('Generate AQI forecast',type='primary'):
        y=float(forecast_model.predict(x)[0]);st.session_state['last_forecast']=(y,x.copy())
    if 'last_forecast' in st.session_state:
        y,_=st.session_state['last_forecast']
        c1,c2,c3=st.columns(3)
        c1.metric('Predicted next-day AQI',f'{y:.1f}')
        c2.metric('Indian AQI category',category(y))
        c3.metric('Threshold alert','HIGH' if y>300 else 'STANDARD')
        st.info('This prediction is conditional on the supplied historical or manual measurements. It is not a current live forecast.')
    st.subheader('Model inputs')
    st.dataframe(x,use_container_width=True,hide_index=True)

elif page=='Anomaly Radar':
    hero('ANOMALY RADAR / 03','Find the unexpected.','Isolation Forest highlights unusual combinations of pollutants in historical Delhi observations.')
    flagged=a[a.Anomaly.eq(-1)].sort_values('Anomaly_Score')
    c1,c2,c3=st.columns(3)
    c1.metric('Test observations',len(a));c2.metric('Flagged anomalies',len(flagged));c3.metric('Flagged rate',f'{100*len(flagged)/len(a):.1f}%')
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=a.Date,y=a.AQI,mode='lines',name='AQI',line=dict(color='#67d8e0')))
    fig.add_trace(go.Scatter(x=flagged.Date,y=flagged.AQI,mode='markers',name='Anomaly',marker=dict(color='#ff8c91',size=11,symbol='diamond')))
    st.plotly_chart(plot_style(fig,420),use_container_width=True)
    st.dataframe(flagged[['Date','AQI','PM2.5','Anomaly_Score']],use_container_width=True,hide_index=True)
    st.subheader('Check a new observation')
    st.caption('Anomaly detection uses seven pollutant features; it does not prove the cause of an unusual reading.')
    cols=st.columns(4)
    af=['AQI','PM2.5','PM10','NO2','CO','SO2','O3']
    vals={}
    for i,k in enumerate(af):
        with cols[i%4]: vals[k]=st.number_input(k,value=float({'AQI':220,'PM2.5':95,'PM10':190,'NO2':35,'CO':1.4,'SO2':12,'O3':30}[k]),min_value=0.,key='anom_'+k)
    if st.button('Analyze observation',type='primary'):
        xx=pd.DataFrame([vals])[af]
        flag=int(anomaly_model.predict(xx)[0]);score=float(anomaly_model.decision_function(xx)[0])
        st.metric('Anomaly result','UNUSUAL' if flag==-1 else 'TYPICAL')
        st.caption(f'Isolation Forest decision score: {score:.4f} (negative typically indicates an outlier).')

elif page=='Scenario Studio':
    hero('SCENARIO STUDIO / 04','Change the inputs. Explore the outcome.','A controlled sensitivity experiment: vary pollutant inputs and compare model forecasts. This is not a causal intervention estimate.')
    x=input_panel('scenario_')
    st.subheader('Scenario controls')
    cs=st.columns(3)
    adjust={}
    for i,k in enumerate(['PM2.5','PM10','NO2','O3','CO','SO2']):
        with cs[i%3]:adjust[k]=st.slider(f'{k} change (%)',-50,100,0,5,key='adj_'+k)
    changed=x.copy()
    for k,v in adjust.items():changed[k]=changed[k]*(1+v/100)
    original=float(forecast_model.predict(x)[0]);scenario=float(forecast_model.predict(changed)[0])
    c1,c2,c3=st.columns(3)
    c1.metric('Original prediction',f'{original:.1f}')
    c2.metric('Scenario prediction',f'{scenario:.1f}',delta=f'{scenario-original:+.1f} AQI',delta_color='inverse')
    c3.metric('Scenario category',category(scenario))
    st.subheader('PM2.5 response curve')
    rows=[]
    for pct in range(-50,101,10):
        temp=x.copy();temp['PM2.5']=x['PM2.5']*(1+pct/100)
        rows.append({'PM2.5 change (%)':pct,'Predicted AQI':float(forecast_model.predict(temp)[0])})
    fig=px.line(pd.DataFrame(rows),x='PM2.5 change (%)',y='Predicted AQI',markers=True,color_discrete_sequence=['#5de4d0'])
    fig.add_hline(y=original,line_dash='dot',line_color='#a5a0ff')
    st.plotly_chart(plot_style(fig),use_container_width=True)
    st.caption('Tree-based models can produce stepwise, non-monotonic changes. All other features are held constant, which may be physically unrealistic.')

elif page=='Seasonal Intelligence':
    hero('SEASONAL INTELLIGENCE / 05','The rhythm of the air.','Discover historical seasonal pollution patterns and explore month-by-month variation.')
    if hist is None:
        st.warning('Upload the original city_day.csv in the sidebar to enable full 2015–2020 seasonal analysis. The model ZIP only contains test-period summaries.')
        d=a[['Date','AQI']].copy()
        st.caption('Showing the available historical test-period observations only.')
    else:d=hist[['Date','AQI']].copy()
    d=d.dropna(subset=['AQI']);d['Month']=d.Date.dt.month
    monthly=d.groupby('Month').AQI.agg(['mean','median','max','count']).reset_index()
    monthly['Month_Name']=pd.to_datetime(monthly.Month,format='%m').dt.strftime('%b')
    c1,c2,c3=st.columns(3)
    c1.metric('Most polluted month',monthly.loc[monthly['mean'].idxmax(),'Month_Name'])
    c2.metric('Peak monthly mean AQI',f"{monthly['mean'].max():.1f}")
    c3.metric('Observed days',len(d))
    fig=px.bar(monthly,x='Month_Name',y='mean',color='mean',color_continuous_scale=['#227e83','#55d8c4','#ffa971'],labels={'mean':'Average AQI','Month_Name':'Month'},text_auto='.0f')
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(plot_style(fig,430),use_container_width=True)
    st.dataframe(monthly[['Month_Name','mean','median','max','count']].rename(columns={'mean':'Average AQI','median':'Median AQI','max':'Maximum AQI','count':'Days'}),use_container_width=True,hide_index=True)

elif page=='Explainable AI':
    hero('EXPLAINABLE AI / 06','Behind every forecast.','See how each feature shifts an XGBoost prediction relative to the model baseline using SHAP.')
    x=input_panel('explain_')
    st.caption('SHAP attributes the model prediction; it does not identify the physical cause of air pollution.')
    if st.button('Explain this forecast',type='primary'):
        try:
            import shap
            imp=forecast_model.named_steps['simpleimputer']
            booster=forecast_model.named_steps['xgbregressor']
            transformed=pd.DataFrame(imp.transform(x),columns=FEATURES)
            explainer=shap.TreeExplainer(booster)
            sv=explainer(transformed)
            vals=pd.DataFrame({'Feature':FEATURES,'SHAP impact':sv.values[0],'Value':transformed.iloc[0].values})
            vals=vals.sort_values('SHAP impact',key=lambda s:s.abs(),ascending=True)
            fig=px.bar(vals,x='SHAP impact',y='Feature',orientation='h',color='SHAP impact',color_continuous_scale=['#ed9ca9','#cbd9e7','#4edac7'],hover_data=['Value'])
            fig.update_layout(coloraxis_showscale=False)
            st.metric('Forecast AQI',f"{float(forecast_model.predict(x)[0]):.2f}")
            st.plotly_chart(plot_style(fig,500),use_container_width=True)
            st.caption(f'Model expected value: {float(np.asarray(sv.base_values).ravel()[0]):.2f} AQI. Positive SHAP values increase the forecast; negative values decrease it.')
        except Exception as e:
            st.warning('SHAP could not calculate an explanation in this environment. Try versions compatible with your Colab training environment.')
            st.exception(e)

st.divider()
st.caption('VAYUDRISHTI • HISTORICAL DELHI RESEARCH PROTOTYPE • XGBOOST + ISOLATION FOREST • NO LIVE SENSOR CONNECTION')
