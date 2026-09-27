"""Reproduce the IBM SpaceX capstone analysis from archived IBM course CSVs.

Source URLs and retrieval date are recorded in README.md. Data are historical,
not current SpaceX operations. Run in this directory: python analysis.py
"""
from pathlib import Path
import json, sqlite3
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import folium, plotly.express as px

HERE=Path(__file__).resolve().parent
OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
df=pd.read_csv(HERE/'dataset_part_2.csv')
df['Year']=pd.to_datetime(df.Date).dt.year
dash=pd.read_csv(HERE/'spacex_launch_dash.csv')
geo=pd.read_csv(HERE/'spacex_launch_geo.csv')
sql_source=pd.read_csv(HERE/'Spacex.csv')
con=sqlite3.connect(':memory:');sql_source.to_sql('SPACEXTBL',con,index=False,if_exists='replace')
queries={
 'sites':"SELECT Launch_Site,COUNT(*) AS launches FROM SPACEXTBL GROUP BY Launch_Site ORDER BY launches DESC",
 'payload':"SELECT ROUND(MIN(PAYLOAD_MASS__KG_),1) AS min_kg,ROUND(MAX(PAYLOAD_MASS__KG_),1) AS max_kg,ROUND(AVG(PAYLOAD_MASS__KG_),1) AS mean_kg FROM SPACEXTBL",
 'site_outcomes':"SELECT Launch_Site,COUNT(*) AS launches,SUM(CASE WHEN Landing_Outcome LIKE 'Success%' THEN 1 ELSE 0 END) AS successful_landings FROM SPACEXTBL GROUP BY Launch_Site ORDER BY launches DESC",
 'orbit':"SELECT Orbit,COUNT(*) AS launches FROM SPACEXTBL GROUP BY Orbit ORDER BY launches DESC LIMIT 8",
 'years':"SELECT SUBSTR(Date,1,4) AS year,COUNT(*) AS launches,SUM(CASE WHEN Landing_Outcome LIKE 'Success%' THEN 1 ELSE 0 END) AS successful_landings FROM SPACEXTBL GROUP BY SUBSTR(Date,1,4) ORDER BY year"
}
sql={k:pd.read_sql_query(v,con) for k,v in queries.items()};con.close()
for k,v in sql.items():v.to_csv(HERE/f'sql_{k}.csv',index=False)

features=['PayloadMass','Flights','GridFins','Reused','Legs','Block','ReusedCount','Orbit','LaunchSite']
X=pd.get_dummies(df[features],columns=['Orbit','LaunchSite'],dtype=float).astype(float)
y=df.Class.astype(int)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
models={
 'Logistic regression':(Pipeline([('scale',StandardScaler()),('model',LogisticRegression(max_iter=2000))]),{'model__C':[.01,.1,1,10]}),
 'Support vector machine':(Pipeline([('scale',StandardScaler()),('model',SVC())]),{'model__C':[.1,1,10],'model__kernel':['linear','rbf']}),
 'Decision tree':(DecisionTreeClassifier(random_state=42),{'max_depth':[2,3,5],'min_samples_leaf':[2,4,8]}),
 'K nearest neighbors':(Pipeline([('scale',StandardScaler()),('model',KNeighborsClassifier())]),{'model__n_neighbors':[3,5,7,9]})}
rows=[];fits={}
for name,(model,grid) in models.items():
    search=GridSearchCV(model,grid,scoring='accuracy',cv=cv,n_jobs=1).fit(Xtr,ytr)
    p=search.predict(Xte);fits[name]=search
    rows.append({'model':name,'cv_accuracy':round(search.best_score_,3),'test_accuracy':round(accuracy_score(yte,p),3),'test_f1':round(f1_score(yte,p,zero_division=0),3),'params':str(search.best_params_)})
scores=pd.DataFrame(rows).sort_values(['cv_accuracy','model'],ascending=[False,True]);scores.to_csv(HERE/'model_scores.csv',index=False)
best=scores.iloc[0]['model'];cm=confusion_matrix(yte,fits[best].predict(Xte),labels=[0,1])

plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
BLUE='#1676a2';GREEN='#249f8a';ORANGE='#ed9444'
def save(name):plt.tight_layout();plt.savefig(OUT/name,dpi=180,bbox_inches='tight',facecolor='white');plt.close()
t=df.groupby('LaunchSite').Class.agg(['size','mean']).sort_values('mean')
fig,ax=plt.subplots(figsize=(8,4));ax.barh(t.index,t['mean']*100,color=GREEN);ax.set(xlim=(0,100),xlabel='Landing success (%)',title='Landing success by launch site');save('site_success.png')
t=df.groupby('Orbit').Class.agg(['size','mean']).sort_values('mean')
fig,ax=plt.subplots(figsize=(8,4));ax.barh(t.index,t['mean']*100,color=BLUE);ax.set(xlim=(0,100),xlabel='Landing success (%)',title='Landing success by orbit');save('orbit_success.png')
t=df.groupby('Year').Class.agg(['size','mean'])
fig,ax=plt.subplots(figsize=(8,4));ax.plot(t.index,t['mean']*100,marker='o',color=BLUE);ax.set(ylim=(0,100),xlabel='Year',ylabel='Landing success (%)',title='Historical cohort trend');save('yearly.png')
fig,ax=plt.subplots(figsize=(8,4));ax.scatter(df[df.Class==0].PayloadMass,df[df.Class==0].Year,c=ORANGE,alpha=.75,label='No landing');ax.scatter(df[df.Class==1].PayloadMass,df[df.Class==1].Year,c=GREEN,alpha=.75,label='Landing');ax.set(xlabel='Payload mass (kg)',ylabel='Year',title='Payload versus year by landing outcome');ax.legend();save('payload_year.png')
fig,ax=plt.subplots(figsize=(8,4));ax.scatter(df.PayloadMass,df.Class+np.random.default_rng(42).normal(0,.04,len(df)),c=df.Class.map({0:ORANGE,1:GREEN}),alpha=.7);ax.set(xlabel='Payload mass (kg)',ylabel='Class (jittered)',title='Payload and landing result');save('payload_class.png')
fig,ax=plt.subplots(figsize=(8,4));ax.bar(scores.model,scores.test_accuracy,color=[GREEN,BLUE,ORANGE,'#75869b']);ax.tick_params(axis='x',rotation=15);ax.set(ylim=(0,1),ylabel='Held-out accuracy',title='Four classifiers');save('models.png')
fig,ax=plt.subplots(figsize=(5,4));ConfusionMatrixDisplay(cm,display_labels=['No landing','Landing']).plot(ax=ax,cmap='Blues',colorbar=False);ax.set_title(best);save('confusion.png')
fig,ax=plt.subplots(figsize=(5,4));ax.pie([int(y.sum()),int((1-y).sum())],labels=['Landing','No landing'],autopct='%1.1f%%',colors=[GREEN,ORANGE]);ax.set_title('Historical dataset outcome share');save('pie.png')
fig,ax=plt.subplots(figsize=(8,4));ax.scatter(geo.Long,geo.Lat,s=40,c=geo['class'].map({0:ORANGE,1:GREEN}),alpha=.8);ax.set(xlabel='Longitude',ylabel='Latitude',title='Launch-site coordinate records');ax.grid(alpha=.3);save('geo.png')

m=folium.Map(location=[29.8,-93.5],zoom_start=4,tiles='OpenStreetMap')
for site,g in geo.groupby('Launch Site'):
    lat,lon=g.Lat.iloc[0],g.Long.iloc[0]
    folium.Marker([lat,lon],tooltip=f'{site}: {len(g)} records; {g["class"].mean():.0%} successful').add_to(m)
    folium.Circle([lat,lon],radius=5000,color=BLUE,fill=False).add_to(m)
m.save(str(HERE/'folium_map.html'))
pie=px.pie(dash,names='Launch Site',values='class',title='Successful landings by launch site (course dashboard cohort)')
scatter=px.scatter(dash,x='Payload Mass (kg)',y='class',color='Launch Site',title='Payload mass and landing outcome')
(HERE/'plotly_dashboard.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>Historical launch dashboard</title></head><body><h1>IBM course historical launch dashboard</h1>'+pie.to_html(full_html=False,include_plotlyjs=True)+scatter.to_html(full_html=False,include_plotlyjs=False)+'</body></html>',encoding='utf-8')
summary={'cohort_n':len(df),'sql_n':len(sql_source),'geo_n':len(geo),'dash_n':len(dash),'successes':int(y.sum()),'rate':float(y.mean()),'date_min':str(df.Date.min()),'date_max':str(df.Date.max()),'missing':df.isna().sum().to_dict(),'best':best,'confusion':cm.tolist(),'models':rows,'site':df.groupby('LaunchSite').Class.agg(['size','mean']).round(3).reset_index().to_dict('records'),'orbit':df.groupby('Orbit').Class.agg(['size','mean']).round(3).reset_index().to_dict('records'),'year':df.groupby('Year').Class.agg(['size','mean']).round(3).reset_index().to_dict('records')}
(HERE/'results.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:summary[k] for k in ['cohort_n','sql_n','geo_n','dash_n','successes','rate','date_min','date_max','best','confusion','models']},indent=2))
