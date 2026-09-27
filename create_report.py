from pathlib import Path
import json, textwrap, os
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image

P=Path(__file__).resolve().parent
pdfmetrics.registerFont(TTFont('DejaVu','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DejaVu-Bold','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
R=json.loads((P/'results.json').read_text())
S={n:pd.read_csv(P/f'sql_{n}.csv') for n in ['sites','payload','site_outcomes','orbit','years']}
W,H=960,540
PDF=P/'Data_Science_Capstone_Project_Report_REVISED.pdf'
c=canvas.Canvas(str(PDF),pagesize=(W,H))
c.setTitle('IBM Data Science Capstone - SpaceX historical launch analysis')
c.setAuthor('Phyo Nyein Chan')
NAVY=HexColor('#19334f');BLUE=HexColor('#176f9c');INK=HexColor('#19334f');MUTED=HexColor('#53697d')
GREEN=HexColor('#0c907e');PALE=HexColor('#eff5f8')
number=0
def page(title,subtitle=''):
 global number
 number+=1;c.setFillColor(HexColor('#ffffff'));c.rect(0,0,W,H,fill=1,stroke=0)
 c.setFillColor(NAVY);c.rect(0,H-115,W,115,fill=1,stroke=0)
 c.setFillColor(HexColor('#ffffff'));c.setFont('DejaVu-Bold',25);c.drawString(44,H-58,title)
 c.setFillColor(HexColor('#ccdfec'));c.setFont('DejaVu',11);c.drawString(45,H-84,subtitle)
 c.setFillColor(MUTED);c.setFont('DejaVu',8);c.drawString(44,22,'IBM course historical datasets | Analysis through 2020 | Phyo Nyein Chan')
 c.drawRightString(914,22,str(number))
def txt(items,x=48,y=390,size=18,leading=28,width=90):
 c.setFillColor(INK);c.setFont('DejaVu',size)
 for item in items:
  for line in textwrap.wrap(item,width=width,break_long_words=False):c.drawString(x,y,line);y-=leading
  y-=13
def table(headers,rows,widths,x=48,y=389,rh=32):
 c.setFillColor(BLUE);c.setFont('DejaVu-Bold',12)
 for i,h in enumerate(headers):c.drawString(x+sum(widths[:i]),y,str(h))
 y-=rh
 for r in rows:
  c.setFillColor(INK);c.setFont('DejaVu',11)
  for i,v in enumerate(r):c.drawString(x+sum(widths[:i]),y,str(v)[:38])
  y-=rh
def fig(name,x=44,y=58,w=870,h=340):
 im=Image.open(P/'figures'/name);iw,ih=im.size;scale=min(w/iw,h/ih)
 nw,nh=iw*scale,ih*scale;c.drawImage(ImageReader(im),x+(w-nw)/2,y+(h-nh)/2,nw,nh)
def end():c.showPage()

page('Winning the space race with data science','Historical Falcon 9 first-stage landing analysis | Applied Data Science Capstone')
txt(['IBM Skills Network historical course datasets, analyzed with Python, SQL, Folium, Plotly and classification models.', '90 launch records in the modeling cohort; data dated 2010 to 2020.', 'This report presents executed results. Its figures do not describe current launch performance.'],size=21,leading=37,width=76);end()
page('Executive summary','Method and result in one view')
txt(['Collected archived IBM course outputs for launch analysis, SQL, mapping and dashboard work. Cleaned the cohort, queried launch patterns and built interactive visuals.', '60 of 90 historical modeling rows have a successful landing label (66.7%).', 'Four classifiers were tuned with five-fold cross-validation on 72 training rows. The best cross-validation score was 87.4%; held-out accuracy was 77.8% on 18 rows.', 'The selected model identified all 12 successful landings in the test set but misclassified 4 of 6 unsuccessful ones.'],size=16,width=99);end()
page('Introduction and business question','Can launch features anticipate first-stage landing success?')
txt(['A successful first-stage landing supports rocket reuse. For a launch with known mission and booster features, predict Class = 1 for landing success.', 'The analytical unit is a Falcon 9 launch record. This study examines historical association and predictive performance; it does not estimate cost savings or causal effects.', 'Data window: 4 June 2010 through 5 November 2020 in the 90-row modeling cohort.'],size=18,width=91);end()
page('Data collection: SpaceX API','Archived IBM output from the course API workflow')
txt(['Course collection sequence: request past launches from the SpaceX v4 API; look up related rocket, launchpad, payload and core records; filter Falcon 9; assemble launch features.', 'For this reproducible report, dataset_part_2.csv is the archived IBM Skills Network output from that workflow. It has 90 rows and 18 original columns.', 'The live API returned HTTP 525 during this revision. The analysis uses the archived course dataset and records its direct source URL in the README.'],size=17,width=96);end()
page('Data collection: web scraping','Historical launch table and SQL source')
txt(['The course scraping workflow extracts Falcon 9 launch-table rows, normalizes column names, removes citation artifacts and parses payload mass and landing outcome.', 'The IBM Skills Network Spacex.csv historical table supplies 101 records for SQLite analysis. The course datasets for mapping and dashboard each contain 56 records.', 'These datasets are separate historical extracts. Their counts and field names differ, so SQL, map, dashboard and model denominators are stated on each result slide.'],size=17,width=96);end()
page('Data wrangling methodology','Consistent outcomes and audit checks')
txt(['Parsed Date to Year, retained the IBM binary Class target and validated its observed values. A value of 1 denotes a successful first-stage landing.', 'Checked nulls and duplicates. The modeling file has 26 missing LandingPad entries; that field was excluded from classification. Predictor columns used in the model have no missing entries.', 'One-hot encoded Orbit and LaunchSite. Standardization for logistic regression, SVM and KNN occurs inside training-fold pipelines.'],size=17,width=96);end()
page('EDA and visualization methodology','The question answered by each chart')
table(['Visual','Purpose','Denominator'],[['Site bar','Compare landing rates by site','90'],['Orbit bar','Compare mission profiles','90'],['Year line','Inspect historical time pattern','90'],['Payload scatter','Inspect mass and outcome','90'],['Map','Locate recorded launch sites','56'],['Dashboard','Filter payload and success by site','56']],[190,440,140],rh=32);end()
page('EDA with SQL methodology','Queries executed in SQLite against SPACEXTBL')
txt(['COUNT(*) GROUP BY Launch_Site identifies sites and ranks launch volume.', 'MIN, MAX and AVG summarize payload mass. CASE WHEN Landing_Outcome LIKE Success% counts successful landing records.', 'GROUP BY Orbit ranks orbit frequencies. SUBSTR(Date,1,4) groups launches and successful landings by year.', 'The SQL table has 101 rows and a different date and outcome schema from the 90-row modeling cohort.'],size=17,width=96);end()
page('Interactive visual analytics','Map and dashboard methods')
txt(['Folium: place one marker at each recorded launch site; tooltip reports site, row count and success fraction. A 5 km circle around each marker shows local proximity.', 'Plotly: a success-by-site pie chart and payload-mass versus landing-outcome scatter. Hover and site color reveal individual records.', 'The accompanying folium_map.html and plotly_dashboard.html open in a browser. The visual cohort has 56 historical records.'],size=18,width=93);end()
page('EDA: payload and landing','Lower and higher payloads occur in both outcome groups');fig('payload_class.png');end()
page('EDA: landing rate by orbit','Single-record orbit categories can show extreme percentages');fig('orbit_success.png');end()
page('EDA: landing rate by site','Historical 90-row model cohort');fig('site_success.png');end()
page('EDA: yearly landing trend','Early years have very few records');fig('yearly.png');end()
page('EDA: payload, time and outcome','A joint view helps separate year trends from mission mix');fig('payload_year.png');end()
page('SQL results: site and payload','101-row historical table')
rows=[]
for r in S['site_outcomes'].itertuples():rows.append([r.Launch_Site,int(r.launches),int(r.successful_landings),f'{r.successful_landings/r.launches:.1%}'])
table(['Launch site','Launches','Landings','Rate'],rows,[275,170,180,120],rh=30)
p=S['payload'].iloc[0];txt([f'Payload mass: minimum {p.min_kg:,.0f} kg; maximum {p.max_kg:,.0f} kg; mean {p.mean_kg:,.0f} kg.'],y=160,size=16);end()
page('SQL results: orbit ranking','Top eight orbit labels by number of launches')
table(['Orbit','Launches'],[[r.Orbit,int(r.launches)] for r in S['orbit'].itertuples()],[250,220],rh=29);end()
page('SQL results: time analysis','SQL counts by year, with successful landing records')
table(['Year','Launches','Landings'],[[r.year,int(r.launches),int(r.successful_landings)] for r in S['years'].itertuples()],[230,230,230],rh=25);end()
page('Folium map: sites and markers','Static preview of historical 56-row coordinate dataset');fig('geo.png');end()
page('Folium map: outcomes and proximity','Interactive marker tooltips and five-kilometre circles')
geo=pd.read_csv(P/'spacex_launch_geo.csv')
table(['Site','Records','Landing rate'],[[s,len(g),f'{g["class"].mean():.1%}'] for s,g in geo.groupby('Launch Site')],[320,190,210],rh=36)
txt(['The circle is a visual distance guide. No claim about distance to coastlines, railways or roads is made.'],y=180,size=16);end()
page('Plotly dashboard: success composition','Historical 90-row model cohort shown for report consistency')
fig('pie.png',x=50,y=85,w=440,h=320)
txt(['Landing: 60 of 90 (66.7%).', 'No landing: 30 of 90 (33.3%).', 'The interactive HTML uses the separate 56-row course dashboard dataset and includes a site pie and payload/outcome scatter.'],x=515,y=360,size=16,width=43);end()
page('Plotly dashboard: payload and result','Static preview; inspect interactive scatter in the HTML file');fig('payload_year.png');end()
page('Predictive analysis: classification','Training design and controls')
txt(['Predictors: payload mass, booster flights, grid fins, reuse, landing legs, block, reuse count, orbit and launch site.', 'Stratified split: 72 training rows, 18 held-out test rows. Five-fold stratified GridSearchCV selects hyperparameters by training accuracy.', 'Logistic regression, support vector machine, decision tree and K nearest neighbors are compared. The test set is held out from tuning.', 'Baseline test accuracy from always predicting landing is 12/18 = 66.7%.'],size=17,width=96);end()
page('Predictive analysis: four models','Cross-validation selects the model; test results audit it')
table(['Model','CV accuracy','Test accuracy','Test F1'],[[r['model'],f'{r["cv_accuracy"]:.1%}',f'{r["test_accuracy"]:.1%}',f'{r["test_f1"]:.3f}'] for r in R['models']],[295,185,185,145],rh=40)
txt(['SVM and KNN tie at 87.4% cross-validation accuracy. The deterministic tie rule selects KNN. All four score 77.8% on this small test split.'],y=170,size=16,width=95);end()
page('Predictive analysis: confusion matrix','Selected KNN model on 18 held-out rows');fig('confusion.png',x=155,y=65,w=650,h=340);end()
page('Conclusions and innovative insights','Three results that affect interpretation')
txt(['1. Test accuracy rises 11.1 percentage points above the majority-class baseline, yet the selected model misses four of six failed landings. Accuracy alone hides that operationally relevant error.', '2. Site differences are sizable in this cohort (60.0% at CCAFS versus 77.3% at KSC), but site and year distributions may differ. These are descriptive comparisons.', '3. Orbit rates of 0% or 100% for one-record groups are unstable. Counts must accompany percentages in any decision briefing.', 'Historical data and an 18-row test set limit generalization. A later cohort would be needed for external validation.'],size=16,width=100);end()
page('Repository and reproducibility','Project files and data provenance')
txt(['GitHub repository: https://github.com/phyonyeinchan/ibm-spacex-capstone', 'The repository contains the executed notebook, Python analysis script, archived IBM CSV inputs, SQL result tables, interactive HTML and this PDF.', 'Data provenance and exact source URLs are in README.md. Run python analysis.py from the project folder to regenerate results.', 'Repository access is currently private. The course grader may need public access to inspect the files.'],size=17,width=95);end()
c.save();print(PDF,number)
