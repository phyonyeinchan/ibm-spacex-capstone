from pathlib import Path
import os,sys,io,contextlib,base64
import pandas as pd,nbformat
P=Path(__file__).resolve().parent
os.chdir(P);sys.path.insert(0,str(P))
N=nbformat.v4
cells=[
 N.new_markdown_cell('# IBM Data Science Capstone: Falcon 9 historical launch analysis\n\n**Phyo Nyein Chan**. This executed notebook analyzes archived IBM Skills Network datasets from 2010-2020. It uses the supplied historical CSV outputs because the live SpaceX endpoint returned HTTP 525 during this revision. Dataset URLs and denominators are documented in `README.md`.'),
 N.new_code_cell('import analysis as a\nimport pandas as pd\nfrom IPython.display import Image\nprint("Modeling cohort:",a.df.shape,"; date range:",a.df.Date.min(),"to",a.df.Date.max())\ndisplay(a.df.head())'),
 N.new_markdown_cell('## Collection and wrangling\n\nThe course API workflow joins Falcon 9 launch, payload, core and launchpad details. The archived `dataset_part_2.csv` is its prepared output. The course scraping workflow normalizes a historical launch table; the separate IBM `Spacex.csv` supplies SQL records. The original live API and Wikipedia page were unavailable during this run, so this notebook does not claim a fresh collection.'),
 N.new_code_cell('print("Target distribution:")\ndisplay(a.df.Class.value_counts().sort_index().rename(index={0:"No landing",1:"Landing"}).to_frame("n"))\nprint("Missing values:")\ndisplay(a.df.isna().sum().loc[lambda s:s>0].to_frame("n"))\nprint("Duplicate FlightNumber:",int(a.df.FlightNumber.duplicated().sum()))'),
 N.new_markdown_cell('## Exploratory analysis with SQL\n\nExecuted SQLite queries group launch sites, payload masses, orbit categories and launch year. This SQL cohort has 101 rows and should not be combined silently with the 90-row model cohort.'),
 N.new_code_cell('for name,query in a.queries.items():\n print("\\n",name,"\\n",query)\n display(a.sql[name])'),
 N.new_markdown_cell('## EDA visualization\n\nThe figures show historical site, orbit and year patterns, plus payload versus landing outcome. Small subgroup counts limit interpretation.'),
 N.new_code_cell('for name in ["site_success.png","orbit_success.png","yearly.png","payload_year.png"]:\n print(name);display(Image(filename=str(a.OUT/name)))'),
 N.new_markdown_cell('## Folium map and Plotly dashboard\n\nThe map uses 56 course geospatial rows, one marker per launch site and 5 km circles. The Plotly HTML uses 56 dashboard rows, a success pie and a payload/outcome scatter. Both HTML files accompany this notebook.'),
 N.new_code_cell('print("Folium map:",a.HERE/"folium_map.html")\nprint("Plotly dashboard:",a.HERE/"plotly_dashboard.html")\ndisplay(Image(filename=str(a.OUT/"geo.png")))\ndisplay(Image(filename=str(a.OUT/"pie.png")))'),
 N.new_markdown_cell('## Predictive analysis\n\nA stratified 72/18 train/test split holds out final evaluation. GridSearchCV tunes four classifiers using five stratified folds. Scaling is confined to training folds in pipelines.'),
 N.new_code_cell('display(a.scores)\nprint("Selected by cross-validation:",a.best)\nprint("Confusion matrix [0,1]:\\n",a.cm)\nprint("Majority-class baseline:",round(a.yte.mean(),3))\ndisplay(Image(filename=str(a.OUT/"confusion.png")))'),
 N.new_markdown_cell('## Interpretation\n\nThe selected KNN classifier achieves 77.8% held-out accuracy (14/18). It detects all 12 successful landings, while 4 of 6 unsuccessful landings are false positives. The 18-row test cohort is too small for a stable operational claim. Replicate this workflow on a later dataset before deployment.')]
nb=N.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
namespace={'__name__':'__main__'}
from IPython.display import Image
count=0
for cell in nb.cells:
 if cell.cell_type!='code':continue
 count+=1;stream=io.StringIO();outputs=[]
 def display(obj):
  if isinstance(obj,pd.DataFrame):outputs.append(N.new_output('display_data',data={'text/plain':obj.to_string(),'text/html':obj.to_html()},metadata={}))
  elif isinstance(obj,Image):
   data=obj.data or Path(obj.filename).read_bytes()
   outputs.append(N.new_output('display_data',data={'image/png':base64.b64encode(data).decode()},metadata={}))
  else:outputs.append(N.new_output('display_data',data={'text/plain':repr(obj)},metadata={}))
 namespace['display']=display
 with contextlib.redirect_stdout(stream):exec(cell.source,namespace)
 cell.execution_count=count;cell.outputs=[]
 if stream.getvalue():cell.outputs.append(N.new_output('stream',name='stdout',text=stream.getvalue()))
 cell.outputs+=outputs
nbformat.write(nb,P/'SpaceX_Capstone_Executed.ipynb')
print('Saved executed notebook with',count,'code cells')
