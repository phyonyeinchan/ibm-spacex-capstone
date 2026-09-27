# IBM Applied Data Science Capstone: Falcon 9 landing analysis

Author: Phyo Nyein Chan. This project analyzes **historical IBM Skills Network course data**, not current SpaceX launches. The main modeling cohort contains 90 Falcon 9 records dated 2010–2020. SQL uses a separate 101-row historical table; Folium and Plotly use separate 56-row course extracts. The report states each denominator.

## Data provenance

Downloaded on 28 September 2026 from IBM Skills Network:

| Local file | Course URL | Use |
| --- | --- | --- |
| `dataset_part_2.csv` | https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv | EDA and prediction |
| `Spacex.csv` | https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_2/data/Spacex.csv | SQL queries |
| `spacex_launch_geo.csv` | https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_geo.csv | Folium map |
| `spacex_launch_dash.csv` | https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_dash.csv | Plotly dashboard |

The course API lab describes retrieval from `https://api.spacexdata.com/v4/launches/past`. It returned HTTP 525 during this revision. The Wikipedia launch page returned HTTP 403 here. The project uses archived course outputs and does **not** claim a new live API extraction or scrape.

## Reproduce

```bash
python -m pip install pandas numpy matplotlib scikit-learn folium plotly reportlab pillow nbformat ipython
python analysis.py
python create_notebook.py
python create_report.py
```

Run the commands in this directory. `analysis.py` generates SQL result CSVs, `model_scores.csv`, `results.json`, charts under `figures/`, and standalone `folium_map.html` and `plotly_dashboard.html`. The notebook contains saved outputs. The PDF is the Coursera upload file.

## Main result

The 90-row cohort has 60 successful landing labels. Four models were tuned using five-fold stratified cross-validation on 72 training observations. KNN and SVM tied at 87.4% mean training-fold accuracy. The deterministic tie ordering selects KNN. All four scored 77.8% accuracy on 18 held-out observations; the selected model's confusion matrix is `[[2, 4], [0, 12]]` for true/predicted labels `[0, 1]`. Failure detection is weak (2/6), so accuracy alone overstates usefulness.

## GitHub publication

Repository: https://github.com/phyonyeinchan/ibm-spacex-capstone

The repository is currently private. The Coursera grader may need public access to inspect the notebook and Python files; visibility should be decided before submission.
