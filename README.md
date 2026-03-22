# MLOps Individual Assignment

**Name:** Megha A  
**Roll Number:** 727823TUAM024  
**Dataset:** Robo-Advisor Allocation (working file: Dataset AM024.xlsx, Aggregate sheet)

## Project Structure
- `code/training.py` - MLflow experiment tracking with 12 runs
- `code/data_prep.py` - Pipeline stage 1
- `code/train_pipeline.py` - Pipeline stage 2
- `code/evaluate.py` - Pipeline stage 3
- `code/pipeline_727823TUAM024.yml` - Azure ML pipeline definition
- `notebooks/eda.ipynb` - EDA notebook
- `screenshots/` - MLflow and Azure screenshots
- `report/report.pdf` - Assignment report

## MLflow Experiment Name
`SKCT_727823TUAM024_RecommendedPortfolio`

## Best Run Details
- **Best Run ID:** `f10a0fca0fd0468493f8e1a347571cc8`
- **Best R2:** `0.7032729513563983`

## Setup Steps
1. Install dependencies from `requirements.txt`
2. Place dataset file in `data/`
3. Run `python code/training.py`
4. Run pipeline scripts individually:
   - `python code/data_prep.py`
   - `python code/train_pipeline.py`
   - `python code/evaluate.py`