# NFL Draft Predictor

A machine learning model that predicts NFL draft potential using college production, athletic testing, and team strength metrics.

## Overview

This project builds position-specific models to evaluate prospects in the 2026 NFL Draft.

It combines:
- College production stats
- Combine / athletic testing
- Team strength metrics (SRS, DSRS)
- Eligibility and experience features

The goal is to rank prospects and identify high-upside players.

## Model

- Position-specific models (QB, WR, EDGE, LB, DB, etc.)
- XGBoost
- Feature engineering for production, athleticism, and team context

Example features:
- Playmaking (INT + PD)
- QB Score
- Pass Rush Score (TFL+Sacks)

## Example Output

Top Skill Players (Model Output):
       name        predicted_score
makai lemon         8.271661
jeremiyah love      7.160297
carnell tate        7.075327
ja'kobi lane        6.572565
omar cooper jr.     5.855565

## Project Structure

src/
- train.py
- preprocess.py

data/
- (not included due to size)

outputs/
- position rankings

  ## How to Run

pip install -r requirements.txt

python src/train.py

## Data

Large datasets are not included due to GitHub size limits.

Data includes:
- College stats
- Combine data
- Team SRS metrics
