# How Pitch Metrics works — explained simply

Read this before showing the project to anyone. These are the questions people ask.

---

## 1. Where does the data come from?

- **Stats**: every Premier League player's season stats from Fantasy Premier League (FPL):
  minutes, goals, assists, xG, xA, clean sheets, saves, and FPL's own scores for
  *threat*, *creativity* and *influence*. One file per season, 2019-20 to now.
- **Market values (the answers)**: Transfermarkt values for players who joined or left a
  Premier League club each summer. About 2,100 values.

**Q: How do you connect the two?** By name and age (`step2_build_dataset.py`). Names are
written differently ("Ødegaard" vs "Odegaard"), so they are cleaned with `unidecode` and
compared with fuzzy matching (`rapidfuzz`, score ≥ 90). The age must match within about a year,
so two players with the same surname are not mixed up. About 1,100 values matched.

---

## 2. Features — what the model looks at

`features.py` turns raw stats into 23 numbers per player-season:

- **Per-90 rates**: goals per 90 minutes, xG per 90, assists per 90… A player with 5 goals in
  900 minutes (0.5 per 90) is compared fairly with one with 10 goals in 3,000 minutes (0.3 per 90).
- **Playing time**: share of all minutes played, share of matches started, minutes last season.
- **Age**, **position**, **club's league position**, **big-six club or not**, **year**
  (prices go up every year).

**Q: What is xG?** Expected goals: how many goals an average player would score from the chances
he had. It measures chance quality, so it is less "lucky" than real goals.

---

## 3. Why predict log(value)?

Values go from €0.3m to €120m. If we predicted the raw number, the model would care almost only
about the expensive players (a €10m error on a €100m player looks huge). Taking the log means a
20% miss counts the same for a cheap player and an expensive one. We turn the answer back with `exp()`.

---

## 4. The models

**XGBoost** (gradient-boosted decision trees). It builds 400 small decision trees, one after
another. Each new tree fixes the mistakes of the trees before it. Trees are good at patterns like
"value rises with age until about 25, then falls".

**Ridge linear regression.** Value = a weighted sum of the features. Simple, and it can predict
higher than any value it has seen. Trees cannot: they can only output averages of training values.

**The blend**: final log value = 0.6 × XGBoost + 0.4 × linear. It scored better than either
model alone in testing (see the results table in README).

**Q: Why is there a `StandardScaler` before the linear model?** It puts every feature on the same
scale (mean 0, standard deviation 1), so "age" (~25) and "goals per 90" (~0.3) can be compared
fairly. Ridge also shrinks the weights a little so they do not overreact (regularisation).

---

## 5. How do you know it works? (testing)

**Cross-validation (5-fold, grouped by player)**: split the data into 5 parts, train on 4, test
on the 5th, repeat 5 times. "Grouped" means the same player is never in both training and test,
otherwise the model could just remember him.

**Time test**: train only on 2019-20 to 2023-24 and predict 2024-25. This is the honest test,
because in real life you always predict the future from the past.

**Metrics**:
- **R² on the log value**: share of the differences between players the model explains (1.0 = perfect).
- **Typical error**: the median of |predicted ÷ real − 1|.
- **Within ±50%**: share of players predicted between half and 1.5× their real value.
- **Baseline**: the plain linear model. A fancier model is only worth it if it beats the simple one.

---

## 6. Predicting this season's value

`step4_predict.py` uses only this season's matches. After 6 games the numbers are noisy: one
lucky week can make anyone look like a superstar. So every per-90 stat is pulled toward the
average for the player's position:

```
weight = minutes / (minutes + 450)
stat   = weight × his stat + (1 − weight) × position average
```

At 450 minutes he counts half; at 2,000 minutes about 80%. This is called **shrinkage**
(a simple form of Bayesian thinking: start from the average, move as evidence comes in).
Goals and assists are also projected to a full 38-game season.

**Likely range**: in cross-validation, 80% of errors were between −X% and +Y%. We apply the same
range to every new prediction.

---

## 7. Explaining each value (SHAP)

SHAP (SHapley Additive exPlanations) splits one prediction into "pushes" from each feature,
starting from the average player. For trees, `shap.TreeExplainer` calculates this exactly. For
the linear model the push is simply *weight × scaled feature*. We mix both 60/40 like the value,
then add related features together ("Goal scoring" = goals + goals per 90 + xG per 90).

---

## 8. Player roles (K-means clustering) — `step6_roles.py`

This is **unsupervised** learning: there is no right answer to learn from.

1. Take each player's per-90 stats over the last 12 months (900+ minutes).
2. Scale them (`StandardScaler`) so every stat counts equally.
3. **K-means** puts players into k groups (k = 2 to 4 per position) so that players in a group
   are as close as possible to the group's centre.
4. Each group is named after the stat its centre is highest in (e.g. defending → "Ball-winner").
5. **PCA** (principal component analysis) squeezes the 6-7 stats into 2 numbers per player that keep
   most of the differences — that is the x/y of the role map.
6. **Similar players** = the 5 nearest players in the scaled stat space (Euclidean distance).

---

## 9. The value calculator

`step3_train.py` exports the trained model to JSON: every XGBoost tree (split feature, threshold,
leaf values) and every linear weight. The website walks the same 400 trees in JavaScript, so a
slider change gets a new price instantly with no server. It matches the Python predictions to
within 0.03% on average.

---

## 10. Match ratings

FPL's Bonus Points System (BPS) scores every action in a match (passes, tackles, saves, goals…).
We map it to the usual 0-10 scale: `rating = 6.0 + 0.06 × BPS`, kept between 3 and 10.

---

## 11. Automation

`.github/workflows/update.yml` tells GitHub to run `run_all.py` every Tuesday on a free cloud
computer, then commit the new `docs/index.html`. GitHub Pages serves the `docs/` folder as the website.

---

## Things you could improve next

- Add LightGBM or CatBoost and compare (log results in a table or with MLflow)
- Tune hyperparameters with Optuna
- Predict actual transfer fees (the Transfermarkt data has a `fee` column)
- Forecast next summer's value using age curves
- Add I-League / ISL players with a league-strength adjustment
