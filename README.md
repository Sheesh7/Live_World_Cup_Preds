# ⚽ Real-Time Bayesian-Adjusted Win-Probability Tracker

An end-to-end predictive analytics framework designed to model, estimate, and visualize match outcome probabilities $P(\text{Home})$, $P(\text{Draw})$, $P(\text{Away})$ dynamically during international football fixtures.

The system couples a **Deep Learning classification pipeline** (establishing static pre-match Priors) with a **stateful real-time data engine** that continuosly updates predictions using in-game telemetry, time decay, and game-state gravity.

---

## 🏗️ Architectural Topology & Workflow

The platform utilizes a decoupled client-server architecture, separating the predicitve modeling layer from the real-time simulation layer.
* **Predictive Baseline Engine:** A Python/TensorFlow pipeline that processes historical match results and squad market values to generate symmetric, unbiased match baselines.
* **Backend (app.py & engine.py):** A FastAPI server that orchestrates the pre-calculated baselines and manages the Bayesian probability engine for live data.
* **Frontend (index.html):** A high peformance, responsive UI built with Vanilla JavaScript and Chart.js, utilizing an auto-polling loop to render real-time telemetry

### Design Decision: Hybrid Machine Learning + Heuristic Architecture
* **The Choice:** A static Neural Network generates pre-match priors, while a deteministic mathematical engine handles live in-game  adjustments.
* **The 'Why':** Training an end-to-end sequential model (e.g., LTSM or Transformer) on live match events requires massive, granular time-series tracking that are highly sensitive to missing API data packets. By separating the static baseline capability (squad strength, historical weight) from a deterministic live adjustment layer, the system remains highly robust, completely transparent, and immune to live API connection drops.

---

## 🧠 Deep Learning Pipeline & Feature Engineering

### 1. Domain-Specific Feature Extraction & Normalization
Instead of feeding raw, high-variance metrics directly to the model, the data pipeline implements several non-linear transformations to isolate accurate performance indicators:

* **Tournament Taxonomy Weighting ('match_wt'):** Historical matches are non-linearly scaled based on competitive pressure to prevent friendly matches from diluting the signal of high-stakes tournaments:
    * FIFA World Cup Games = 4
    * Continental Tournaments (e.g., Euros, Copa América, AFCON) & Qualifiers = 3
    * UEFA/CONCACAF Nations League & Standard Qualifiers = 2
    * International Friendlies = 1

* **Exponential Decay Ranking Transformation ('ranking_delta):** Raw FIFA ranking are ordinal and do not represent the non-linear gap in skill between elite squads and lower-ranked teams. To correct this, rankings are mapped to a continuous latent "Power Score" space using an exponential decay function before calculating the delta:
    $$\text{Power} = e^{-\frac{\text{Ranking}}{20.0}}$$

$$(\text{Ranking Delta} = \text{Power}_{\text{Home}} - \text{Power}_{\text{Away}})$$

* **Log-Transform Market Value Standardizer ('log_market_value_delta):** Roster values span multiple orders of magnitude (from millions to billions of Euros). A regex parser standardizes currency text representations into uniform floats, and a log transformation $\log_e(1 + x)$ is applied to compress the extreme right skew of finacial distributions, allowing the neural network to stably process financial disparities.

* **Attacking/Defensive Delta Synthesis ('scoring_delta):** Combines the home team's historical offensive output against the away team's defensive concession rate ($GF_{\text{Home}} - GA_{\text{Away}}$) to model direct tactical match-ups.

### 2. Neural Network Topology & Training Routine
The classification task maps continuous and categorical features to a categorical probability distribution across three mutually exclusive classes: `[0: Home Win, 1: Draw, 2: Away Win]`.

* **Architecture: **
    * **Input Layer:** Multi-dimensional scaled vector derived via a fitted `StandardScaler` to guarantee zero mean and unit variance ($z = \frac{x - \mu}{\sigma}$).
    * **Layer 1:** `Dense(128, activation='relu')` paired with a `Dropout(0.3)` regularizer to mitigate co-adaptation of features.
    * **Layer 2:** `Dense(64, activation='relu')` paired with `Dropout(0.2)` to prevent overfitting on historical outliers.
    * **Layer 3:** `Dense(32, activation='relu')` to project data into a dense latent space.
    * **Output Layer:** `Dense(3, activation='softmax')`, yielding a valid, normalized probability distribution: $$\sum_{i=0}^{2} P(\text{Class}_i) = 1.0$$
* **Compilation Details: ** Optimized via the **Adam** algorithm (`learning_rate = 0.001`) minimizing a **Sparse Categorical Cross-Entropy** objective function, which eliminates the memory overhead of one-hot encoding target labels.

### 3. Bias-Free Symmetric Inference Ensembling
* **Feature Implementation:** Because tournament matches take place at neutral venues, assigning a team as "Home" or "Away" introduces an artificial home-field advantage bias within the neural network.
* **Methodology:** The pipeline executes a **Two-Pass Cross-Swap Ensemble**:
    1. **Pass 1:** Evaluates the scheduled match configuration (`Team A` coded as Home, `Team B` as Away) to produce a probability vector $\mathbf{P}_1 = [P_H, P_D, P_A]$.
    2. **Pass 2:** Synthesizes an artificial inverted match (`Team B` coded as Home, `Team A` as Away) to produce a probability vector $\mathbf{P}_2 = [P'_H, P'_D, P'_A]$.
    3. **Ensemble Aggregation:** The final unbiased baseline probabilities are computed by cross-averaging opposing viewpoints:
    $$\text{Final Home Win \%} = \frac{P_H + P'_A}{2} \times 100$$
    $$\text{Final Draw \%} = \frac{P_D + P'_D}{2} \times 100$$
    $$\text{Final Away Win \%} = \frac{P_A + P'_H}{2} \times 100$$

---

## ⚡ Real-Time Processing Engine (`engine.py`)
The engine functions as stateful middleware, pulling boxscores via an API and executing a deterministic **Bayesian Update Process** that treats the neural network outputs as priors.

### 1. Frontend & API Integration
The frontend is built for performance, moving away from monolithic server-side rendering to a client-side reactive model.
* **RESTful API Interface:** The FastAPI backend servers match fixtures and live probabilities over JSON.
* **Client-Side State Management:** JavaScript manages history arrays for charts, allowing high-frequency probability timeline visulaization.
* **Non-Blocking Polling:** A 10-second `setInterval` loop ensures the UI stays synchronized with live ESPN boxscore updates without blocking the browser thread.

### 2. Bayesian Live Probability Modifiers
Live metrics modify the baseline priors ($p_h, p_d, p_a$) dynamically through explicit mathematical penalties and multipliers:

* **Disciplinary Penalties:** Reduces a squad's probability weight based on severe roster changes:
    $$\text{Red Card Penalty} = -7.0\% \quad | \quad \text{Yellow Card Penalty} = -1.5\%$$
* **Momentum Modification Coefficient:** Quantifies field pressure by combining possession delta with an exponential attacking volume multiplier:
    $$\text{Threat Vol} = \text{Shots} + (2 \times \text{Shots on Target})$$
    $$\text{Momentum Modifier} = (\Delta\text{Possession} \times 0.10) + (\Delta\text{Threat Vol} \times 0.015)$$
- **Time Decay and State Gravity:** Using the proportion of time remaining ($\tau = \frac{90 - t}{90}$), probabilities are bound to live score states: `gd = Score_Home - Score_Away`
    * **Leading State ($gd \neq 0$):** Activates a non-linear decay function. As $\tau \to 0$, the trailing team’s chances vanish and the leading team's holding capability approaches 1.0:
    $$\text{Holding Power} = 1.0 - \tau^{(1.2 \cdot |gd|)}$$
    $$\text{Live } P_{\text{Leader}} = P_{\text{Leader}} + (1.0 - P_{\text{Leader}}) \times \text{Holding Power}$$
    $$\text{Live } P_{\text{Trailer}} = P_{\text{Trailer}} \times \tau^{(|gd| + 1)}$$
    * **Level State ($gd = 0$):** Activates **Draw Gravity**. As time runs out ($\tau \to 0$), the likelihood of breaking a tie decreases linearly, pulling the entire remaining probability distribution into the Draw class:
    $$\text{Draw Gravity} = 1.0 - \tau$$
    $$\text{Live } P_{\text{Draw}} = P_{\text{Draw}} + (1.0 - P_{\text{Draw}}) \times \text{Draw Gravity}$$
    $$\text{Live } P_{\text{Teams}} = P_{\text{Teams}} \times \tau$$

---

## 🛠️ Setup & Installation

### 1. Prerequisites
Install the required standard libraries and frameworks:
```bash
pip install fastapi uvicorn pandas numpy requests tensorflow scikit-learn
```
### 2. Running the Application
1. Generate baselines by running prediction notebook to generate `world_cup_26_baselines`.
2. Start the API: `uvicorn app:app --reload`
3. Access the Dashboard: Open `index.html` in your browser
## 👥 Author

**Yashish Eriki** *Data Science @ Purdue University* * [GitHub](https://github.com/Sheesh7)  
* [LinkedIn](https://linkedin.com/in/yashisheriki)  
* Email: yashish.eriki7@gmail.com  

---

## 🚀 Future Roadmap & Improvements

To align this tracker with my ongoing research workflows and technical data stack, the following improvements are prioritized for the next deployment cycles:

### 1. Database Migration & Master Data Management
* **SQLite Integration:** Transition data asset storage (`all_matches.csv`, profiles, and baselines) from flat CSV files into an organized **SQLite** relational database schema. 
* **Data Integrity Governance:** Implement relational primary/foreign key constraints and automated validation scripts to manage squad profiles, enforcing strict data quality assurance and validation principles similar to enterprise master data systems.

### 2. Algorithmic Upgrades & Anomaly Detection
* **Ensemble Modeling (XGBoost / Random Forest):** Expand the modeling pipeline beyond the Keras Neural Network by training **XGBoost** and **Random Forest** classification baselines to compare categorical loss convergence and evaluate predictability.
* **In-Game Anomaly Tracking:** Develop rolling-window statistical indicators to monitor data drift and capture anomalies during live matches (e.g., sudden possession collapses, extreme underperformance against pre-match power-score deltas).

### 3. Automated Data Pipelines & Logging
* **Pipeline Refactoring:** Apply software maintenance and refactoring practices (similar to my pipeline standardization work with Purdue and NOAA) to modularize the ingestion script, separating raw payload fetching from feature mapping.
* **State Logging:** Implement automated reporting logs within the Streamlit polling loop to track and output statistical summaries of data processing performance upon match finalization.
