# FractalGCL - Graph Contrastive Learning for Urban Road Networks

:warning:**!! Please open in Colab**


https://drive.google.com/drive/folders/147F0-2HkwwbVFDa3jovgVqoJB5PjRqqG?usp=sharing


*(Click the badge to open every notebook on Google Colab – no local setup needed.)*


## 0 Data Confidentiality Notice !!

Because the original city datasets are subject to confidentiality agreements, we cannot fully disclose them. 

However, we have generated and provided **100 random samples** of each city’s data. 

This approach lets reviewers reproduce all experiments without accessing the raw, proprietary data. 

!! Because the dataset from 100 samples is too large, we have stored the sampling results in the Google Drive link shared above. !!

For details and code, see `city/fractalgcl_city.ipynb`.

---

## 1  Repository layout


FractalGCL/
├── pre_experiments/ # preliminary sanity-checks on small graphs
│ ├── pre_experiments.ipynb # implements Exp-1 & Exp-2 + all plots
│ └── R2_num_data/ # CSV files used by the notebook
│
└── city/ # full urban pipeline
├── fractalgcl_city.ipynb # FractalGCL + 5 baselines on NY/SF/Chicago
└── data/ # small helper files 


All algorithms are implemented **in pure Python with PyTorch Geometric** and were tested end-to-end on Google Colab.

---

## 2  Quick start (for reviewers)

| What you want to check | How to run (≈ time) |
|------------------------|---------------------|
| **Preliminary Exp 1 & 2** | Open **`pre_experiments/pre_experiments.ipynb`** in Colab and run **_Runtime ▶ Run all_**. &lt. |
| **Full city benchmark** | Open **`city/fractalgcl_city.ipynb`**, choose a city in the first cell (`NY`, `SF`, `Chicago`).|

> ⚠️ **Nothing to download manually.**  
> The notebooks automatically pull their tiny helper CSVs from `pre_experiments/R2_num_data/` or `city/data/`, and generate everything else on-the-fly.

---

## 3  Key notebooks

| Notebook | Contents | Output |
|----------|----------|--------|
| **pre_experiments.ipynb** | • Box-dimension estimator<br>• R² filtering vs. downstream accuracy<br>• All figures in Sec. 3 of the paper | PNG figures saved to `pre_experiments/figures/` |
| **fractalgcl_city.ipynb** | • Builds GIS graph for chosen city<br>• Trains 5 SSL baselines (DGI, InfoGraph, SimGRACE, GCL-Manual, JOAO) **+ FractalGCL**<br>• Runs 17 downstream probing tasks with 1000×10-fold CV | CSV tables + plots under `city/results/` |

---
