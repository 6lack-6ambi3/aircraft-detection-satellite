# Aircraft Detection in Satellite Imagery\
\
Binary classification of 20\'d720 px satellite image chips from the\
[PlanesNet dataset](https://www.kaggle.com/datasets/rhammell/planesnet),\
determining whether each chip contains an **aircraft** or **background**.\
\
---\
\
## Project Overview\
\
This project implements and compares two methodological approaches:\
\
**Classical Method**\
- Feature extraction: Histogram of Oriented Gradients (HOG) + colour histograms \uc0\u8594  536-D feature vector\
- Classifiers: SVM (RBF kernel), Random Forest (200 trees), k-NN (k=7)\
\
**Modern Method**\
- Deep Multi-Layer Perceptron: 1200 \uc0\u8594  512 \u8594  256 \u8594  128 \u8594  64 \u8594  1\
- Trained on raw pixel inputs with Adam optimiser\
\
All implementation is fully local \'97 no external APIs or pre-trained models.\
\
---\
\
## Dataset\
\
**PlanesNet** \'97 32,000 satellite image chips (20\'d720 px RGB) from Planet Labs.\
\
| Split | Chips | Aircraft | Background |\
|-------|-------|----------|------------|\
| Train | 22,400 | 5,600 (25%) | 16,800 (75%) |\
| Val   | 4,800  | 1,200 (25%) | 3,600 (75%)  |\
| Test  | 4,800  | 1,200 (25%) | 3,600 (75%)  |\
\
> **Note:** The dataset file `planesnet.json` (~300 MB) is not included in\
> this repository. Download it from\
> [Kaggle](https://www.kaggle.com/datasets/rhammell/planesnet) and place\
> it in the project root directory.\
\
---\
\
## Results\
\
| Model | Accuracy | Sensitivity | Specificity | F1 | AUC-ROC | Inference |\
|-------|----------|-------------|-------------|----|---------|-----------|\
| SVM (RBF) | 97.3% | 93.2% | 98.7% | 0.945 | 0.994 | 501.5 ms |\
| Random Forest | 91.0% | 69.1% \uc0\u9888  | 98.3% | 0.793 | 0.965 | 4.14 ms |\
| k-NN (k=7) | 97.0% | 95.1% | 97.6% | 0.940 | 0.994 | 0.295 ms |\
| **Deep MLP** | **97.3%** | **95.6%** | **97.9%** | **0.948** | **0.993** | **0.007 ms** |\
\
\uc0\u9888  Random Forest sensitivity (69.1%) fails the 82% minimum requirement\
due to class imbalance effects on Gini impurity splitting.\
\
**Recommended for deployment:** Deep MLP \'97 highest sensitivity,\
fastest inference (71,600\'d7 faster than SVM).\
\
---\
\
## Key Findings\
\
- Colour histogram features dominate over HOG shape features in Random\
  Forest importance rankings \'97 the grey/white metallic aircraft surface\
  is the primary discriminating signal on real PlanesNet data\
- HOG + SVM matches Deep MLP accuracy at 20\'d720 pixel resolution\
- Deep MLP is the only viable choice for real-time deployment\
\
---\
\
## Installation\
\
```bash\
# Clone the repository\
git clone https://github.com/YOUR_USERNAME/aircraft-detection-satellite.git\
cd aircraft-detection-satellite\
\
# Install dependencies\
pip install -r requirements.txt\
\
# Download PlanesNet dataset from Kaggle and place planesnet.json here\
# Then run:\
python aircraft_detection_planesnet.py\
```\
\
---\
\
## Output\
\
Running the script generates 13 plots in an `outputs/` directory:\
\
- Sample chips visualisation\
- Class distribution charts\
- Preprocessing comparison\
- HOG feature maps\
- Feature importance (Random Forest)\
- Training curves (MLP)\
- Confusion matrices for all 4 models\
- ROC curves comparison\
- Metrics bar chart\
- Speed comparison\
\
---\
\
## Project Structure\
\
```\
aircraft-detection-satellite/\
aircraft_detection.py  # Main pipeline\
requirements.txt                 # Dependencies\
outputs/                         # Generated plots\
*.png\
data/\
gitkeep                    # planesnet.json goes here (not tracked)\
README.md\
```\
\
---\
\
## Tools & Libraries\
\
| Library | Purpose |\
|---------|---------|\
| scikit-learn | SVM, RF, k-NN, MLP, metrics |\
| scikit-image | HOG extraction, image filters |\
| NumPy | Array operations |\
| Matplotlib | All visualisations |\
\
---
