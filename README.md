# CSIRO Image2Biomass — Project Overview (Brief)

## 1. Objective
Predict pasture biomass components from top-view images and associated metadata such as NDVI, height, and species information.  
The model estimates five targets — `Dry_Clover_g`, `Dry_Dead_g`, `Dry_Green_g`, `Dry_Total_g`, and `GDM_g` — using a **single multi-task deep learning model**.

---

## 2. Dataset
The dataset contains:
- RGB pasture images (`train/` and `test/` folders)
- Metadata and biomass targets in `train.csv`
- `test.csv` for inference and `sample_submission.csv` for submission format

Each image has multiple biomass component measurements, resulting in a **long-format dataset**.

---

## 3. Preprocessing
- **Handling missing values:** All missing `target` values are filled with the mean of their respective target columns to prevent training instability.  
- **Pivoting:** The dataset is transformed from long to wide format so each image has one row with five target columns.  
- **Encoding:** Categorical columns like `State` and `Species` are label-encoded.  
- **Scaling:** Numerical features such as NDVI and height are standardized.  
- **Augmentation:** Training images are augmented using random crops, flips, brightness, and rotation to improve generalization.

---

## 4. Model Architecture
- A **single multi-task model** built using a `timm` CNN backbone (e.g., EfficientNet).
- Image features and tabular features (NDVI, height, state, species) are extracted separately and then fused.
- The final layer outputs five regression values corresponding to the biomass components.
- **Loss:** Mean Squared Error (MSE)  
- **Optimizer:** AdamW  
- **Metric:** Root Mean Squared Error (RMSE)

---

## 5. Training Strategy
- **Cross-validation:** 5-Fold GroupKFold is used to ensure all data from the same image stays in one fold.
- Each fold trains independently, and the best-performing checkpoint (lowest validation RMSE) is saved.
- Results are averaged across folds for robustness.

---

## 6. Inference & Submission
- Trained fold models are loaded for inference.
- Predictions from all folds are averaged to create final results.
- Negative predictions are clipped to zero.
- The final output follows the Kaggle submission format (`sample_id`, `target`) and is saved as `submission.csv`.

---

## 7. Key Takeaways
- Filling NaNs early is essential for stable training.
- GroupKFold prevents data leakage across the same image.
- Multi-task learning improves efficiency and correlation between targets.
- Ensemble averaging across folds yields smoother, more reliable predictions.
- Model checkpoints and preprocessing artifacts (encoders/scaler) are stored for reproducibility.

---

## 8. Limitations

- **Limited dataset size:** The number of unique images is relatively small, which may restrict the model’s generalization on unseen vegetation patterns or lighting conditions.  
- **Imbalanced targets:** Certain biomass components (e.g., Dry Clover vs. Dry Dead) may have uneven distributions, potentially biasing the model toward more common classes.  
- **Simplified feature fusion:** The model combines image and tabular features through simple concatenation, which might not fully capture complex interactions between environmental and visual cues.  
- **Environmental variability:** Differences in lighting, soil color, and camera orientation can introduce noise that basic augmentations may not fully mitigate.  
- **2D representation only:** The approach relies solely on RGB imagery — lacking depth or spectral information (e.g., multispectral or hyperspectral data) that could improve biomass estimation accuracy.  
- **Limited explainability:** While Grad-CAM or other XAI techniques can be applied, this implementation does not deeply interpret model decisions.  





