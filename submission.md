
1. Problem Statement
What you tackled: “6-way classification of TREC questions into ABBR, ENTY, DESC, HUM, LOC, NUM.”

Why it matters: quick question‐type detection enables downstream QA systems to route or answer appropriately.

2. Dataset & Preprocessing
Source: Hugging Face’s SetFit/TREC-QC (5,452 train, 500 test).

Cleaning: dropped the 50-class fine labels, kept only the 6-class coarse label; removed the tag from the question text.

Tokenization: DistilBERT tokenizer, max_length=64, padding/truncation.

3. Model & Training
Architecture: TFDistilBertForSequenceClassification (6 output heads).

Framework: TensorFlow/Keras, AdamW optimizer with warmup (10% of steps), LR=2e-5.

Hyperparameters: 3 epochs, batch size 16.

Results: final train/validation losses and the 97.2% val accuracy.

4. Evaluation
Metrics: Precision, recall, F1 per class (macro F1 = 0.96, weighted F1 = 0.97).

Confusion Matrix: embed your reports/metrics_tf.png:

markdown
Copy
Edit
![Confusion Matrix](reports/metrics_tf.png)
Key observations:

Perfect ABBR classification.

ENT Y recall slightly lower (0.90) with some confusion into NUM/LOC.

DESC class small support (9 examples), F1 = 0.88—target for data augmentation.

5. Next Steps
Data augmentation for under-represented classes (back-translation).

Multilingual extension: swap in XLM-RoBERTa and fine-tune on translated TREC.

Ensemble or distillation to speed up inference on edge devices.

6. Key Learnings
Importance of cleaning label columns to avoid out-of-range errors.

How to convert between PyTorch tensors and NumPy for a TF model.

Setting up reproducible pipelines with DatasetDict → tf.data.Dataset.