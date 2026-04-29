from src.data_loader import load_mrpc, load_sts
from src.preprocessing import preprocess_dataset
from src.baseline import BaselineModel
from src.sbert_model import SBERTModel
from src.evaluation import evaluate
from src.visualization import plot_model_comparison
from scipy.stats import pearsonr

def compute_sts_scores(model, dataset):
    scores = []
    for s1, s2 in zip(dataset["sentence1"], dataset["sentence2"]):
        score = model.similarity(s1, s2)
        scores.append(score)
    return scores

def main():
    print("🚀 Starting Paraphrase Detection Project...\n")

    # =========================
    # MRPC TASK
    # =========================
    print("📥 Loading MRPC dataset...")
    data = load_mrpc()

    print("🧹 Preprocessing MRPC...")
    data = preprocess_dataset(data)

    # BASELINE
    print("\n🔹 Running Baseline Model...")
    baseline_model = BaselineModel()
    baseline_model.train(data["train"])
    baseline_preds = baseline_model.predict(data["test"])

    print("\n--- BASELINE RESULTS ---")
    evaluate(data["test"]["label"], baseline_preds)

    
    # STS TASK
    # =========================
    print("\n📥 Loading STS dataset...")
    sts_data = load_sts()

    print("🧹 Preprocessing STS...")
    sts_data = preprocess_dataset(sts_data)

    for model_name in ['MiniLM', 'MPNet', 'Elite']:
        print(f"\n=====================================")
        print(f"🔹 Running Model: {model_name}")
        print(f"=====================================")
        
        sbert_model = SBERTModel(model_name)
        sbert_preds = sbert_model.predict(data["test"])

        print(f"--- {model_name} PARAPHRASE RESULTS ---")
        evaluate(data["test"]["label"], sbert_preds)

        print(f"\n🔹 Running STS Similarity for {model_name}...")
        pred_scores = compute_sts_scores(sbert_model, sts_data["validation"])
        true_scores = sts_data["validation"]["label"]

        correlation = pearsonr(pred_scores, true_scores)

        print(f"--- {model_name} STS RESULTS ---")
        print("Pearson Correlation:", correlation[0])


    # VISUALIZATION
    # =========================
    print("\n📊 Generating graphs...")
    plot_model_comparison()

    print("\n✅ Project Execution Completed!")

if __name__ == "__main__":
    main()
