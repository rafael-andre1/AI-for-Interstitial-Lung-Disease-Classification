warnings.filterwarnings("ignore")

test_roc_results_aug = {}

# Getting each individual result
# 2D
test_roc_results_aug["2d_noAug"] = getROC(results_aug["2d_noAug"][3], test_dataset_2d_noAug, "2d_noAug")
test_roc_results_aug["2d_Aug"] = getROC(results_aug["2d_Aug"][3], test_dataset_2d_Aug, "2d_Aug")
# 2.5D
test_roc_results_aug["2_5d_noAug"] = getROC(results_aug["2_5d_noAug"][3], test_dataset_2_5d_noAug, "2_5d_noAug")
test_roc_results_aug["2_5d_Aug"] = getROC(results_aug["2_5d_Aug"][3], test_dataset, "2_5d_Aug")

plt.figure(figsize=(8, 6))
for label, result in test_roc_results_aug.items():
    plt.plot(result["fpr"], result["tpr"], label=f'{label} (AUC = {result["auc"]:.2f})')
    plt.scatter(result["fpr"][result["best_idx"]], result["tpr"][result["best_idx"]], color='red', s=20)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Combined ROC Curves')
plt.legend()
plt.tight_layout()
plt.show()