evalPatientProbResNet(resnet_eval, test_dataset, mean_thresh)

for key, val in results_patient.items():
    print(f"               Method: {key}\n")
    m_patient = evalResNet(val[3], test_dataset, threshold=all_rocs[key])
    metrics_aug[key] = m_aug
    print(" --------------------- \n")