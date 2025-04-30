### **1. Introduction**

This internship was carried out in collaboration with INESC‑TEC and FEUP/FCUP, and its main objective was to develop a model capable of identifying key radiological features associated with Interstitial Lung Diseases (ILDs) through analysis of High-Resolution Computed Tomography (HRCT) scans. 

ILDs show manifestations of over 200 distinct conditions, each portraying different degrees of lung fibrosis and inflammation. Regarding respiratory pathologies, they represent a significant source of both morbidity and mortality.

Diagnosis is usually performed through patient history, physical examination, and pulmonary function testing. However, **(HRCT)** remains the gold standard for radiological evaluation, as it allows for non-invasive identification of distinctive features such as **reticular opacities, ground-glass opacities, cystic lesions**, **honeycombing** and **nodular patterns**. The type, distribution and stages of these manifestations can help distinguish between subtypes and assess severity. Although interpretation of HRCT scans appears to be a viable path, it remains a challenging and time-consuming task, even for experienced radiologists. Visual assessment of most patterns often involves a degree of subjectivity, which may introduce variability into clinical decisions.

Nonetheless, the main issue resides in the presence of multiple manifestations often complicates interpretation, leading to **diagnostic overlap and ambiguity**. Even for experienced radiologists, interpretation can vary, resulting in **inter\intra-observer inconsistency**. 

To address these complications, the application of robust artificial intelligence (AI) and computer vision has gained significant traction. Well-developed models show promising results in enhancing radiological workflows through automated and reproducible image analysis tools, and they are increasingly used to support accurate diagnosis and classification.

Despite the technical challenges, this project offered an amazing experience. Through research and dedicated orientation, it allowed for the significant improvement of the understanding of medical imaging concepts, disease evaluation, manifestation\symptom identification, data extraction\processing, model design and optimization, as well as important technical features.


---

### **2. Background / State of the Art**

The application of artificial intelligence in medical imaging has undergone a significant transformation in recent years, primarily driven by advances in deep learning—particularly convolutional neural networks (CNNs). In thoracic imaging, CNN-based models have demonstrated state-of-the-art performance in a wide range of tasks including lung segmentation, lesion detection, and disease classification \cite{litjens2017survey, shen2017deep}. Within the specific context of ILDs, automated segmentation of lung abnormalities remains an active and complex research problem.

ILDs are typically associated with a variety of imaging patterns visible on HRCT scans. These patterns include but are not limited to:  
- **Ground-glass opacities (GGOs):** hazy areas of increased attenuation without obscuring underlying vessels  
- **Reticulation:** a network of fine lines indicating fibrosis  
- **Honeycombing:** clustered cystic air spaces usually located in subpleural regions

Traditional image processing techniques, including histogram analysis and texture-based descriptors, have provided limited success due to the nuanced appearance and variability of these patterns. The advent of deep learning has enabled more sophisticated approaches. Notably, U-Net and its variants \cite{ronneberger2015unet} have become a standard backbone for biomedical image segmentation tasks due to their ability to capture fine-grained features while preserving spatial context.

Publicly available datasets such as LIDC-IDRI and the ILD Database from the NIH have catalyzed progress in this domain, although challenges persist regarding annotation quality, class imbalance, and generalizability. Recent works have proposed semi-supervised and attention-based networks to overcome these issues, often improving performance by integrating radiologist-in-the-loop feedback mechanisms \cite{gao2020focus, chen2021transformer}.

From a software and implementation standpoint, frameworks like **PyTorch** and **TensorFlow** have accelerated development, while tools such as **SimpleITK** and **MONAI** facilitate the manipulation and preprocessing of medical imaging data. Evaluation metrics commonly used include Dice Similarity Coefficient (DSC), Intersection over Union (IoU), and Hausdorff Distance, providing quantitative benchmarks for model performance.

Overall, the field is rapidly evolving, and although promising results have been achieved in research settings, there remains a gap in translating these solutions into robust, clinically usable tools. This project is positioned within this context: to bridge algorithmic innovation with clinical applicability, through rigorous experimentation and real-world validation.

---

If you'd like, I can now integrate this into your LaTeX report and format it properly—or help you write the next section (“Project Description”) in the same tone and quality. Would you like to continue with that?