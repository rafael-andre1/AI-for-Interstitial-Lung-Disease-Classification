# Intro

Using AI, I was able to put together a transformer learning sheet (includes theory, articles and source code), in order to (possibly) apply them in my internship project, in CT scan analysis, paired with CNNs.

---

## **1. Beginner-Friendly Explanations**  
📌 **Articles & Tutorials**  
- **The Illustrated Transformer** (by Jay Alammar) → [🔗 Link](https://jalammar.github.io/illustrated-transformer/)  
  - A fantastic visual explanation of how transformers work, including self-attention.  
- **Transformers from Scratch** → [🔗 Link](https://peterbloem.nl/blog/transformers)  
  - A detailed step-by-step guide on implementing transformers from scratch.  

📌 **Videos**  
- **DeepMind’s Introduction to Transformers** → [🔗 Link](https://www.youtube.com/watch?v=TQQlZhbC5ps)  
  - Great explanation from Google DeepMind on why transformers are powerful.  
- **Andrej Karpathy’s Neural Networks: Zero to Hero (GPT-focused)** → [🔗 Link](https://www.youtube.com/watch?v=kCc8FmEb1nY)  
  - Covers transformers in detail, including GPT-style architectures.  

---

## **2. Code Implementations & Hands-On Learning**  
📌 **Hugging Face (Industry-Standard Library for Transformers)**  
- Hugging Face provides pre-trained transformer models for various tasks, including medical imaging.  
- 🔗 **Hugging Face Transformers Library** → [https://huggingface.co/transformers/](https://huggingface.co/transformers/)  
- Example Code (Load a Transformer Model in Python):
  ```python
  from transformers import AutoModel, AutoTokenizer
  
  model_name = "bert-base-uncased"  # Example model
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModel.from_pretrained(model_name)
  
  text = "Transformers are amazing!"
  inputs = tokenizer(text, return_tensors="pt")
  outputs = model(**inputs)
  print(outputs.last_hidden_state.shape)
  ```

📌 **Open-Source Transformer Implementations in PyTorch**  
- **Annotated Transformer (Harvard NLP)** → [🔗 Link](https://nlp.seas.harvard.edu/2018/04/03/attention.html)  
  - A fully annotated PyTorch implementation of the original **Attention Is All You Need** paper.  

- **minGPT (Andrej Karpathy’s Tiny GPT)** → [🔗 GitHub](https://github.com/karpathy/minGPT)  
  - A minimalistic implementation of **GPT** from scratch.  

📌 **Transformers for Medical Imaging**  
- **Swin Transformer for Medical Imaging (MONAI)** → [🔗 GitHub](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR)  
  - **Swin-UNETR** is a transformer-based model for 3D medical imaging (perfect for lung CT analysis).  

- **Vision Transformer (ViT) in PyTorch** → [🔗 GitHub](https://github.com/lucidrains/vit-pytorch)  
  - A simple, modular implementation of **Vision Transformers (ViT)** for image-based tasks.  

---

## **3. Research Papers for Advanced Understanding**  
📄 **Must-Read Transformer Papers**  
- 🔬 **Attention Is All You Need** (2017, Original Transformer Paper) → [🔗 PDF](https://arxiv.org/abs/1706.03762)  
- 📊 **Vision Transformer (ViT) by Google** → [🔗 PDF](https://arxiv.org/abs/2010.11929)  
- 🏥 **Swin Transformer for Medical Image Segmentation** → [🔗 PDF](https://arxiv.org/abs/2105.05537)  

---

## **4. Interactive Learning & Courses**  
📌 **Free Courses**  
- **Stanford CS25: Transformers in Vision & NLP (2023)** → [🔗 YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rO1NB9TD4iUZ3qghGEGtqNX)  
  - Advanced course covering transformers in both **text & image processing**.  

- **Hugging Face Course (Hands-on with Transformers)** → [🔗 Link](https://huggingface.co/course/)  
  - A practical, code-heavy introduction to transformers in real-world applications.  
 

---

## Specifically related to my internship  

✅ **Vision Transformers (ViT, Swin Transformer)** – Handle 2D lung CT scans efficiently.  
✅ **Swin-UNETR (3D Transformer-Based UNet)** – Designed for volumetric (3D) medical image segmentation.  
✅ **Hybrid CNN-Transformer Models** – Combine **CNNs for feature extraction** and **Transformers for contextual understanding** of fibrosis patterns.  
