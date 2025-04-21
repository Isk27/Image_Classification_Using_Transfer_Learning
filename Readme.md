
# 🧠 Image Classification Using Transfer Learning

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> ⚡ Classify images with high accuracy using AI-powered transfer learning.  
> Built for fast, accurate, and intuitive performance — no prior training data needed.

---

<p align="center">
  <img src="https://github.com/your-username/image-classification-transfer-learning/blob/main/assets/sample_demo.jpg?raw=true" width="100%" />
</p>

---

## ✨ Key Features

- 🧠 **Transfer Learning**: Uses pre-trained models like ResNet50 for better accuracy
- 🚀 **Fast Inference**: Classifies images in seconds
- 🖼️ **Clean UI**: Built with Streamlit for an intuitive experience
- 📈 **Visualization**: Accuracy/Loss graphs, confusion matrix, system workflow

---

## 🛠️ Tech Stack

| Component           | Tech Used                        | Why It’s Used                                  |
|---------------------|----------------------------------|------------------------------------------------|
| Model Training       | `TensorFlow`, `Keras`            | Deep learning and pre-trained models           |
| Data Processing      | `NumPy`, `Pandas`                | Efficient image data handling                  |
| Visualization        | `Matplotlib`, `Seaborn`          | Accuracy, Loss, and Evaluation graphs          |
| UI Interface         | `Streamlit`                     | Interactive and quick deployment               |

---

## ⚙️ Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/isk27/Image_Classification_Using_Transfer_Learning.git
cd Image_Classification_Using_Transfer_Learning

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app/app.py
```

> 💡 Make sure to place your trained model (`.h5` or `.keras`) in the `model/` folder before launching.

---

## 📊 How It Works

1. Upload an image through the Streamlit interface  
2. Preprocessing is applied to prepare the image  
3. The image is passed to a fine-tuned pre-trained model (ResNet50)  
4. The model returns a predicted class with confidence score  
5. Visual outputs include image preview, prediction graph, and more

---

## 📦 Project Structure

```
├── app/                    # Streamlit frontend
├── model/                  # Trained models (.h5/.keras)
├── images/                 # Sample images
├── utils/                  # Helper functions
├── train.py                # Training logic
├── requirements.txt        # Python dependencies
├── PPT.pptx                # Project presentation
└── README.md               # This file
```

---

## 👨‍💻 Team Members

- 👩‍💻 Ishwari Kshirsagar  
- 👩‍💻 Monika Yawale  
- 👩‍💻 Janhavi Baraskar  
- 👩‍💻 Vaishnavi Baraskar  

**🧑‍🏫 Guide:** Prof. P. D. Bharsakle Ma'am  

---

## 💡 Future Enhancements

- 📈 Add real-time training visualization
- 🌐 Deploy to cloud (e.g., Heroku or AWS)
- 🔍 Add support for multi-class dataset expansion

---

## 🙌 Acknowledgments

- 🧠 [TensorFlow/Keras](https://www.tensorflow.org/) for transfer learning tools  
- 🎨 [Streamlit](https://streamlit.io/) for interactive web apps  
- 📸 You, for testing and supporting the project ⭐

---

## 🔗 Connect With Us

Stay connected for more cool projects and updates!

