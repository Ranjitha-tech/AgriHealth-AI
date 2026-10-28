\# 🌱 AgriHealth AI — Plant Disease Detection



AgriHealth AI is a machine learning web application that helps farmers identify plant diseases from leaf images and provides treatment and prevention guidance.



\## ✨ Features



\- 🌿 Plant disease prediction from uploaded leaf images

\- 📊 Prediction confidence percentage

\- 💊 Disease-specific treatment suggestions

\- 🛡️ Plant disease prevention guidance

\- 👨‍🌾 Farmer registration and login

\- 📜 User-specific prediction history

\- 🗄️ SQLite database for storing analysis history

\- 🖥️ Interactive web interface built with Streamlit



\## 🛠️ Technologies Used



\- Python

\- TensorFlow / Keras

\- Streamlit

\- SQLite

\- NumPy

\- Pillow

\- Machine Learning and Image Classification



\## 📁 Project Structure



```text

AgriHealth-AI/

├── app.py

├── evaluate\_model.py

├── models/

│   └── class\_names.json

├── src/

│   ├── check\_dataset.py

│   ├── predict.py

│   └── train.py

├── results/

│   ├── classification\_report.txt

│   ├── confusion\_matrix.png

│   └── evaluation\_results.txt

├── sample\_images/

│   └── test\_leaf.JPG

└── .gitignore

```



\## 🚀 Getting Started



\### 1. Clone the repository



```bash

git clone https://github.com/Ranjitha-tech/AgriHealth-AI.git

cd AgriHealth-AI

```



\### 2. Create and activate a virtual environment



Windows PowerShell:



```powershell

python -m venv venv

.\\venv\\Scripts\\Activate.ps1

```



\### 3. Install dependencies



Install the Python packages required by the application and model. Use the project's dependency list if provided.



\### 4. Configure the trained model



The trained model file is excluded from this repository because of its size. Place your trained model at:



```text

models/plant\_disease\_model.keras

```



The class labels are provided in `models/class\_names.json`.



\### 5. Run the application



```bash

streamlit run app.py

```



Open the local URL displayed in the terminal, usually `http://localhost:8501`.



\## 📈 Model Evaluation



The project includes a classification report, confusion matrix, and model evaluation results in the `results/` directory.



The model achieved approximately \*\*91.23% validation accuracy\*\* during the recorded evaluation. Results may vary depending on the dataset, training configuration, and evaluation setup.



\## 🔒 Data and Privacy



The training dataset, trained model file, and local analysis database are excluded from the Git repository. Users should keep credentials and private configuration files out of version control.



\## ⚠️ Disclaimer



Predictions are intended as decision-support information and should not replace expert agricultural advice. Verify disease identification and treatment recommendations before applying pesticides.



\## 👩‍💻 Author



\*\*Ranjitha N\*\*



\- GitHub: https://github.com/Ranjitha-tech



\---



If you find this project useful, consider giving the repository a ⭐ on GitHub.



