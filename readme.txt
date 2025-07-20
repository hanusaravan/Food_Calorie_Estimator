🥗 Calorie & Macronutrient Estimator
Calorie & Macronutrient Estimator is a deep learning-based tool that estimates calories, carbohydrates, proteins, and fats from an uploaded food image.
Built with PyTorch and Streamlit, this project combines computer vision and nutrition data to provide real-time predictions for food analysis.


🚀 Features
- 📸 Upload a food image
- 🔍 Predicts:
  - Calories
  - Carbohydrates
  - Proteins
  - Fats
- 🧠 CNN-based model (e.g., ResNet18)
- 💻 Streamlit web interface
- 💾 Git LFS support for large model files


🛠️ Tech Stack
- Python
- PyTorch
- Streamlit
- Git LFS
- NumPy, Pandas
- PIL / OpenCV


📂 Project Structure
📦 calomacroesti/
┣ 📁 model/
┃ ┗ best_model.pth
┣ 📁 data/
┣ 📄 app.py
┣ 📄 predict.py
┣ 📄 utils.py
┣ 📄 calories_macros.json
┣ 📄 requirements.txt
┗ 📄 README.md

▶️ Run the App
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

📸 Example Outputs
Upload a food image, and get estimated:

