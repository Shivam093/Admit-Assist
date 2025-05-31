
---

# AdmitAssist

**AdmitAssist** is a Django-based web application designed to help students assess their chances of gaining admission to their desired Master’s programs. By entering their academic details, users can receive a percentage likelihood of acceptance based on various machine learning and deep learning models.

## Features

- **Predictive Admission Analysis:** Users can input their academic details and receive a percentage chance of admission.
- **Multiple Model Evaluations:** The app uses various machine learning (ML) and deep learning (DL) models, such as Random Forest, SVM, Logistic Regression, and ANN, to predict outcomes.
- **User-Friendly Interface:** The frontend is designed for ease of use, allowing students to interact with the application seamlessly.

## Technologies Used

- **Backend:** Python, Django
- **Frontend:** HTML, CSS, JavaScript
- **Machine Learning:** Random Forest, SVM, Logistic Regression, Artificial Neural Networks (ANN)
- **Database:** MySQL

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- pip (Python Package Installer)
- MySQL

### Clone the Repository

```bash
git clone https://github.com/Shivam093/Admit-Assist.git
```

### Install Required Libraries

Install the necessary dependencies using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### Run the Application

Start the Django development server:

```bash
python manage.py runserver
```

## Usage

1. **Enter Academic Details:** Fill out the form with your GPA, test scores, and other academic information.
2. **Select Desired University:** Choose from a list of universities to see your admission chances.
3. **View Prediction:** Receive a percentage chance of admission based on the chosen model.
4. **University Recommendations:** Suggest alternative universities based on user input and model predictions.


## Future Enhancements

- **Hybrid Modeling:** Combine multiple models to improve prediction accuracy.
- **Expanded Dataset:** Incorporate more data points from various universities to improve model training.

## Contributions

Contributions are welcome! Feel free to open an issue or submit a pull request.

---
