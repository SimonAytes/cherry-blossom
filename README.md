# Cherry Blossom Peak Bloom Prediction

> [!NOTE]
> This is my entry for the 2024 CMU International Cherry Blossom Prediction Competition. Learn more about it [here](https://competition.statistics.gmu.edu/).

This project aims to predict the peak bloom dates of cherry blossom trees using machine learning. With cherry blossom trees blooming earlier than ever due to climate change, accurately forecasting their peak bloom has become a significant challenge. This repository contains the code, data, and documentation related to our efforts in developing a model to predict cherry blossom peak bloom dates.

## Project Overview

In this project, we developed a Gradient Boosting Regressor (GBR) model to predict the peak bloom date of cherry blossom trees. Leveraging over 245 unique features from previous year's weather data, our model can forecast peak bloom dates with remarkable accuracy. To streamline data acquisition and maintain model performance, we introduced the concept of "Representative Locations," strategically chosen to capture geographical diversity while minimizing data sourcing efforts.

## Repository Structure

- `Cherry-Blossom_files/`: Source files for this project's [Quarto](https://quarto.org/) document.
- `data/`: Contains datasets used for model training and evaluation.
- `figures/`: Associated graphics for this project.
- `notebooks/`: Jupyter notebooks containing code for data exploration, model development, and evaluation.
- `src/`: Source code for data preprocessing, model training, and evaluation.
- `models/`: Trained models stored for deployment. The model used in my competiton submission is `GBR-20240218-02`.

## Getting Started

To get started with the project, follow these steps:

1. Clone the repository:

```
git clone https://github.com/your_username/cherry-blossom-peak-bloom-prediction.git
```

2. Navigate to the project directory:

```
cd cherry-blossom
```

3. Install the required dependencies:

```
pip install -r requirements.txt
```

4. Explore the notebooks in the `notebooks/` directory to understand the data preprocessing, model development, and evaluation process.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
