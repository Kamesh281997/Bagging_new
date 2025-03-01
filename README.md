
# MMO-EvoBagging: Multi-O# MMO-EvoBagging: Multi-Objective Optimization for Ensemble Learning

## Overview
MMO-EvoBagging is an advanced machine learning project that implements a multi-objective optimization approach for ensemble learning using bagging. The project combines evolutionary algorithms with bagging techniques to create robust and efficient ensemble models.

## Features
- Multi-objective optimization for ensemble creation
- Interactive dashboard for model training and monitoring
- Real-time experiment tracking with MLflow
- Support for multiple datasets
- Customizable training parameters
- Performance visualization and metrics tracking

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
1. Clone the repository:bjective Optimization for Ensemble Learning

## Overview
MMO-EvoBagging is an advanced machine learning project that implements a multi-objective optimization approach for ensemble learning using bagging. The project combines evolutionary algorithms with bagging techniques to create robust and efficient ensemble models.

## Features
- Multi-objective optimization for ensemble creation
- Interactive dashboard for model training and monitoring
- Real-time experiment tracking with MLflow
- Support for multiple datasets
- Customizable training parameters
- Performance visualization and metrics tracking

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
1. Clone the repository:git clone https://github.com/yourusername/mmo-evobagging.git
2.## Project Structure
mmo-evobagging/
├── app.py # Streamlit dashboard
├── exp_main.py # Main experiment runner
├── mmo_evoBagging.py # Core MMO-EvoBagging implementation
├── exp_diversity.py # Diversity metrics implementation
├── preprocess_data.py # Data preprocessing utilities
├── evo_req.txt # Project dependencies
└── mlruns/ # MLflow experiment tracking data

## Usage
1. Start the MLflow tracking server: mlflow server --host 127.0.0.1 --port 5000
2. Launch the Streamlit dashboard: streamlit run app.py
3. Access the dashboard at `http://localhost:8501`

### Training Parameters
- **Dataset**: Choose from available datasets (pima, breast_cancer, ionosphere, sonar, heart)
- **Test Size**: Proportion of data used for testing (0.1-0.5)
- **Number of Experiments**: Number of experimental runs
- **Number of Bags**: Size of the ensemble
- **Number of Iterations**: Training iterations per experiment

## Experiment Tracking
The project uses MLflow for experiment tracking and visualization. Access the MLflow UI at `http://localhost:5000` to:
- View experiment history
- Compare different runs
- Analyze metrics and parameters
- Export results

## Key Components

### MMO-EvoBagging
- Implements multi-objective optimization for ensemble creation
- Balances diversity and accuracy in the ensemble
- Supports various voting schemes (majority, weighted)

### Dashboard
- Real-time training monitoring
- Interactive parameter configuration
- Results visualization
- Experiment tracking integration

## Results
The system provides various metrics and visualizations:
- Training and Testing Accuracy
- F1 Score
- Precision and Recall
- Training History Plots
- Ensemble Diversity Metrics

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.


## Contact
- Kaemsh Kewlani
- Email: karankewlani1997@gmail.com
