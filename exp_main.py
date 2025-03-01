from preprocess_data import load_data
from mmo_evoBagging import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.decomposition import PCA
import seaborn as sns
import argparse
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import warnings
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
warnings.filterwarnings("ignore")

def selection( bags, mode="selection"):
        selected_ids = []
        bag_idx, payoff_list = [], []
        for idx, bag in bags.items():
            bag_idx.append(idx)
            payoff_list.append(bag['payoff'])
        if mode=="selection":
            selected_ids = [idx for _, idx in sorted(zip(payoff_list, bag_idx), reverse=True)][:9]
            return None, selected_ids
        
def run(dataset_name, test_size, n_exp, metric, n_bags, n_iter, n_select, n_new_bags, n_mutation, mutation_rate, size_coef, clf_coef, voting, n_test, procs=4):
    print(f"Loading dataset: {dataset_name}")
    X_train, X_test, y_train, y_test = load_data(dataset_name, test_size=test_size)
    
    oversample = BorderlineSMOTE()
    X_train, y_train = oversample.fit_resample(X_train, y_train)
    
    max_initial_size = X_train.shape[0]
    mutation_size = int(max_initial_size*mutation_rate)
    n_crossover = n_bags - n_select - n_new_bags
    
    optimizer = MMO_EvoBagging(X_train, y_train, X_test, y_test, n_bags, n_select, n_new_bags, 
                            max_initial_size, n_crossover, n_mutation, 
                            mutation_size, size_coef, clf_coef, metric, n_test, procs)
    
    all_voting_train, all_voting_test = [], []
    all_f1_scores = []
    
    for t in range(n_exp):
        print(f"\nExperiment {t+1}/{n_exp}")
        
        # Generate N new bags
        bags = {i: optimizer.gen_new_bag() for i in range(n_bags)}
        
        # Evaluate bags
        bags = optimizer.evaluate_bags(bags)
        
        voting_train, voting_test = [], []
        f1_scores = []
        
        for i in range(n_iter):
            print(f"Iteration {i+1}/{n_iter}")
            
            _, selection_idx = selection(bags)
            bags = optimizer.mmo_evobagging_optimization(bags, X_test, y_test)
            
            # Get predictions and metrics
            majority_voting_train_metric, _, _, _, f1, precision, recall, _, _ = optimizer.voting_metric_roc(bags, X_train, y_train)
            majority_voting_test_metric, _, _, _, test_f1, _, _, _, _ = optimizer.voting_metric_roc(bags, X_test, y_test)
            
            voting_train.append(majority_voting_train_metric)
            voting_test.append(majority_voting_test_metric)
            f1_scores.append(f1)
            
            print(f"Train accuracy: {majority_voting_train_metric:.4f}")
            print(f"Test accuracy: {majority_voting_test_metric:.4f}")
            print(f"F1 score: {f1:.4f}")
        
        best_iter = np.argmax(voting_train)
        all_voting_train.append(voting_train[best_iter])
        all_voting_test.append(voting_test[best_iter])
        all_f1_scores.append(f1_scores[best_iter])
    
    final_results = {
        "train_accuracy": float(np.mean(all_voting_train)) / 100,  # Convert from percentage
        "test_accuracy": float(np.mean(all_voting_test)) / 100,
        "f1_score": float(np.mean(all_f1_scores)),
        "experiment_details": {
            "dataset": dataset_name,
            "n_experiments": n_exp,
            "n_bags": n_bags,
            "n_iterations": n_iter
        }
    }
    
    print("\nFinal Results:")
    print(f"Average training accuracy: {final_results['train_accuracy']:.4f}")
    print(f"Average test accuracy: {final_results['test_accuracy']:.4f}")
    print(f"Average F1 score: {final_results['f1_score']:.4f}")
    
    return final_results

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Main experiment for real datasets')
    parser.add_argument('--dataset_name', type=str,
                        help='Dataset name')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Percentage of test data')
    parser.add_argument('--n_exp', type=int, default=30,
                        help='Number of experiments')
    parser.add_argument('--metric', type=str, default='accuracy',
                        help='Classification metric')
    parser.add_argument('--n_bags', type=int,
                        help='Number of bags')
    parser.add_argument('--n_iter', type=int, default=20,
                        help='Number of iterations')
    parser.add_argument('--n_select', type=int, default=0,
                        help='Number of selected bags each iteration')
    parser.add_argument('--n_new_bags', type=int,
                        help='Generation gap')
    parser.add_argument('--n_mutation', type=int,
                        help='Number of bags to perform mutation on')
    parser.add_argument('--mutation_rate', type=float, default=0.05,
                        help='Percentage of mutated instances in each bag')
    parser.add_argument('--size_coef', type=float,
                        help='Constant K for controlling size')
    parser.add_argument('--clf_coef', type=float,
                        help='Constant P for controlling Bag Performance')
    parser.add_argument('--voting', type=str, default='majority',
                        help='Type of voting rule')
    
    parser.add_argument('--n_test', type=int, default=8,
                        help='Number of test bags')
    parser.add_argument('--procs', type=int, default=16,
                        help='Number of parallel processes')
    args = parser.parse_args()

    run(dataset_name=args.dataset_name, 
        test_size=args.test_size, 
        n_exp=args.n_exp,
        metric=args.metric,
        n_bags=args.n_bags, 
        n_iter=args.n_iter,
        n_select=args.n_select,
        n_new_bags=args.n_new_bags,
        n_mutation=args.n_mutation,
        mutation_rate=args.mutation_rate,
        size_coef=args.size_coef,
        clf_coef=args.clf_coef,
        voting=args.voting,
        n_test=args.n_test,
        procs=args.procs)
