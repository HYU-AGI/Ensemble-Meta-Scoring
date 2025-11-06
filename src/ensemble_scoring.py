import os
import torch
from datetime import datetime
from tqdm import tqdm
import argparse
import logging
from logger import setup_logger
from utils import load_json, save_json, floatify
import numpy as np
from tabm_reference import Model, make_parameter_groups
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import f1_score
import joblib


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def train(args, dataset):
    dataset_X, dataset_Y = dataset['X'], dataset['Y']
    test_ratio = args.testset_ratio
        
    X = np.array(dataset_X)
    y = np.array(dataset_Y)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(np.unique(y_encoded))
    print(f"# of classes: {num_classes}")
    joblib.dump(le, os.path.join(args.model_save_dir, f"label_encoder.pkl"))

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, 
                                                      test_size=(test_ratio), 
                                                      random_state=42, 
                                                      stratify=y_encoded)


    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)


    model = Model(n_num_features = X_train.shape[1],
                    cat_cardinalities = [],
                    n_classes = num_classes,
                    backbone = {'type': 'MLP',
                                'n_blocks': 3,
                                'd_block': 512,
                                'dropout': 0.1,},
                    bins = None,
                    num_embeddings = None,
                    arch_type = 'tabm',
                    k = 32,
                    share_training_batches = True)

    print("Model parameter: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    model.to(args.device)
    optimizer = torch.optim.AdamW(make_parameter_groups(model), lr=2e-3, weight_decay=3e-4)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds  = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)

    best_test_f1 = 0.0
    best_test_acc = 0.0

    for epoch in range(1, args.n_epochs+1):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(args.device)
            batch_y = batch_y.to(args.device)
            optimizer.zero_grad()

            y_pred = model(batch_X, None)  # shape: (batch, k, n_classes)
            b, k, n_cls = y_pred.shape
            loss = F.cross_entropy(y_pred.reshape(b * k, n_cls),
                                batch_y.unsqueeze(1).repeat(1, k).reshape(b * k))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()

        y_true_test = []
        y_pred_test = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(args.device)
                batch_y = batch_y.to(args.device)
                outputs = model(batch_X, None)
                outputs_prob = F.softmax(outputs, dim=-1)
                outputs_mean = outputs_prob.mean(dim=1)
                preds = outputs_mean.argmax(dim=1)
                y_true_test.append(batch_y.cpu().numpy())
                y_pred_test.append(preds.cpu().numpy())

        y_true_test = np.concatenate(y_true_test)
        y_pred_test = np.concatenate(y_pred_test)
        test_acc = (y_true_test == y_pred_test).mean()

        test_f1 = f1_score(y_true_test, y_pred_test, average='macro')
        
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_test_acc = test_acc
            save_path = os.path.join(args.model_save_dir, f'best_ensemble_scorer_model.pt')
            torch.save(model.state_dict(), save_path)
            print(f"[Epoch {epoch}] new best test f1 score: {test_f1*100:.2f}% & accuracy: {test_acc*100:.2f}%")
            print(f"Model saved to: {save_path}")
    
    print(f"\nAccuracy: {best_test_acc*100:.2f}%, F1-score: {best_test_f1:.4f}\n")


def scoring(args, data):

    le_path = os.path.join(args.model_save_dir, f"label_encoder.pkl")
    le = joblib.load(le_path)
    num_classes = len(le.classes_)
    model = Model(n_num_features = len(data),
                    cat_cardinalities = [],
                    n_classes = num_classes,
                    backbone = {'type': 'MLP',
                                'n_blocks': 3,
                                'd_block': 512,
                                'dropout': 0.1,},
                    bins = None,
                    num_embeddings = None,
                    arch_type = 'tabm',
                    k = 32,
                    share_training_batches = True)
    
    checkpoint_path = os.path.join(args.model_save_dir, 'best_ensemble_scorer_model.pt')
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint)
    model.to(args.device)
    model.eval()

    x = torch.tensor([data], dtype=torch.float32).to(args.device)
    outputs = model(x, None)                        # (1, k, n_classes)
    outputs_prob = F.softmax(outputs, dim=-1)       # (1, k, n_classes)
    outputs_mean = outputs_prob.mean(dim=1)         # (1, n_classes)
    probs = outputs_mean.squeeze(0).detach().cpu().numpy()  # (n_classes,)

    label_names = le.classes_
    prob_dict = {label: float(prob) for label, prob in zip(label_names, probs)}

    return prob_dict

    

def main():
    parser = argparse.ArgumentParser(description="Compute meta-scores for already generated responses.")
    parser.add_argument("--seed", type=int, default=42, help="random seed setting")
    parser.add_argument("--model_name", type=str, default="Mathstral-7B", help="Model name used for generation")
    parser.add_argument("--dataset_name", type=str, default="CreativeMath", help="dataset name")
    parser.add_argument("--meta_score_res_dir", type=str, default="output", help="meta-scoring results directory")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory for ensemble meta-scores")
    parser.add_argument("--ensemble_scorer_dir", type=str, default="ensemble_scorer_checkpoint", help="Directory to save ensemble_scorer model parameters")
    parser.add_argument("--log_dir", type=str, default="logs", help="logging log dir")
    parser.add_argument("--log_level", type=str, default="INFO", help="logging log level")
    parser.add_argument("--mode", type=str, default="ensemble_scoring", help="train or ensembel_scoring")
    parser.add_argument("--testset_ratio", type=float, default=0.3, help="testset ratio")
    parser.add_argument("--n_epochs", type=int, default=50, help="# of epochs")
    

    args = parser.parse_args()

    set_seed(args)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    log_dir = os.path.join(args.log_dir, args.dataset_name, 'ensemble_meta_scoring')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{args.model_name}_{timestamp}.log")
    logger = setup_logger(log_file, args.log_level)
    logger = logging.getLogger(__name__)
    logger.info("Starting ensemble_meta_scoring for already computed meta-scores...")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info(f"Logs saved to {os.path.abspath(log_file)}")


    # meta-scoring 결과 불러오기
    saved_res_path = os.path.join(args.meta_score_res_dir, args.dataset_name, 'meta_scoring', f"{args.model_name}_meta_scores.json")
    saved_data = load_json(saved_res_path)

    # ensemble meta-scoring 결과 저장 경로 설정
    output_dir = os.path.join(args.output_dir, args.dataset_name, 'ensemble_meta_scoring')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.model_name}_ensemble_meta_scores.json")

    args.model_save_dir = os.path.join(args.ensemble_scorer_dir, args.dataset_name, args.model_name)

    
    if args.mode == "train":
        os.makedirs(args.model_save_dir, exist_ok=True)
        dataset = {'X': [], 'Y': []}
        for idx, entry in tqdm(enumerate(saved_data)):
            features = []
            label = entry['label']
            meta_scores = entry["Computed_metrics"]
            for k, score in meta_scores.items():
                try:
                    features.append(float(score))
                except:
                    pass
            dataset['X'].append(features)
            dataset['Y'].append(label)

        train(args, dataset)


    elif args.mode == "ensemble_scoring":
        results = []
        for idx, entry in tqdm(enumerate(saved_data)):
            features = []
            label = entry['label']
            meta_scores = entry["Computed_metrics"]
            for k, score in meta_scores.items():
                try:
                    features.append(float(score))
                except:
                    pass
            if features:
                res = scoring(args, features)
                entry['ensemble_scoring'] = res
                results.append(entry)
                
        save_json(floatify(results), output_file)
        logger.info(f"All results saved to {os.path.abspath(output_file)}")


if __name__ == "__main__":
    main()