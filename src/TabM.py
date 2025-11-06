import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score, matthews_corrcoef, roc_curve, auc, average_precision_score
import csv
import argparse
from tabm_reference import Model, make_parameter_groups


def get_roc_scores(scores: np.array, labels: np.array):
    fpr, tpr, _ = roc_curve(labels, scores)
    arc = auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    low = tpr[np.where(fpr < 0.05)[0][-1]]
    return arc, acc, low

def get_roc_auc_scores(scores: np.array, labels: np.array):
    fpr, tpr, _ = roc_curve(labels, scores)
    arc = auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    low = tpr[np.where(fpr < 0.05)[0][-1]]
    return arc, acc, low, fpr, tpr


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser(description="Run Step 4. Evaluation Strategy - TabM")
    parser.add_argument("--seed", type=int, default=42, help="randoom seed setting")
    parser.add_argument("--model_name", type=str, default="Deepseek-math-7b-rl", help="The model was used in the experiment and will be evaluated.")
    parser.add_argument("--output_dir", type=str, default="output", help="evaluation output dir")
    parser.add_argument("--dataset_name", type=str, default="TEST")
    
    args = parser.parse_args()

    set_seed(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_dir = os.path.join(args.output_dir, args.dataset_name, "evaluation", "TabM")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{args.model_name}.csv')

    with open(output_path, 'a+', encoding='utf-8', newline='') as csvf:
        wr = csv.writer(csvf)
            
        train_path = os.path.join(args.output_dir, "REF", "labeling", f"{args.model_name}_labeled.json")
        with open(train_path, 'r') as f:
            train_data = json.load(f)
            # print("Train sample count: ", len(train_data))

        test_path = os.path.join(args.output_dir, args.dataset_name, "labeling", f"{args.model_name}_labeled.json")
        if not os.path.isfile(test_path):
            return
        with open(test_path, 'r') as f:
            test_data = json.load(f)
            # print("Test sample count: ", len(test_data))


        X_train_list = []
        y_train_list = []
        for item in train_data:
            X_train_list.append(item['CLAWS'])
            y_train_list.append(item['label'])
    
        X_train = np.array(X_train_list)
        y_train = np.array(y_train_list)
        unique_labels, counts = np.unique(y_train, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"Label: {label}, Count: {count}")

        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        num_classes = len(np.unique(y_train_encoded))
        print(f"# of classes: {num_classes}")

        all_zero_val = le.transform(["Hallucinated_Solution"])[0]
        y_train_binary = (y_train_encoded != all_zero_val).astype(np.int64)


        with open(test_path, 'r') as f:
            test_data = json.load(f)
            print("Test sample count: ", len(test_data))

        X_test_list = []
        y_test_list = []

        for item in test_data:
            X_test_list.append(item['CLAWS'])
            y_test_list.append(item['label'])
            
        X_test = np.array(X_test_list)
        y_test = np.array(y_test_list)
        y_test_encoded = le.transform(y_test)

        y_test_binary = (y_test_encoded != all_zero_val).astype(np.int64)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test  = scaler.transform(X_test).astype(np.float32)


        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded)
        y_train_binary_sub = (y_train_sub != all_zero_val).astype(np.int64)
        y_val_binary = (y_val != all_zero_val).astype(np.int64)

        X_test_sub = X_test
        y_test_multi = y_test_encoded
        y_test_binary = (y_test_encoded != all_zero_val).astype(np.int64)

        if num_classes == 2:
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
            X_train = X_train_sub
            y_train = y_train_binary_sub
            X_val = X_val
            y_val = y_val_binary
            X_test = X_test_sub
            y_test = y_test_binary
            print("Model parameter: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
            
        else:
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
            X_train = X_train_sub
            y_train = y_train_sub
            X_val = X_val
            y_val = y_val
            X_test = X_test_sub
            y_test = y_test_multi
            print("Model parameter: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

        print(f"\n{args.dataset_name} {args.model_name} {args.method}")
        wr.writerow([args.dataset_name, args.model_name, args.method])

        model.to(device)
        optimizer = torch.optim.AdamW(make_parameter_groups(model), lr=2e-3, weight_decay=3e-4)
        best_val_acc = 0.0
        best_test_metrics = (0.0, 0.0)  # (accuracy, f1)

        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_ds   = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        test_ds  = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)
        test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)

        n_epochs = 50
        for epoch in range(1, n_epochs+1):
            model.train()
            total_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()

                y_pred = model(batch_X, None)  # shape: (batch, k, n_classes)
                b, k, n_cls = y_pred.shape
                loss = F.cross_entropy(y_pred.reshape(b * k, n_cls),
                                    batch_y.unsqueeze(1).repeat(1, k).reshape(b * k))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    outputs = model(batch_X, None)  # (batch, k, n_classes)
                    outputs_prob = F.softmax(outputs, dim=-1)
                    outputs_mean = outputs_prob.mean(dim=1)
                    preds = outputs_mean.argmax(dim=1)
                    correct += (preds == batch_y).sum().item()
                    total += batch_y.size(0)
            val_acc = correct / total
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                y_true_all = []
                y_pred_all = []
                y_pred_probs_all = []
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        batch_X = batch_X.to(device)
                        batch_y = batch_y.to(device)
                        outputs = model(batch_X, None)
                        outputs_prob = F.softmax(outputs, dim=-1)
                        outputs_mean = outputs_prob.mean(dim=1)
                        preds = outputs_mean.argmax(dim=1)
                        y_true_all.append(batch_y.cpu().numpy())
                        y_pred_all.append(preds.cpu().numpy())
                        y_pred_probs_all.append(outputs_mean.cpu().numpy())
                y_true_all = np.concatenate(y_true_all)
                y_pred_all = np.concatenate(y_pred_all)
                y_pred_probs_all = np.concatenate(y_pred_probs_all)
                test_acc = float((y_true_all == y_pred_all).mean())
                if num_classes == 2:
                    test_f1 = sklearn.metrics.f1_score(y_true_all, y_pred_all, average='binary')
                else:
                    test_f1 = sklearn.metrics.f1_score(y_true_all, y_pred_all, average='macro')
                best_test_metrics = (test_acc, test_f1)
                print(f"[Epoch {epoch}] new best val_acc: {val_acc*100:.2f}% → Test: acc={test_acc*100:.2f}%, f1={test_f1:.4f}")
        best_test_acc, best_test_f1 = best_test_metrics
        print(f"\nAccuracy: {best_test_acc*100:.2f}%, F1-score: {best_test_f1:.4f}\n")
        
        print("Test Accuracy:", best_test_acc)
        f1_weighted = f1_score(y_true_all, y_pred_all, average='weighted')
        f1_macro = f1_score(y_true_all, y_pred_all, average='macro')
        f1_micro = f1_score(y_true_all, y_pred_all, average='micro')
        
        print("F1 Score (weighted):", f1_weighted)
        wr.writerow(["F1 Score (weighted):", f1_weighted])
        print("F1 Score (macro):", f1_macro)
        wr.writerow(["F1 Score (macro):", f1_macro])
        print("F1 Score (micro):", f1_micro)
        wr.writerow(["F1 Score (micro):", f1_micro])

        # balanced_acc = balanced_accuracy_score(y_true_all, y_pred_all)
        # print("Balanced Accuracy:", balanced_acc)
        # mcc = matthews_corrcoef(y_true_all, y_pred_all)
        # print("Matthews Correlation Coefficient:", mcc)
        
        if num_classes > 2:
            auprc_list = []
            for i in range(num_classes):
                y_true_one_hot = np.eye(num_classes)[y_true_all]
                ap = average_precision_score(y_true_one_hot[:, i], y_pred_probs_all[:, i])
                auprc_list.append(ap)
                class_name = le.classes_[i]
                print(f"Class '{class_name}' Average Precision (AUPRC): {ap:.4f}")
            auprc_macro = np.mean(auprc_list)
            print("Multi-class Average Precision (macro-average):", auprc_macro)
            wr.writerow(["Multi-class Average Precision (macro-average):", auprc_macro.item()])

            roc_auc_list = []
            roc_auc_dict = {}
            for i in range(num_classes):
                y_true_one_hot = np.eye(num_classes)[y_true_all]
                fpr, tpr, _ = roc_curve(y_true_one_hot[:, i], y_pred_probs_all[:, i])
                roc_auc_val = auc(fpr, tpr)
                roc_auc_list.append(roc_auc_val)
                class_name = le.classes_[i]
                roc_auc_dict[class_name] = roc_auc_val
                print(f"Class '{class_name}': ROC AUC = {roc_auc_val:.4f}")
            roc_auc_overall = np.mean(roc_auc_list)
            print("Multi-class ROC AUC (macro-average):", roc_auc_overall)
            wr.writerow(["Multi-class ROC AUC (macro-average):", roc_auc_overall.item()])
        else:
            
            fpr, tpr, _ = roc_curve((y_true_all==1).astype(int), y_pred_probs_all[:, 1])
            roc_auc_val = auc(fpr, tpr)
            ap_val = average_precision_score((y_true_all==1).astype(int), y_pred_probs_all[:, 1])

            print(f"Binary Average Precision (macro-average): {ap_val:.4f}")
            wr.writerow(["Binary Average Precision (macro-average):", ap_val])
            print(f"Binary ROC AUC: {roc_auc_val:.4f}")
            wr.writerow(["Binary ROC AUC (macro-average):", roc_auc_val])
        
        print("\nClassification Report:")
        print(classification_report(y_true_all, y_pred_all, target_names=le.classes_, zero_division=0))
        wr.writerow([classification_report(y_true_all, y_pred_all, target_names=le.classes_, zero_division=0)])
        print(f"Accuracy: {best_test_acc*100:.2f}%, F1-score: {best_test_f1:.4f}")


if __name__ == "__main__":
    main()