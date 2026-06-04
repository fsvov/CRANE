import os
import torch
from torch import nn
from tqdm import tqdm
from utils.metricsTop import MetricsTop, ConformalMetrics
from utils.en_model import CRANEModel, CRANEModelMVE, gaussian_nll_loss, freeze_backbone
import random
import numpy as np
from utils.data_loader import data_loader
from utils.conformal import (SplitConformalPredictor, MCAdaptiveConformalPredictor,
                              MondrianConformalPredictor, sentiment_group,
                              mc_dropout_interval, ClassificationConformalPredictor,
                              classification_set_metrics, classification_conditional_by_sentiment,
                              map_to_7class)
from collections import defaultdict
from utils.visualization import save_all_figures

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def count_trainable_params_by_module(model):
    module_params = defaultdict(int)
    for name, param in model.named_parameters():
        if param.requires_grad:
            module_name = name.split('.')[0]
            module_params[module_name] += param.numel()

    print(f"{'Module':<30} {'Trainable Params':>20}")
    print("=" * 55)
    total = 0
    for module_name, count in module_params.items():
        print(f"{module_name:<30} {count:>20,}")
        total += count
    print("=" * 55)
    print(f"{'Total':<30} {total:>20,}")


def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " %(key, src_dict[key]) 
    return dst_str

class EnConfig(object):
    def __init__(self,
                 seed=42,
                 batch_size=8,
                 learning_rate=5e-6,
                 fusion_method='v2',
                 dataset='mosi',
                 num_hidden_layer= 1,
                 model_save_path = './saved/',
                 train_mode='regression',
                 early_stop = 8,
                 dropout=0.3,
                ):
        self.seed = seed
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.fusion_method = fusion_method
        self.dataset = dataset
        self.num_hidden_layer = num_hidden_layer
        self.model_save_path = model_save_path
        self.train_mode = train_mode
        self.early_stop = early_stop
        self.dropout = dropout

class EnTrainer():
    def __init__(self, config):
        self.config = config
        self.criterion = nn.L1Loss() if config.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(config.train_mode).getMetics(config.dataset)


    def do_train(self, model, data_loader):
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        total_loss = 0
        for batch in tqdm(data_loader):
            text_inputs = batch["text_tokens"].to(device)
            text_mask = batch["text_masks"].to(device)
            audio_inputs = batch["audio_inputs"].to(device)
            audio_mask = batch["audio_masks"].to(device)
            targets = batch["targets"].to(device).view(-1, 1)
            optimizer.zero_grad()  # To zero out the gradients.
            outputs = model(text_inputs, text_mask, audio_inputs, audio_mask)
            loss = self.criterion(outputs, targets)
            total_loss += loss.item() * text_inputs.size(0)
            loss.backward()
            optimizer.step()
        total_loss = round(total_loss / len(data_loader.dataset), 4)
        return total_loss

    def do_test(self, model, data_loader, mode):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model.eval()
        y_pred = []
        y_true = []
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(data_loader):
                text_inputs = batch["text_tokens"].to(device)
                text_mask = batch["text_masks"].to(device)
                audio_inputs = batch["audio_inputs"].to(device)
                audio_mask = batch["audio_masks"].to(device)
                targets = batch["targets"].to(device).view(-1, 1)
                outputs = model(text_inputs, text_mask, audio_inputs, audio_mask)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()*text_inputs.size(0)
                y_pred.append(outputs.cpu())
                y_true.append(targets.cpu())
            total_loss = round(total_loss / len(data_loader.dataset), 4)
            print(mode+" >> loss: ",total_loss)
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            eval_results = {}
            eval_results = self.metrics(pred, true)
            print('%s: >> ' %('M') + dict_to_str(eval_results))
            eval_results['Loss'] = total_loss
            allocated = torch.cuda.max_memory_allocated(device) / 1024 ** 2
            reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 2
            print(f"Max memory allocated: {allocated:.2f} MB")
            print(f"Max memory reserved : {reserved:.2f} MB")
        return eval_results

    def do_mc_inference(self, model, data_loader, k=20):
        torch.cuda.empty_cache()
        all_preds = []
        y_true = []
        model.train()
        with torch.no_grad():
            for _ in range(k):
                batch_preds = []
                batch_targets = []
                for batch in data_loader:
                    text_inputs = batch["text_tokens"].to(device)
                    text_mask = batch["text_masks"].to(device)
                    audio_inputs = batch["audio_inputs"].to(device)
                    audio_mask = batch["audio_masks"].to(device)
                    targets = batch["targets"].to(device).view(-1, 1)
                    outputs = model(text_inputs, text_mask, audio_inputs, audio_mask)
                    batch_preds.append(outputs.cpu())
                    if _ == 0:
                        batch_targets.append(targets.cpu())
                all_preds.append(torch.cat(batch_preds).numpy())
                if _ == 0:
                    y_true = torch.cat(batch_targets).numpy()
        model.eval()
        all_preds = np.array(all_preds)
        y_pred = all_preds.mean(axis=0)
        mc_std = all_preds.std(axis=0)
        return y_pred, mc_std, y_true

    def do_train_mve(self, model, data_loader, epochs=5, lr=1e-4):
        """Fine-tune MVE variance head with frozen backbone."""
        model.train()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(data_loader, desc=f"MVE epoch {epoch+1}/{epochs}"):
                text_inputs = batch["text_tokens"].to(device)
                text_mask = batch["text_masks"].to(device)
                audio_inputs = batch["audio_inputs"].to(device)
                audio_mask = batch["audio_masks"].to(device)
                targets = batch["targets"].to(device).view(-1, 1)
                optimizer.zero_grad()
                outputs = model(text_inputs, text_mask, audio_inputs, audio_mask)
                loss = gaussian_nll_loss(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * text_inputs.size(0)
            avg_loss = total_loss / len(data_loader.dataset)
            print(f"  MVE epoch {epoch+1}: loss={avg_loss:.4f}")

    def do_mc_inference_modality(self, model, data_loader, modality='both', k=20):
        """MC inference with single modality: 'text', 'audio', or 'both'."""
        torch.cuda.empty_cache()
        all_preds = []
        y_true = []
        model.train()
        with torch.no_grad():
            for _ in range(k):
                batch_preds = []
                batch_targets = []
                for batch in data_loader:
                    text_inputs = batch["text_tokens"].to(device)
                    text_mask = batch["text_masks"].to(device)
                    audio_inputs = batch["audio_inputs"].to(device)
                    audio_mask = batch["audio_masks"].to(device)
                    targets = batch["targets"].to(device).view(-1, 1)

                    if modality == 'text':
                        audio_inputs = torch.zeros_like(audio_inputs)
                    elif modality == 'audio':
                        # Replace text with [PAD] tokens (token_id=1 for RoBERTa)
                        text_inputs = torch.ones_like(text_inputs)
                        text_mask = torch.zeros_like(text_mask)

                    outputs = model(text_inputs, text_mask, audio_inputs, audio_mask)
                    batch_preds.append(outputs.cpu())
                    if _ == 0:
                        batch_targets.append(targets.cpu())
                all_preds.append(torch.cat(batch_preds).numpy())
                if _ == 0:
                    y_true = torch.cat(batch_targets).numpy()
        model.eval()
        all_preds = np.array(all_preds)
        y_pred = all_preds.mean(axis=0)
        mc_std = all_preds.std(axis=0)
        return y_pred, mc_std, y_true

    def do_inference_mve(self, model, data_loader):
        """Deterministic inference for MVE model. Returns mean, std, true."""
        model.eval()
        means, stds, truths = [], [], []
        with torch.no_grad():
            for batch in tqdm(data_loader):
                text_inputs = batch["text_tokens"].to(device)
                text_mask = batch["text_masks"].to(device)
                audio_inputs = batch["audio_inputs"].to(device)
                audio_mask = batch["audio_masks"].to(device)
                targets = batch["targets"].to(device).view(-1, 1)
                outputs = model(text_inputs, text_mask, audio_inputs, audio_mask)
                means.append(outputs[:, 0:1].cpu())
                stds.append(torch.exp(outputs[:, 1:2] / 2).cpu())
                truths.append(targets.cpu())
        return (torch.cat(means).numpy(), torch.cat(stds).numpy(),
                torch.cat(truths).numpy())

def EnRun(config):
    print(f"config are as follows: \n"
          f"batch_size: {config.batch_size}, \n"
          f"fusion_method: {config.fusion_method}, \n"
          f"dataset: {config.dataset}, \n"
          f"dropout: {config.dropout}, \n"
          f"early_stop: {config.early_stop}, \n"
          f"learning_rate: {config.learning_rate}, \n"
          f"seed: {config.seed}, \n")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True
    os.makedirs(config.model_save_path, exist_ok=True)
    train_loader, test_loader, es_loader, cal_loader = data_loader(config.batch_size, config.dataset, config.seed)
    model = CRANEModel(config).to(device)
    #froze the data2vec
    for param in model.data2vec_model.parameters():
        param.requires_grad = False
    trainer = EnTrainer(config)
    #compute the trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters (PARA): {total_params}")
    count_trainable_params_by_module(model)
    lowest_eval_loss = 100
    highest_eval_acc = 0
    #custom the threshold
    acc2 = 0.80
    acc7 = 0.45

    num_model = 0
    epoch = 0
    best_epoch = 0
    while True:
        print('---------------------EPOCH: ', epoch, '--------------------')
        trainer.do_train(model, train_loader)
        epoch += 1
        eval_results = trainer.do_test(model, es_loader,"VAL")
        if eval_results['Loss']<lowest_eval_loss:
             lowest_eval_loss = eval_results['Loss']
             model_save_path_name = os.path.join(config.model_save_path, 'RH_loss.pth')
             torch.save(model.state_dict(), model_save_path_name)
             best_epoch = epoch
        if eval_results['Has0_acc_2']>=highest_eval_acc:
             highest_eval_acc = eval_results['Has0_acc_2']
             model_save_path_name_acc = os.path.join(config.model_save_path, 'RH_acc.pth')
             torch.save(model.state_dict(), model_save_path_name_acc)
        if eval_results['Has0_acc_2']>=acc2 and eval_results['Mult_acc_7']>=acc7:
             model_save_path = os.path.join(config.model_save_path, f'acc2&7high{num_model}.pth')
             torch.save(model.state_dict(), model_save_path)
             num_model += 1
        if epoch - best_epoch >= config.early_stop:
            break
    model.load_state_dict(torch.load(config.model_save_path + 'RH_acc.pth', weights_only=True))
    test_results_loss = trainer.do_test(model, test_loader, "TEST")
    print('\n%s: >> ' % ('TEST (highest val acc) ') + dict_to_str(test_results_loss))
    model.load_state_dict(torch.load(config.model_save_path + 'RH_loss.pth', weights_only=True))
    test_results_acc = trainer.do_test(model, test_loader, "TEST")
    print('\n%s: >> ' % ('TEST (lowest val loss) ') + dict_to_str(test_results_acc))
    for index in range(num_model):
        model.load_state_dict(torch.load(config.model_save_path + f'acc2&7high{index}.pth', weights_only=True))
        test_results_loss = trainer.do_test(model, test_loader, "TEST")
        print('\n%s: >> ' % (f'TEST (highest val acc2&acc7)[{index}] ') + dict_to_str(test_results_loss))

    # ============================================================
    # Conformal Prediction Evaluation
    # ============================================================
    print("\n" + "=" * 70)
    print(" CONFORMAL PREDICTION EVALUATION ")
    print("=" * 70)

    model.load_state_dict(torch.load(config.model_save_path + 'RH_loss.pth', weights_only=True))
    model.eval()

    # =====================================================
    # MC inference on cal and test sets
    # =====================================================
    print("Running MC inference on calibration set (K=20)...")
    y_pred_cal, mc_std_cal, y_true_cal = trainer.do_mc_inference(model, cal_loader, k=20)
    print("Running MC inference on test set (K=20)...")
    y_pred_test, mc_std_test, y_true_test = trainer.do_mc_inference(model, test_loader, k=20)

    # Sentiment groups for Mondrian CP
    cal_groups = sentiment_group(y_true_cal)
    test_groups = sentiment_group(y_true_test)

    alphas = [0.05, 0.10, 0.15, 0.20]

    # =====================================================
    # 1. Split Conformal (Constant Width)
    # =====================================================
    split_cp = SplitConformalPredictor()
    split_cp.calibrate(y_true_cal, y_pred_cal)

    print("\n--- 1. Split Conformal (Constant Width) ---")
    for alpha in alphas:
        lower, upper = split_cp.predict(y_pred_test, alpha)
        print(ConformalMetrics.format_results(y_true_test, y_pred_test, lower, upper, alpha))

    # =====================================================
    # 2. Adaptive Conformal (MC Dropout variance)
    # =====================================================
    adaptive_cp = MCAdaptiveConformalPredictor()
    adaptive_cp.calibrate(y_true_cal, y_pred_cal, mc_std_cal)

    print("\n--- 2. Adaptive Conformal (MC Dropout, K=20) ---")
    for alpha in alphas:
        lower, upper = adaptive_cp.predict(y_pred_test, mc_std_test, alpha)
        print(ConformalMetrics.format_results(y_true_test, y_pred_test, lower, upper, alpha))

    # =====================================================
    # 3. MC Dropout RAW (Gaussian assumption, no conformal)
    # =====================================================
    print("\n--- 3. MC Dropout RAW (Gaussian, no calibration) ---")
    for alpha in alphas:
        lower, upper = mc_dropout_interval(y_pred_test, mc_std_test, alpha)
        print(ConformalMetrics.format_results(y_true_test, y_pred_test, lower, upper, alpha, label="raw"))

    # =====================================================
    # 4. Mondrian Conformal (per-sentiment conditional)
    # =====================================================
    mondrian_cp = MondrianConformalPredictor()
    mondrian_cp.calibrate(y_true_cal, y_pred_cal, cal_groups)

    print("\n--- 4. Mondrian Conformal (Per-Sentiment Conditional) ---")
    for alpha in alphas:
        lower, upper = mondrian_cp.predict(y_pred_test, test_groups, alpha)
        print(ConformalMetrics.format_results(y_true_test, y_pred_test, lower, upper, alpha))

    # =====================================================
    # 5. MVE: train variance head, then adaptive conformal
    # =====================================================
    print("\n--- 5. MVE (Mean-Variance Estimation) ---")
    print("Initializing MVE model and loading backbone weights...")
    mve_model = CRANEModelMVE(config).to(device)
    # Copy backbone weights from trained model
    pretrained_dict = model.state_dict()
    mve_dict = mve_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in mve_dict and not k.startswith('var_head')}
    mve_dict.update(pretrained_dict)
    mve_model.load_state_dict(mve_dict)
    freeze_backbone(mve_model)

    print("Training MVE variance head...")
    trainer.do_train_mve(mve_model, es_loader, epochs=5, lr=1e-4)

    # Get MVE predictions
    print("Running MVE inference on calibration set...")
    mve_mean_cal, mve_std_cal, _ = trainer.do_inference_mve(mve_model, cal_loader)
    print("Running MVE inference on test set...")
    mve_mean_test, mve_std_test, _ = trainer.do_inference_mve(mve_model, test_loader)

    # MVE + Adaptive Conformal
    mve_cp = MCAdaptiveConformalPredictor()
    mve_cp.calibrate(y_true_cal, mve_mean_cal, mve_std_cal)

    for alpha in alphas:
        lower, upper = mve_cp.predict(mve_mean_test, mve_std_test, alpha)
        print(ConformalMetrics.format_results(y_true_test, mve_mean_test, lower, upper, alpha))

    # =====================================================
    # 6. Comparison table at α=0.10
    # =====================================================
    print("\n" + "=" * 70)
    print(" COMPARISON TABLE (α=0.10) ")
    print("=" * 70)
    print(f"{'Method':<35} {'Coverage':>10} {'Avg Width':>10} {'Med Width':>10} {'Int Score':>10}")
    print("-" * 75)

    def _row(label, yt, yp, lo, up, a=0.10):
        from utils.conformal import compute_coverage, compute_interval_width, compute_interval_score
        cov = compute_coverage(yt.flatten(), lo.flatten(), up.flatten())
        aw, mw = compute_interval_width(lo.flatten(), up.flatten())
        sc = compute_interval_score(yt.flatten(), lo.flatten(), up.flatten(), a)
        print(f"{label:<35} {cov:>10.4f} {aw:>10.4f} {mw:>10.4f} {sc:>10.4f}")

    split_lo, split_up = split_cp.predict(y_pred_test, 0.10)
    adapt_lo, adapt_up = adaptive_cp.predict(y_pred_test, mc_std_test, 0.10)
    raw_lo, raw_up = mc_dropout_interval(y_pred_test, mc_std_test, 0.10)
    mond_lo, mond_up = mondrian_cp.predict(y_pred_test, test_groups, 0.10)
    mve_lo, mve_up = mve_cp.predict(mve_mean_test, mve_std_test, 0.10)

    _row("Split Conformal (constant)", y_true_test, y_pred_test, split_lo, split_up)
    _row("Adaptive Conformal (MC Dropout)", y_true_test, y_pred_test, adapt_lo, adapt_up)
    _row("MC Dropout RAW (Gaussian)", y_true_test, y_pred_test, raw_lo, raw_up)
    _row("Mondrian Conformal (sentiment)", y_true_test, y_pred_test, mond_lo, mond_up)
    _row("MVE + Adaptive Conformal", y_true_test, mve_mean_test, mve_lo, mve_up)

    # Conditional coverage detail
    print("\n--- Conditional Coverage by Sentiment (α=0.10) ---")
    print("  Mondrian Conformal:")
    print(ConformalMetrics.format_conditional_sentiment(y_true_test, y_pred_test, mond_lo, mond_up))
    print("  Adaptive Conformal (MC Dropout):")
    print(ConformalMetrics.format_conditional_sentiment(y_true_test, y_pred_test, adapt_lo, adapt_up))

    print("\n--- Conditional Coverage by Prediction Bucket (Adaptive MC Dropout, α=0.10) ---")
    print(ConformalMetrics.format_conditional_bucket(y_pred_test, y_true_test, adapt_lo, adapt_up))

    # =====================================================
    # 7. Calibration Size Sensitivity Analysis
    # =====================================================
    print("\n" + "=" * 70)
    print(" CALIBRATION SIZE SENSITIVITY (α=0.10, seed=42) ")
    print("=" * 70)
    cal_sizes = [20, 40, 60, 80, 100, 140, 180, len(y_true_cal)]
    print(f"{'n_cal':>6}  {'Split_Cov':>10}  {'Split_W':>10}  {'Adapt_Cov':>10}  {'Adapt_W':>10}  {'Adapt_MW':>10}")
    print("-" * 70)
    n_cal_full = len(y_true_cal)
    for n in cal_sizes:
        if n > n_cal_full:
            continue
        idx = np.random.RandomState(42).choice(n_cal_full, n, replace=False)
        y_c_sub = y_true_cal[idx]
        y_p_sub = y_pred_cal[idx]
        s_sub = mc_std_cal[idx]

        sc = SplitConformalPredictor()
        sc.calibrate(y_c_sub, y_p_sub)
        sl, su = sc.predict(y_pred_test, 0.10)

        ac = MCAdaptiveConformalPredictor()
        ac.calibrate(y_c_sub, y_p_sub, s_sub)
        al, au = ac.predict(y_pred_test, mc_std_test, 0.10)

        from utils.conformal import compute_coverage, compute_interval_width
        scov, sw, _ = compute_coverage(y_true_test.flatten(), sl.flatten(), su.flatten()), *compute_interval_width(sl.flatten(), su.flatten())
        acov, aw, amw = compute_coverage(y_true_test.flatten(), al.flatten(), au.flatten()), *compute_interval_width(al.flatten(), au.flatten())
        print(f"{n:>6}  {scov:>10.4f}  {sw:>10.4f}  {acov:>10.4f}  {aw:>10.4f}  {amw:>10.4f}")

    # =====================================================
    # 8. Multimodal Uncertainty Decomposition
    # =====================================================
    print("\n" + "=" * 70)
    print(" MULTIMODAL UNCERTAINTY DECOMPOSITION (Adaptive CP, α=0.10) ")
    print("=" * 70)
    print("Running text-only MC inference...")
    yp_text, std_text, _ = trainer.do_mc_inference_modality(model, cal_loader, 'text', k=20)
    yp_text_t, std_text_t, _ = trainer.do_mc_inference_modality(model, test_loader, 'text', k=20)

    print("Running audio-only MC inference...")
    yp_audio, std_audio, _ = trainer.do_mc_inference_modality(model, cal_loader, 'audio', k=20)
    yp_audio_t, std_audio_t, _ = trainer.do_mc_inference_modality(model, test_loader, 'audio', k=20)

    # Calibrate and evaluate each modality
    cp_text = MCAdaptiveConformalPredictor()
    cp_text.calibrate(y_true_cal, yp_text, std_text)
    tl, tu = cp_text.predict(yp_text_t, std_text_t, 0.10)
    from utils.conformal import compute_coverage, compute_interval_width
    t_cov = compute_coverage(y_true_test.flatten(), tl.flatten(), tu.flatten())
    t_aw, t_mw = compute_interval_width(tl.flatten(), tu.flatten())

    cp_audio = MCAdaptiveConformalPredictor()
    cp_audio.calibrate(y_true_cal, yp_audio, std_audio)
    al, au = cp_audio.predict(yp_audio_t, std_audio_t, 0.10)
    a_cov = compute_coverage(y_true_test.flatten(), al.flatten(), au.flatten())
    a_aw, a_mw = compute_interval_width(al.flatten(), au.flatten())

    # Multimodal (already computed)
    m_lo, m_up = adaptive_cp.predict(y_pred_test, mc_std_test, 0.10)
    m_cov = compute_coverage(y_true_test.flatten(), m_lo.flatten(), m_up.flatten())
    m_aw, m_mw = compute_interval_width(m_lo.flatten(), m_up.flatten())

    print(f"{'Modality':<15} {'Coverage':>10} {'Avg Width':>10} {'Med Width':>10}")
    print("-" * 50)
    print(f"{'Text-only':<15} {t_cov:>10.4f} {t_aw:>10.4f} {t_mw:>10.4f}")
    print(f"{'Audio-only':<15} {a_cov:>10.4f} {a_aw:>10.4f} {a_mw:>10.4f}")
    print(f"{'Multimodal':<15} {m_cov:>10.4f} {m_aw:>10.4f} {m_mw:>10.4f}")
    width_reduction = (m_mw - min(t_mw, a_mw)) / max(t_mw, a_mw) * 100
    print(f"\nMultimodal median width reduction vs best single-modality: {width_reduction:.1f}%")

    # Uncertainty attribution
    print("\n--- Uncertainty Attribution (test set, per-sample std) ---")
    std_text_corr = np.corrcoef(std_text_t.flatten(), std_audio_t.flatten())[0, 1]
    print(f"Corr(text_std, audio_std): {std_text_corr:.4f}")
    avg_text_std = np.mean(std_text_t)
    avg_audio_std = np.mean(std_audio_t)
    avg_multi_std = np.mean(mc_std_test)
    print(f"Avg MC-std:  Text={avg_text_std:.4f}  Audio={avg_audio_std:.4f}  Multi={avg_multi_std:.4f}")
    text_contrib = avg_text_std / (avg_text_std + avg_audio_std) * 100
    audio_contrib = avg_audio_std / (avg_text_std + avg_audio_std) * 100
    print(f"Uncertainty contribution:  Text={text_contrib:.1f}%  Audio={audio_contrib:.1f}%")

    # UBG learned confidence
    print("\n--- UBG Learned Modality Confidence (test set) ---")
    model.eval()
    conf_text_vals, conf_audio_vals, conf_labels = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            text_inputs = batch["text_tokens"].to(device)
            text_mask = batch["text_masks"].to(device)
            audio_inputs = batch["audio_inputs"].to(device)
            audio_mask = batch["audio_masks"].to(device)
            targets = batch["targets"].to(device).view(-1, 1)
            _ = model(text_inputs, text_mask, audio_inputs, audio_mask)
            ct = model.ubi_gated_model._last_conf_text.cpu().numpy()
            ca = model.ubi_gated_model._last_conf_audio.cpu().numpy()
            conf_text_vals.append(ct)
            conf_audio_vals.append(ca)
            conf_labels.append(targets.cpu().numpy())
    conf_t = np.concatenate(conf_text_vals).flatten()
    conf_a = np.concatenate(conf_audio_vals).flatten()
    conf_lab = np.concatenate(conf_labels).flatten()
    print(f"  Avg conf_text={np.mean(conf_t):.4f}  Avg conf_audio={np.mean(conf_a):.4f}")
    print(f"  Corr(conf_text, conf_audio)={np.corrcoef(conf_t, conf_a)[0,1]:.4f}")
    neg_mask = conf_lab < 0
    pos_mask = conf_lab > 0
    print(f"  Negative samples: conf_text={np.mean(conf_t[neg_mask]):.4f}  conf_audio={np.mean(conf_a[neg_mask]):.4f}")
    print(f"  Positive samples: conf_text={np.mean(conf_t[pos_mask]):.4f}  conf_audio={np.mean(conf_a[pos_mask]):.4f}")

    # =====================================================
    # 9. Deep Ensemble (if available)
    # =====================================================
    ensemble_seeds = [42, 123, 456, 789]
    ensemble_ckpt_paths = [
        os.path.join('./saved_ensemble', f'seed{s}', 'RH_loss.pth')
        for s in ensemble_seeds]
    ensemble_models_exist = all(os.path.exists(p) for p in ensemble_ckpt_paths)

    if ensemble_models_exist:
        print("\n" + "=" * 70)
        print(" DEEP ENSEMBLE (4 seeds) + Adaptive Conformal ")
        print("=" * 70)
        ens_means_cal, ens_stds_cal = [], []
        ens_means_test, ens_stds_test = [], []

        for s in ensemble_seeds:
            ckpt_path = os.path.join('./saved_ensemble', f'seed{s}', 'RH_loss.pth')
            print(f"Loading ensemble member seed={s} from {ckpt_path}...")
            ckpt = torch.load(ckpt_path, weights_only=True)
            model.load_state_dict(ckpt)
            model.eval()
            yp_c, std_c, _ = trainer.do_mc_inference(model, cal_loader, k=20)
            yp_t, std_t, _ = trainer.do_mc_inference(model, test_loader, k=20)
            ens_means_cal.append(yp_c)
            ens_stds_cal.append(std_c)
            ens_means_test.append(yp_t)
            ens_stds_test.append(std_t)

        ens_mean_cal = np.mean(ens_means_cal, axis=0)
        ens_std_cal = np.std(ens_means_cal, axis=0)  # between-model std
        ens_mean_test = np.mean(ens_means_test, axis=0)
        ens_std_test = np.std(ens_means_test, axis=0)

        ens_cp = MCAdaptiveConformalPredictor()
        ens_cp.calibrate(y_true_cal, ens_mean_cal, ens_std_cal)

        print("\nDeep Ensemble + Adaptive Conformal (α=0.10):")
        for alpha in alphas:
            lower, upper = ens_cp.predict(ens_mean_test, ens_std_test, alpha)
            print(ConformalMetrics.format_results(y_true_test, ens_mean_test, lower, upper, alpha, label="ens"))

        # Comparison: MC Dropout vs Ensemble
        from utils.conformal import compute_coverage, compute_interval_width
        el, eu = ens_cp.predict(ens_mean_test, ens_std_test, 0.10)
        e_cov = compute_coverage(y_true_test.flatten(), el.flatten(), eu.flatten())
        e_aw, e_mw = compute_interval_width(el.flatten(), eu.flatten())
        print(f"\n{'Method':<35} {'Coverage':>10} {'Avg Width':>10} {'Med Width':>10}")
        print("-" * 60)
        print(f"{'Adaptive (MC Dropout)':<35} {m_cov:>10.4f} {m_aw:>10.4f} {m_mw:>10.4f}")
        print(f"{'Adaptive (Deep Ensemble)':<35} {e_cov:>10.4f} {e_aw:>10.4f} {e_mw:>10.4f}")
        width_change = (e_mw - m_mw) / m_mw * 100
        print(f"\nEnsemble width change vs MC Dropout: {width_change:+.1f}%")
    else:
        print("\nDeep Ensemble checkpoints not found — skipping.")
        print(f"Run run_ensemble.sh to train {len(ensemble_seeds)} seeds.")

    # =====================================================
    # 10. Classification Conformal Prediction Sets
    # =====================================================
    print("\n" + "=" * 70)
    print(" CLASSIFICATION CONFORMAL: 7-CLASS PREDICTION SETS ")
    print("=" * 70)

    cls_cp = ClassificationConformalPredictor()
    cls_cp.calibrate(y_true_cal, y_pred_cal)

    for alpha in alphas:
        pred_sets = cls_cp.predict(y_pred_test, alpha)
        metrics = classification_set_metrics(y_true_test, pred_sets)
        print(f"\nα={alpha:.2f}:  Coverage={metrics['coverage']:.4f}  "
              f"Avg_Set_Size={metrics['avg_set_size']:.2f}  "
              f"Med_Set_Size={metrics['med_set_size']}  "
              f"Singleton_Rate={metrics['singleton_rate']:.4f}  "
              f"Max_Set_Size={metrics['max_set_size']}")
        print(f"  Size distribution: {metrics['size_distribution']}")

    # Examples at α=0.10
    print("\n--- Example Prediction Sets (α=0.10) ---")
    example_sets = cls_cp.predict(y_pred_test, 0.10)
    yt_disc = map_to_7class(y_true_test)
    # Pick representative examples: singleton, ambiguous, wide
    by_size = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}
    for i, s in enumerate(example_sets):
        if len(s) <= 7:
            by_size[len(s)].append(i)
    for sz in [1, 2, 3, 5, 7]:
        if by_size[sz]:
            idx = by_size[sz][0]
            print(f"  Size={sz} | Pred={y_pred_test[idx][0]:.2f} | True={yt_disc[idx]} | Set={example_sets[idx]}")

    # Conditional by sentiment
    print("\n--- Conditional by Sentiment (α=0.10) ---")
    cond = classification_conditional_by_sentiment(y_true_test, example_sets)
    for label, info in cond.items():
        print(f"  {label:<10}  Count={info['count']:>4}  "
              f"Coverage={info['coverage']:.4f}  Avg_Set_Size={info['avg_size']:.2f}")

    # =====================================================
    # Collect visualization data
    # =====================================================
    from utils.conformal import compute_coverage, compute_interval_width, compute_interval_score

    viz = {}

    # --- Figure 1: Coverage-Width trade-off ---
    cw_results = {}
    for name, cp in [('Split', split_cp), ('Adaptive', adaptive_cp),
                      ('Mondrian', mondrian_cp), ('MVE', mve_cp)]:
        pts = {}
        for a in alphas:
            if name == 'Mondrian':
                lo, up = cp.predict(y_pred_test, test_groups, a)
            elif name == 'MVE':
                lo, up = cp.predict(mve_mean_test, mve_std_test, a)
            elif name == 'Adaptive':
                lo, up = cp.predict(y_pred_test, mc_std_test, a)
            else:  # Split
                lo, up = cp.predict(y_pred_test, a)
            cov = compute_coverage(y_true_test.flatten(), lo.flatten(), up.flatten())
            _, mw = compute_interval_width(lo.flatten(), up.flatten())
            pts[a] = (cov, mw)
        cw_results[name] = pts

    # MC RAW
    raw_pts = {}
    for a in alphas:
        lo, up = mc_dropout_interval(y_pred_test, mc_std_test, a)
        cov = compute_coverage(y_true_test.flatten(), lo.flatten(), up.flatten())
        _, mw = compute_interval_width(lo.flatten(), up.flatten())
        raw_pts[a] = (cov, mw)
    cw_results['MC RAW'] = raw_pts

    # Ensemble (if available)
    if ensemble_models_exist:
        ens_pts = {}
        for a in alphas:
            lo, up = ens_cp.predict(ens_mean_test, ens_std_test, a)
            cov = compute_coverage(y_true_test.flatten(), lo.flatten(), up.flatten())
            _, mw = compute_interval_width(lo.flatten(), up.flatten())
            ens_pts[a] = (cov, mw)
        cw_results['Ensemble'] = ens_pts

    viz['coverage_width'] = {
        'methods': list(cw_results.keys()),
        'alphas': alphas,
        'results': cw_results,
    }

    # --- Figure 2: Calibration sensitivity (recompute & store) ---
    cal_cov_split, cal_cov_adapt, cal_w_split, cal_w_adapt, cal_n = [], [], [], [], []
    for n in cal_sizes:
        if n > n_cal_full:
            continue
        idx = np.random.RandomState(42).choice(n_cal_full, n, replace=False)
        sc_tmp = SplitConformalPredictor()
        sc_tmp.calibrate(y_true_cal[idx], y_pred_cal[idx])
        sl, su = sc_tmp.predict(y_pred_test, 0.10)
        ac_tmp = MCAdaptiveConformalPredictor()
        ac_tmp.calibrate(y_true_cal[idx], y_pred_cal[idx], mc_std_cal[idx])
        al, au = ac_tmp.predict(y_pred_test, mc_std_test, 0.10)
        cal_n.append(n)
        cal_cov_split.append(compute_coverage(y_true_test.flatten(), sl.flatten(), su.flatten()))
        cal_cov_adapt.append(compute_coverage(y_true_test.flatten(), al.flatten(), au.flatten()))
        _, sw = compute_interval_width(sl.flatten(), su.flatten())
        _, amw = compute_interval_width(al.flatten(), au.flatten())
        cal_w_split.append(sw)
        cal_w_adapt.append(amw)

    viz['calibration_sensitivity'] = {
        'n_cal': cal_n, 'split_cov': cal_cov_split, 'adapt_cov': cal_cov_adapt,
        'split_w': cal_w_split, 'adapt_mw': cal_w_adapt,
    }

    # --- Figure 3: UBG confidence ---
    viz['ubg_confidence'] = (conf_t, conf_a,
                              ['negative' if l < 0 else 'neutral' if l == 0 else 'positive'
                               for l in conf_lab])

    # --- Figure 4: Residual distribution ---
    cal_residuals = np.abs(y_true_cal.flatten() - y_pred_cal.flatten())
    test_residuals = np.abs(y_true_test.flatten() - y_pred_test.flatten())
    sl, su = split_cp.predict(y_pred_test, 0.10)
    q_val = (su[0] - sl[0]) / 2  # half-width
    viz['residuals'] = (cal_residuals, test_residuals, q_val, 0.10)

    # --- Figure 5: Conditional coverage ---
    adapt_lo, adapt_up = adaptive_cp.predict(y_pred_test, mc_std_test, 0.10)
    from utils.conformal import conditional_coverage_by_sentiment, conditional_coverage_by_bucket
    sent_cond = conditional_coverage_by_sentiment(y_true_test, y_pred_test, adapt_lo, adapt_up)
    buck_cond = conditional_coverage_by_bucket(y_pred_test, y_true_test, adapt_lo, adapt_up)
    viz['conditional_coverage'] = (sent_cond, buck_cond)

    # --- Figure 6: Width vs |ŷ| ---
    adapt_w = adapt_up - adapt_lo
    covered = (y_true_test.flatten() >= adapt_lo.flatten()) & (y_true_test.flatten() <= adapt_up.flatten())
    viz['width_vs_magnitude'] = (y_pred_test, adapt_w, covered, 0.10)

    # --- Figure 7: Prediction set sizes ---
    set_dists = {}
    for a in alphas:
        ps = cls_cp.predict(y_pred_test, a)
        metrics = classification_set_metrics(y_true_test, ps)
        set_dists[a] = metrics['size_distribution']
    viz['prediction_sets'] = (set_dists, 0.10)

    # --- Figure 8: Reliability diagram ---
    viz['reliability'] = (y_pred_test, y_true_test, adapt_w, mc_std_test)

    # Generate all figures
    save_all_figures(viz)

    print("=" * 70)




