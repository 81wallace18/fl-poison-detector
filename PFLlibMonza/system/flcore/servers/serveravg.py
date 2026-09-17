import time
import uuid
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import numpy as np
import torch
from collections import Counter
import copy
import csv
import os
from flcore.attack.attack import model_alie, verify_alie_attack
from flcore.detector import fl_save
from flcore.detector.cc_mlp import ClientCheckMLP
from flcore.detector.context_features import (
    _eval_state_dict,
    _canonical_model_state,
)

# Rollback safety net: revert aggregation if val accuracy drops by more than
# this many percentage points relative to the best accuracy seen so far.
_ROLLBACK_DROP_PP = 0.15  # 15 percentage points
class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.fpr_frr_results = []
        self.run_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        self.disable_quarantine = bool(getattr(args, 'disable_quarantine', False) or (os.environ.get('DISABLE_QUARANTINE', '0') == '1'))

        # Open the CSV file in append mode to save results over time
        suffix = f"{self.cc}_noquarantine" if self.disable_quarantine else f"{self.cc}"
        if self.cc in (2, 3, 7, 8):
            self.csv_filename = f'fpr_frr_results_{suffix}.csv'
        else:
            self.csv_filename = 'f.csv'
        self._ensure_csv_header(
            self.csv_filename,
            ['RunID', 'Round', 'DetectionFPR', 'DetectionFRR', 'QuarantineFPR', 'QuarantineFRR'],
        )
        self.cc_detail_filename = f'cc_detail_results_{suffix}.csv'
        self.cc_type_filename = f'cc_type_results_{suffix}.csv'
        if self.cc in (2, 3, 7, 8):
            detail_header = [
                'RunID', 'Round', 'CC', 'ClientID', 'AttackType', 'IsMaliciousRound',
                'MaliciousGroup', 'Removed', 'Reason', 'MLPHit', 'MLPScore',
                'MLPLabelScore', 'MLPBinaryHit', 'MLPLabelHit',
                'BinaryThreshold', 'LabelThreshold', 'DecisionRule', 'BaselineScore',
            ]
            self._ensure_csv_header(self.cc_detail_filename, detail_header)
            self._ensure_csv_header(
                self.cc_type_filename,
                ['RunID', 'Round', 'CC', 'AttackType', 'Total', 'Removed', 'Rate', 'Metric'],
            )

        self.dump_dir = getattr(args, 'dump_state_dicts', '') or ''
        self.dump_start_round = int(getattr(args, 'dump_start_round', 0))
        self.client_check = None
        if self.cc == 7:
            detector_dir = getattr(args, 'detector_dir', '') or ''
            if not detector_dir:
                raise ValueError("cc=7 requer --detector_dir apontando pro MLP artifacts dir (ex: jpt/detector_mlp_monza_cnn_mnist/).")
            print(f"[cc=7] Carregando detector MLP de {detector_dir}")
            self.client_check = ClientCheckMLP(
                detector_dir,
                threshold_key=getattr(args, 'mlp_threshold_key', 'threshold_label_fpr05'),
                threshold_value=getattr(args, 'mlp_threshold_value', None),
            )

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()

    def _ensure_csv_header(self, filename, header):
        if os.path.exists(filename):
            with open(filename, newline='') as file:
                current_header = next(csv.reader(file), [])
            if current_header != header:
                legacy_name = f"{filename}.legacy_{self.run_id}"
                os.rename(filename, legacy_name)
                print(f"[cc={self.cc}] Arquivando CSV com header antigo: {legacy_name}")
        if not os.path.exists(filename):
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)

    def save_fpr_frr_to_csv(self, round_number, detection_fpr, detection_frr,
                            quarantine_fpr='', quarantine_frr=''):
        """
        Saves per-round detection FPR/FRR (paper Eq 14/15, headline) and the
        quarantine-occupancy snapshot (diagnostic) to a CSV file for each round.
        """
        with open(self.csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                self.run_id, round_number,
                detection_fpr, detection_frr,
                quarantine_fpr, quarantine_frr,
            ])

    def save_cc_detail_to_csv(self, rows):
        if not rows:
            return
        with open(self.cc_detail_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            for row in rows:
                writer.writerow([
                    self.run_id, row['round'], row['cc'], row['client_id'], row['attack_type'],
                    int(row['is_malicious_round']), int(row['malicious_group']),
                    int(row['removed']), row['reason'], int(row['mlp_hit']),
                    row['mlp_score'], row.get('mlp_label_score', 0.0),
                    int(row.get('mlp_binary_hit', False)),
                    int(row.get('mlp_label_hit', False)),
                    row.get('binary_threshold', ''), row.get('label_threshold', ''),
                    row.get('decision_rule', 'binary'),
                    row.get('baseline_score', 0.0),
                ])

    def save_cc_type_to_csv(self, round_number, rows):
        if not rows:
            return
        grouped = {}
        for row in rows:
            attack_type = row['attack_type']
            bucket = grouped.setdefault(attack_type, {'total': 0, 'removed': 0})
            bucket['total'] += 1
            bucket['removed'] += int(row['removed'])
        with open(self.cc_type_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            for attack_type in sorted(grouped):
                bucket = grouped[attack_type]
                total = bucket['total']
                removed = bucket['removed']
                rate = removed / total if total else 0.0
                metric = 'FPR' if attack_type == 'benign' else 'recall'
                writer.writerow([self.run_id, round_number, self.cc, attack_type, total, removed, rate, metric])

    def _build_cc_detail_rows(self, round_number, all_client_ids, removed_clients, reason, scores=None):
        clients_by_id = {c.id: c for c in self.clients}
        removed_set = set(int(cid) for cid in removed_clients)
        scores = scores or {}
        rows = []
        for client_id in all_client_ids:
            client = clients_by_id.get(int(client_id))
            attack_type = getattr(client, 'last_attack_type', 'unknown')
            is_malicious_round = bool(getattr(client, 'is_malicious', False))
            removed = int(client_id) in removed_set
            rows.append({
                'round': round_number,
                'cc': self.cc,
                'client_id': int(client_id),
                'attack_type': attack_type,
                'is_malicious_round': is_malicious_round,
                'malicious_group': int(client_id) in self.index_malicious,
                'removed': removed,
                'reason': reason if removed else 'none',
                'mlp_hit': False,
                'mlp_score': 0.0,
                'mlp_label_score': 0.0,
                'mlp_binary_hit': False,
                'mlp_label_hit': False,
                'binary_threshold': '',
                'label_threshold': '',
                'decision_rule': reason,
                'baseline_score': float(scores.get(int(client_id), 0.0)),
            })
        return rows

    def compute_upload_fpr_frr(self, rows):
        """Per-round detection FPR/FRR (paper Eq 14/15) -- the headline metric.

        Computed only over the clients that uploaded this round: a removal of a client
        that attacked this round is a TP, removal of a benign upload is a FP, etc.
        This is what Table 4 of the MONZA paper reports (DetectionFPR/DetectionFRR).
        """
        if not rows:
            return '', ''
        FP = TP = FN = TN = 0
        for row in rows:
            removed = bool(row['removed'])
            is_malicious = bool(row['is_malicious_round'])
            if removed and not is_malicious:
                FP += 1
            elif removed and is_malicious:
                TP += 1
            elif not removed and is_malicious:
                FN += 1
            else:
                TN += 1
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        frr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
        return fpr, frr

    def set_client_quarantine(self, client_id):
        if getattr(self, 'disable_quarantine', False):
            return
        self.client_quarantine_dict[client_id]['quarentena'] = self.client_quarantine_dict[client_id]['quarentena'] +1
        self.client_quarantine_dict[client_id]['roundsQuarent'] = min(2, 2 ** self.client_quarantine_dict[client_id]['quarentena'])

    def decrease_quarentine(self, client_id):
        if self.client_quarantine_dict[client_id]['roundsQuarent'] ==0:
            self.client_quarantine_dict[client_id]['roundsQuarent'] = 0
        else:
            self.client_quarantine_dict[client_id]['roundsQuarent'] = self.client_quarantine_dict[client_id]['roundsQuarent'] -1
    def compute_fpr_frr(self):
        """
        Quarantine-occupancy snapshot (DIAGNOSTIC, NOT the paper detection FPR/FRR).

        Counts how many of the 100 clients are currently held in quarantine vs the
        designated-malicious set. Because quarantine duration grows exponentially
        (2**n rounds, see set_client_quarantine), recurrent benign clients stay
        quarantined indefinitely, so this metric snowballs and saturates over the run.
        Saved as QuarantineFPR/QuarantineFRR. For the paper metric use
        compute_upload_fpr_frr (DetectionFPR/DetectionFRR).
        """
        FP = 0  # Falsos positivos: clientes em quarentena mas não maliciosos
        TP = 0  # Verdadeiros positivos: clientes em quarentena e maliciosos
        FN = 0  # Falsos negativos: maliciosos não detectados
        TN = 0  # Verdadeiros negativos: não maliciosos e não em quarentena

        for client_id in range(self.num_clients):
            in_quarantine = self.client_quarantine_dict[client_id]['roundsQuarent'] > 0
            is_malicious = client_id in self.index_malicious

            if in_quarantine and not is_malicious:
                FP += 1
            elif in_quarantine and is_malicious:
                TP += 1
            elif not in_quarantine and is_malicious:
                FN += 1
            elif not in_quarantine and not is_malicious:
                TN += 1

        # Evitar divisão por zero
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        FRR = FN / (FN + TP) if (FN + TP) > 0 else 0

        return FPR, FRR

    def compute_fpr_frr_cluster(self, removed_clients, cluster_tuples):
        FP = 0
        TP = 0
        FN = 0
        TN = 0

        for client_id in removed_clients:
            if client_id in self.index_malicious:
                TP += 1
            else:
                FP += 1

        for client_id, _cluster in cluster_tuples:
            if client_id not in removed_clients:
                if client_id in self.index_malicious:
                    FN += 1
                else:
                    TN += 1

        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        FRR = FN / (FN + TP) if (FN + TP) > 0 else 0
        return FPR, FRR

    def _quick_val_accuracy(self, state_dict):
        """Evaluate state_dict on public validation set. Returns accuracy 0-1."""
        try:
            public_val_dir = os.environ.get('PUBLIC_VAL_DIR', '')
            if not public_val_dir or not os.path.isdir(public_val_dir):
                return None
            device = next(iter(state_dict.values())).device if state_dict else torch.device('cpu')
            result = _eval_state_dict(state_dict, public_val_dir, device)
            return float(result['acc'])
        except Exception as e:
            print(f'[ROLLBACK] val accuracy check failed: {e}')
            return None

    def train(self):
        _best_val_acc = 0.0
        _rollback_count = 0
        _consecutive_rollbacks = 0
        for i in range(self.global_rounds+1):
            self.current_round = i
            s_t = time.time()
            global_state_before_round = {
                k: v.detach().clone()
                for k, v in self.global_model.state_dict().items()
            }
            quarantined_at_round_start = (
                set() if getattr(self, 'disable_quarantine', False)
                else {
                    client_id for client_id, status in self.client_quarantine_dict.items()
                    if status['roundsQuarent'] > 0
                }
            )
            self.selected_clients = self.select_clients()
            self.send_models()
            self.removed_clients = []
            self.cluster_tuples = ()
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
            round_detail_rows = []
            #for client in self.selected_clients:
            #    client.train()
            for client in self.selected_clients:
                client.current_round = self.current_round

            threads = [Thread(target=client.train)
                       for client in self.selected_clients]
            [t.start() for t in threads]
            [t.join() for t in threads]

            self.receive_models()

            # ALIE Attack Synthesis & Verification
            # ALIE requires inspecting honest client updates from the current round
            clients_by_id = {c.id: c for c in self.clients}
            alie_indices = [
                idx for idx, cid in enumerate(self.uploaded_ids)
                if getattr(clients_by_id.get(cid), 'pending_attack_type', 'benign') == 'malicious_alie'
            ]
            benign_models = [
                self.uploaded_models[idx] for idx, cid in enumerate(self.uploaded_ids)
                if getattr(clients_by_id.get(cid), 'pending_attack_type', 'benign') == 'benign'
            ]
            if alie_indices and benign_models:
                print(f"[ALIE Attack] Crafting ALIE perturbation using {len(benign_models)} benign updates for {len(alie_indices)} ALIE client(s)...")
                n_malicious = len(self.index_malicious) if hasattr(self, 'index_malicious') else len(alie_indices)
                alie_model_template = model_alie(benign_models, self.num_clients, n_malicious)
                
                # Replace uploaded model for ALIE clients
                for idx in alie_indices:
                    self.uploaded_models[idx] = copy.deepcopy(alie_model_template)

                # Verification check on crafted ALIE model
                for idx in alie_indices:
                    is_valid, stats = verify_alie_attack(benign_models, self.uploaded_models[idx], self.num_clients, n_malicious)
                    if is_valid:
                        print(f"[ALIE Verification PASSED] Client {self.uploaded_ids[idx]}: max_diff={stats['max_diff']:.6e}, NaN count={stats['nan_count']}")
                    else:
                        print(f"[ALIE Verification FAILED] Client {self.uploaded_ids[idx]}: max_diff={stats['max_diff']:.6e}, NaN count={stats['nan_count']}")

            # Dump state_dicts pra geracao de dataset (modo --dump_state_dicts)
            if self.dump_dir and i >= self.dump_start_round:
                clients_by_id = {c.id: c for c in self.clients}
                n_saved = fl_save.save_round_dump(
                    self.uploaded_models, self.uploaded_ids,
                    clients_by_id, self.index_malicious,
                    round_idx=i, out_dir=self.dump_dir,
                    global_state_dict=global_state_before_round,
                )
                print(f'[dump] round {i}: salvos {n_saved} state_dicts em {self.dump_dir}')
            elif self.dump_dir:
                print(f'[dump] round {i}: ignorado antes de dump_start_round={self.dump_start_round}')

            if i > 0 and self.uploaded_models:
                if self.current_round <= getattr(self, 'round_init_atk', 0):
                    print(f"Warmup round {self.current_round} <= {getattr(self, 'round_init_atk', 0)}; skipping detector.")
                elif self.cc==2:
                    oi = time.time()
                    round_upload_ids = list(self.ids)
                    if len(self.uploaded_models) < 2:
                        print("cc=2: menos de 2 uploads validos; pulando clustering neste round.")
                    else:
                        similarity_matrix, _ = self.calculate_similarity_scores()
                        num_clusters = min(2, len(self.uploaded_models))
                        clusters = self.perform_clustering(similarity_matrix, num_clusters)

                        self.cluster_tuples = [(self.ids[idx], cluster) for idx, cluster in enumerate(clusters)]
                        for idx, cluster in enumerate(clusters):
                            print(f"Client {self.ids[idx]} is in cluster {cluster}")
                        cluster_counts = Counter([cluster for _, cluster in self.cluster_tuples])
                        min_cluster = min(cluster_counts, key=cluster_counts.get)

                        for idx in range(len(self.cluster_tuples) - 1, -1, -1):
                            client_id, cluster = self.cluster_tuples[idx]
                            if cluster == min_cluster:
                                print(f"Removing client {client_id} from cluster {cluster}")
                                self.removed_clients.append(client_id)
                                del self.uploaded_models[idx]
                                del self.ids[idx]
                                del self.uploaded_ids[idx]
                                del self.uploaded_weights[idx]
                        if self.uploaded_weights:
                            s = sum(self.uploaded_weights)
                            if s > 0:
                                self.uploaded_weights = [weight / s for weight in self.uploaded_weights]
                    detail_rows = self._build_cc_detail_rows(
                        i, round_upload_ids, self.removed_clients, reason='cluster_minority'
                    )
                    round_detail_rows = detail_rows
                    self.save_cc_detail_to_csv(detail_rows)
                    self.save_cc_type_to_csv(i, detail_rows)
                    print(f"Tempo de execução: {time.time()-oi:.4f} segundos")

                elif self.cc==3:
                    oi = time.time()
                    round_upload_ids = list(self.ids)
                    round_removed_clients = []
                    client_scores = {}
                    if len(self.uploaded_models) < 2:
                        print("cc=3: menos de 2 uploads validos; pulando score neste round.")
                    else:
                        _similarity_matrix, client_scores  = self.calculate_similarity_scores()
                        scores_array = np.array(list(client_scores.values()))
                        mean_score = np.mean(scores_array)
                        std_score = np.std(scores_array)
                        print(f"Average score: {mean_score:.4f}")
                        mean_score = mean_score - std_score
                        print(f"Average score: {mean_score:.4f}")
                        client_tuples = [(self.ids[idx], client_scores[self.ids[idx]]) for idx in range(len(self.ids))]
                        total = len(self.index_malicious)
                        found = 0
                        for idx in range(len(client_tuples) - 1, -1, -1):
                            client_id, score = client_tuples[idx]
                            print(f"Esse  {client_id} with score {score:.4f} ")
                            if score < mean_score:
                                if client_id in self.index_malicious:
                                    found += 1
                                print(f"Removing client {client_id} with score {score:.4f} (below average)")
                                round_removed_clients.append(client_id)
                                self.set_client_quarantine(client_id)
                                del self.uploaded_models[idx]
                                del self.ids[idx]
                                del self.uploaded_ids[idx]
                                del self.uploaded_weights[idx]
                        found_pct = (found/total) * 100 if total > 0 else 0.0
                        print("porcentagem de clientes maliciosos de verdade achados: "+ str(found_pct) + "%")
                        if self.uploaded_weights:
                            s = sum(self.uploaded_weights)
                            if s > 0:
                                self.uploaded_weights = [weight / s for weight in self.uploaded_weights]
                    detail_rows = self._build_cc_detail_rows(
                        i, round_upload_ids, round_removed_clients,
                        reason='score_below_mean_minus_std',
                        scores=client_scores,
                    )
                    round_detail_rows = detail_rows
                    self.save_cc_detail_to_csv(detail_rows)
                    self.save_cc_type_to_csv(i, detail_rows)
                    print(f"Tempo de execução: {time.time()-oi:.4f} segundos")

                elif self.cc==5:
                    print("vai rolar nada")
                elif self.cc == 7:
                    oi = time.time()
                    clients_by_id = {c.id: c for c in self.clients}
                    true_positive_uploads = 0
                    malicious_uploads = 0
                    detail_rows = []
                    uploaded_sds = [m.state_dict() for m in self.uploaded_models]
                    mlp_results = self.client_check.classify_batch(
                        uploaded_sds, global_state_dict=global_state_before_round
                    )
                    for idx in range(len(self.uploaded_models) - 1, -1, -1):
                        client_id = self.ids[idx]
                        client = clients_by_id.get(client_id)
                        attack_type = getattr(client, 'last_attack_type', 'unknown')
                        is_malicious_round = bool(getattr(client, 'is_malicious', False))
                        if is_malicious_round:
                            malicious_uploads += 1
                        mlp_result = mlp_results[idx]
                        mlp_hit = bool(mlp_result['is_malicious'])
                        removed = mlp_hit
                        reason = 'mlp' if mlp_hit else 'none'
                        detail_rows.append({
                            'round': i,
                            'cc': self.cc,
                            'client_id': client_id,
                            'attack_type': attack_type,
                            'is_malicious_round': is_malicious_round,
                            'malicious_group': client_id in self.index_malicious,
                            'removed': removed,
                            'reason': reason if removed else 'none',
                            'mlp_hit': mlp_hit,
                            'mlp_score': float(mlp_result.get('score', 0.0)),
                            'mlp_label_score': float(mlp_result.get('label_score', 0.0)),
                            'mlp_binary_hit': bool(mlp_result.get('binary_hit', False)),
                            'mlp_label_hit': bool(mlp_result.get('label_hit', False)),
                            'binary_threshold': mlp_result.get('binary_threshold'),
                            'label_threshold': mlp_result.get('label_threshold'),
                            'decision_rule': mlp_result.get('decision_rule', 'binary'),
                        })
                        if removed:
                            if is_malicious_round:
                                true_positive_uploads += 1
                            print(f'cc={self.cc}: removing client {client_id} (MLP detector)')
                            self.set_client_quarantine(client_id)
                            del self.uploaded_models[idx]
                            del self.ids[idx]
                            del self.uploaded_ids[idx]
                            del self.uploaded_weights[idx]
                    self.save_cc_detail_to_csv(detail_rows)
                    self.save_cc_type_to_csv(i, detail_rows)
                    round_detail_rows = detail_rows
                    round_recall = true_positive_uploads / malicious_uploads if malicious_uploads > 0 else 0.0
                    print(
                        f'recall de uploads maliciosos no round (cc={self.cc}): '
                        f'{round_recall:.2%} ({true_positive_uploads}/{malicious_uploads})'
                    )
                    if self.uploaded_weights:
                        s = sum(self.uploaded_weights)
                        if s > 0:
                            self.uploaded_weights = [w / s for w in self.uploaded_weights]
                    print(f'Tempo de execução cc={self.cc}: {time.time()-oi:.4f}s')

                elif self.cc == 8:
                    oi = time.time()
                    clients_by_id = {c.id: c for c in self.clients}
                    true_positive_uploads = 0
                    malicious_uploads = 0
                    detail_rows = []
                    
                    # 1. Element-wise Sign Extraction
                    sign_vectors = []
                    for model in self.uploaded_models:
                        sign_vec = []
                        for k, v in model.state_dict().items():
                            if v.is_floating_point():
                                global_v = global_state_before_round[k].to(v.device)
                                delta = v - global_v
                                sign = torch.where(torch.abs(delta) > 1e-8, torch.sign(delta), torch.zeros_like(delta))
                                sign_vec.append(sign.view(-1))
                        sign_vectors.append(torch.cat(sign_vec))
                    sign_vectors = torch.stack(sign_vectors) # [num_clients, total_params]
                    
                    # 2. Compute similarity matrix A_{i,j} = (1 + cos(S_i, S_j)) / 2
                    norms = torch.norm(sign_vectors, dim=1, keepdim=True)
                    normalized_signs = sign_vectors / (norms + 1e-10)
                    sim_matrix = torch.matmul(normalized_signs, normalized_signs.T)
                    sim_matrix = (1 + sim_matrix) / 2
                    
                    # 3. Determine threshold O_phi
                    num_clients = len(self.uploaded_models)
                    m = len(self.index_malicious) if hasattr(self, 'index_malicious') else 0
                    if m > 1 and num_clients > 1:
                        phi = int(m * (m - 1) / 2)
                        sim_matrix_np = sim_matrix.cpu().numpy()
                        off_diag_idx = np.triu_indices(num_clients, k=1)
                        similarities = sim_matrix_np[off_diag_idx]
                        similarities = np.sort(similarities)[::-1] # descending
                        
                        if phi > 0 and phi <= len(similarities):
                            o_phi = similarities[phi - 1]
                        else:
                            o_phi = similarities[0] if len(similarities) > 0 else 0
                        
                        # 4. Calculate attack density P_i
                        p_i = np.zeros(num_clients)
                        for idx_i in range(num_clients):
                            for idx_j in range(num_clients):
                                if idx_i != idx_j and sim_matrix_np[idx_i, idx_j] > o_phi:
                                    p_i[idx_i] += 1
                                    
                        p_m = np.mean(p_i)
                        
                        for idx in range(num_clients - 1, -1, -1):
                            client_id = self.ids[idx]
                            client = clients_by_id.get(client_id)
                            attack_type = getattr(client, 'last_attack_type', 'unknown')
                            is_malicious_round = bool(getattr(client, 'is_malicious', False))
                            if is_malicious_round:
                                malicious_uploads += 1
                                
                            fedsign_hit = bool(p_i[idx] >= p_m) and p_m > 0
                            removed = fedsign_hit
                            reason = 'fedsign_pad' if fedsign_hit else 'none'
                            
                            detail_rows.append({
                                'round': i,
                                'cc': self.cc,
                                'client_id': client_id,
                                'attack_type': attack_type,
                                'is_malicious_round': is_malicious_round,
                                'malicious_group': client_id in self.index_malicious,
                                'removed': removed,
                                'reason': reason,
                                'mlp_hit': fedsign_hit,
                                'mlp_score': float(p_i[idx]),
                                'mlp_label_score': float(p_m),
                                'mlp_binary_hit': False,
                                'mlp_label_hit': False,
                                'binary_threshold': float(o_phi),
                                'label_threshold': 0.0,
                                'decision_rule': 'fedsign',
                            })
                            
                            if removed:
                                if is_malicious_round:
                                    true_positive_uploads += 1
                                print(f'cc={self.cc}: removing client {client_id} (FedSIGN PAD)')
                                self.set_client_quarantine(client_id)
                                del self.uploaded_models[idx]
                                del self.ids[idx]
                                del self.uploaded_ids[idx]
                                del self.uploaded_weights[idx]
                    else:
                        print("FedSIGN requires >1 clients and >1 malicious clients to set phi threshold.")
                        
                    self.save_cc_detail_to_csv(detail_rows)
                    self.save_cc_type_to_csv(i, detail_rows)
                    round_detail_rows = detail_rows
                    round_recall = true_positive_uploads / malicious_uploads if malicious_uploads > 0 else 0.0
                    print(
                        f'recall de uploads maliciosos no round (cc={self.cc}): '
                        f'{round_recall:.2%} ({true_positive_uploads}/{malicious_uploads})'
                    )
                    if self.uploaded_weights:
                        s = sum(self.uploaded_weights)
                        if s > 0:
                            self.uploaded_weights = [w / s for w in self.uploaded_weights]
                    print(f'Tempo de execução cc={self.cc}: {time.time()-oi:.4f}s')
            print(self.client_quarantine_dict)
            # Quarantine-occupancy snapshot (diagnostic, NOT the paper FPR/FRR).
            quarantine_fpr = 0
            quarantine_frr = 0
            if self.cc ==2:
                quarantine_fpr, quarantine_frr = self.compute_fpr_frr_cluster(self.removed_clients, self.cluster_tuples)
            if self.cc in (3, 7, 8):
                quarantine_fpr, quarantine_frr = self.compute_fpr_frr()
            # Per-round detection rate (paper Eq 14/15, headline metric).
            detection_fpr, detection_frr = self.compute_upload_fpr_frr(round_detail_rows)
            if detection_fpr != '':
                print(f"Round {i}: DetectionFPR = {detection_fpr:.4f}, DetectionFRR = {detection_frr:.4f}")
            print(f"Round {i}: QuarantineFPR = {quarantine_fpr:.4f}, QuarantineFRR = {quarantine_frr:.4f}")
            self.save_fpr_frr_to_csv(i, detection_fpr, detection_frr, quarantine_fpr, quarantine_frr)
            for client_id in quarantined_at_round_start:
                self.decrease_quarentine(client_id)
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            # ---- Rollback safety net (cc=7 only) ----
            if self.cc == 7 and i > 0:
                new_acc = self._quick_val_accuracy(
                    self.global_model.state_dict()
                )
                if new_acc is not None:
                    if new_acc > _best_val_acc:
                        _best_val_acc = new_acc
                        _consecutive_rollbacks = 0
                    drop = _best_val_acc - new_acc
                    if drop > _ROLLBACK_DROP_PP and _consecutive_rollbacks < 3:
                        print(
                            f'[ROLLBACK] Round {i}: val accuracy {new_acc:.4f} '
                            f'dropped {drop:.4f} from best {_best_val_acc:.4f}. '
                            f'Restoring pre-round global model (consecutive={_consecutive_rollbacks+1}).'
                        )
                        self.global_model.load_state_dict(global_state_before_round)
                        _rollback_count += 1
                        _consecutive_rollbacks += 1
                    else:
                        if drop > _ROLLBACK_DROP_PP:
                            print(
                                f'[ROLLBACK OVERRIDE] Round {i}: max consecutive rollbacks reached ({_consecutive_rollbacks}). '
                                f'Updating best val acc to {new_acc:.4f} and continuing.'
                            )
                            _best_val_acc = new_acc
                        _consecutive_rollbacks = 0
                        print(
                            f'[VAL] Round {i}: val accuracy {new_acc:.4f} '
                            f'(best={_best_val_acc:.4f}, drop={drop:.4f})'
                        )

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
