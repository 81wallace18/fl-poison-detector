import time
import uuid
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import numpy as np
from collections import Counter
import torch
import csv
import os
from torch.utils.data import DataLoader
from flcore.detector import fl_save
from flcore.detector.cc import ClientCheck
from flcore.detector.cc_mlp import ClientCheckMLP
from flcore.detector.label_flip_check import LabelFlipCheck
from flcore.detector.targeted_label_flip_check import TargetedLabelFlipCheck
from flcore.detector.validation_check import PublicValidationCheck
from utils.data_utils import read_client_data
class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.fpr_frr_results = []
        self.run_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"

        # Open the CSV file in append mode to save results over time
        if self.cc ==3:
            self.csv_filename = 'fpr_frr_results_3.csv'
        elif self.cc ==2:
            self.csv_filename = 'fpr_frr_results_2.csv'
        elif self.cc ==6:
            self.csv_filename = 'fpr_frr_results_6.csv'
        elif self.cc ==7:
            self.csv_filename = 'fpr_frr_results_7.csv'
        elif self.cc ==8:
            self.csv_filename = 'fpr_frr_results_8.csv'
        elif self.cc ==9:
            self.csv_filename = 'fpr_frr_results_9.csv'
        elif self.cc ==10:
            self.csv_filename = 'fpr_frr_results_10.csv'
        else:
            self.csv_filename = 'f.csv'
        self._ensure_csv_header(self.csv_filename, ['RunID', 'Round', 'FPR', 'FRR'])
        self.cc_detail_filename = f'cc_detail_results_{self.cc}.csv'
        self.cc_type_filename = f'cc_type_results_{self.cc}.csv'
        if self.cc in (6, 7, 8, 9, 10):
            detail_header = [
                'RunID', 'Round', 'CC', 'ClientID', 'AttackType', 'IsMaliciousRound',
                'MaliciousGroup', 'Removed', 'Reason', 'MLPHit', 'BERTHit',
                'ValHit', 'LFHit', 'BehaviorHit', 'TargetLFHit', 'MLPScore', 'BERTScore',
                'ValScore', 'LFCos', 'LFWorstClassDelta',
                'BehaviorMarginDelta', 'BehaviorLossDelta',
                'TargetLFScore', 'TargetLFSuspectClass', 'TargetLFTargetClass',
                'TargetLFMarginDelta', 'TargetLFLossDelta',
                'TargetLFTargetProbDelta', 'TargetLFHeadScore', 'TargetLFFinalDelta',
                'TargetLFScoreOutlier', 'TargetLFBehaviorAllowed', 'TargetLFCapped',
            ]
            self._ensure_csv_header(self.cc_detail_filename, detail_header)
            self._ensure_csv_header(
                self.cc_type_filename,
                ['RunID', 'Round', 'CC', 'AttackType', 'Total', 'Removed', 'Rate', 'Metric'],
            )

        self.dump_dir = getattr(args, 'dump_state_dicts', '') or ''
        self.client_check = None
        self.bert_client_check = None
        self.mlp_client_check = None
        self.cc9_lf_standalone = bool(getattr(args, 'cc9_lf_standalone', False))
        if self.cc == 6:
            detector_dir = getattr(args, 'detector_dir', '') or ''
            if not detector_dir:
                raise ValueError("cc=6 requer --detector_dir apontando pro modelo treinado (ex: jpt/detector_final/).")
            print(f"[cc=6] Carregando detector DistilBERT de {detector_dir}")
            self.client_check = ClientCheck(detector_dir)
        elif self.cc == 7:
            detector_dir = getattr(args, 'detector_dir', '') or ''
            if not detector_dir:
                raise ValueError("cc=7 requer --detector_dir apontando pro MLP artifacts dir (ex: jpt/detector_mlp_monza_cnn_mnist/).")
            print(f"[cc=7] Carregando detector MLP de {detector_dir}")
            self.client_check = ClientCheckMLP(detector_dir)
        elif self.cc == 8:
            detector_dir = getattr(args, 'detector_dir', '') or ''
            if not detector_dir:
                raise ValueError("cc=8 requer --detector_dir apontando pro MLP artifacts dir (ex: jpt/detector_mlp_monza_cnn_mnist/).")
            print(f"[cc=8] Carregando detector MLP+validacao publica de {detector_dir}")
            self.client_check = ClientCheckMLP(detector_dir)
            self.public_val_check = None
        elif self.cc == 9:
            mlp_detector_dir = getattr(args, 'mlp_detector_dir', '') or getattr(args, 'detector_dir', '') or ''
            bert_detector_dir = getattr(args, 'bert_detector_dir', '') or ''
            if not mlp_detector_dir:
                raise ValueError("cc=9 requer --mlp_detector_dir ou --detector_dir apontando pro MLP artifacts dir.")
            if not bert_detector_dir:
                raise ValueError("cc=9 requer --bert_detector_dir apontando pro modelo DistilBERT treinado.")
            print(f"[cc=9] Carregando detector MLP de {mlp_detector_dir}")
            print(f"[cc=9] Carregando detector DistilBERT de {bert_detector_dir}")
            self.mlp_client_check = ClientCheckMLP(mlp_detector_dir)
            self.bert_client_check = ClientCheck(bert_detector_dir)
            self.label_flip_check = None
        elif self.cc == 10:
            mlp_detector_dir = getattr(args, 'mlp_detector_dir', '') or getattr(args, 'detector_dir', '') or ''
            bert_detector_dir = getattr(args, 'bert_detector_dir', '') or ''
            if not mlp_detector_dir:
                raise ValueError("cc=10 requer --mlp_detector_dir ou --detector_dir apontando pro MLP artifacts dir.")
            if not bert_detector_dir:
                raise ValueError("cc=10 requer --bert_detector_dir apontando pro modelo DistilBERT treinado.")
            print(f"[cc=10] Carregando detector MLP de {mlp_detector_dir}")
            print(f"[cc=10] Carregando detector DistilBERT de {bert_detector_dir}")
            self.mlp_client_check = ClientCheckMLP(mlp_detector_dir)
            self.bert_client_check = ClientCheck(bert_detector_dir)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)
        if self.cc == 8:
            self.public_val_check = PublicValidationCheck(
                self._build_public_validation_loader(),
                self.device,
                min_delta=getattr(args, 'val_check_min_delta', 0.02),
                mad_k=getattr(args, 'val_check_mad_k', 3.0),
            )
        if self.cc == 9:
            self.label_flip_check = LabelFlipCheck(
                self._build_public_validation_loader(label='cc=9'),
                self.device,
                root_lr=getattr(args, 'lf_check_root_lr', 0.01),
                root_steps=getattr(args, 'lf_check_root_steps', 5),
                min_loss_delta=getattr(args, 'lf_check_min_loss_delta', 0.02),
                mad_k=getattr(args, 'lf_check_mad_k', 3.0),
                max_final_cos=getattr(args, 'lf_check_max_final_cos', 0.0),
            )
        if self.cc == 10:
            self.targeted_label_flip_check = TargetedLabelFlipCheck(
                self._build_public_validation_loader(label='cc=10'),
                self.device,
                num_classes=getattr(args, 'num_classes', 10),
                min_score=getattr(args, 'target_lf_min_score', 2.0),
                min_margin_delta=getattr(args, 'target_lf_min_margin_delta', 0.05),
                min_loss_delta=getattr(args, 'target_lf_min_loss_delta', -0.10),
                mad_k=getattr(args, 'target_lf_mad_k', 2.5),
                max_reject_fraction=getattr(args, 'target_lf_max_reject_fraction', 0.30),
                head_weight=getattr(args, 'target_lf_head_weight', 0.35),
                margin_weight=getattr(args, 'target_lf_margin_weight', 1.0),
                loss_weight=getattr(args, 'target_lf_loss_weight', 0.50),
                target_prob_weight=getattr(args, 'target_lf_target_prob_weight', 0.50),
            )

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

    def _build_public_validation_loader(self, label='cc=8'):
        target = int(getattr(self.args, 'val_check_samples', 256))
        batch_size = int(getattr(self.args, 'val_check_batch_size', 128))
        buckets = {c: [] for c in range(self.num_classes)}
        all_samples = []
        take = max(1, (target + max(self.num_clients, 1) - 1) // max(self.num_clients, 1))
        per_class = max(1, (target + max(self.num_classes, 1) - 1) // max(self.num_classes, 1))
        for cid in range(self.num_clients):
            client_data = read_client_data(self.dataset, cid, is_train=False, few_shot=self.few_shot)
            if not client_data:
                continue
            for sample in client_data[:max(take, per_class)]:
                all_samples.append(sample)
                y = sample[1]
                cls = int(y.item() if hasattr(y, 'item') else y)
                if cls in buckets and len(buckets[cls]) < per_class:
                    buckets[cls].append(sample)
            if sum(len(v) for v in buckets.values()) >= target:
                break
        samples = []
        seen = set()
        while len(samples) < target:
            added = False
            for cls in range(self.num_classes):
                if buckets[cls]:
                    sample = buckets[cls].pop(0)
                    samples.append(sample)
                    seen.add(id(sample))
                    added = True
                    if len(samples) >= target:
                        break
            if not added:
                break
        for sample in all_samples:
            if len(samples) >= target:
                break
            if id(sample) not in seen:
                samples.append(sample)
        if not samples:
            raise ValueError(f"{label} requer dados de teste para montar holdout publico.")
        samples = samples[:target]
        print(f"[{label}] Holdout publico: {len(samples)} amostras | batch={batch_size}")
        return DataLoader(samples, batch_size=batch_size, drop_last=False, shuffle=False)
        
    def save_fpr_frr_to_csv(self, round_number, FPR, FRR):
        """
        Saves the FPR and FRR results to a CSV file for each round.
        """
        with open(self.csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.run_id, round_number, FPR, FRR])

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
                    int(row['bert_hit']), int(row['val_hit']), int(row['lf_hit']),
                    int(row['behavior_hit']), int(row['target_lf_hit']), row['mlp_score'], row['bert_score'],
                    row['val_score'], row['lf_cos'], row['lf_worst_class_delta'],
                    row['behavior_margin_delta'], row['behavior_loss_delta'],
                    row['target_lf_score'], row['target_lf_suspect_class'],
                    row['target_lf_target_class'], row['target_lf_margin_delta'],
                    row['target_lf_loss_delta'], row['target_lf_target_prob_delta'],
                    row['target_lf_head_score'], row['target_lf_final_delta'],
                    int(row['target_lf_score_outlier']), int(row['target_lf_behavior_allowed']),
                    int(row['target_lf_capped']),
                ])

    def _final_layer_delta(self, client_model):
        global_sd = self.global_model.state_dict()
        client_sd = client_model.state_dict()
        final_key = None
        for key in ('head.weight', 'fc.weight', 'base.fc.weight'):
            if key in global_sd and key in client_sd:
                final_key = key
                break
        if final_key is None:
            candidates = [
                key for key in global_sd
                if key in client_sd and (key.endswith('fc.weight') or key.endswith('head.weight'))
            ]
            if not candidates:
                return 0.0
            final_key = candidates[-1]
        delta = torch.linalg.norm((client_sd[final_key].detach() - global_sd[final_key].detach()).float())
        bias_key = final_key[:-len('weight')] + 'bias'
        if bias_key in global_sd and bias_key in client_sd:
            delta = delta + torch.linalg.norm((client_sd[bias_key].detach() - global_sd[bias_key].detach()).float())
        return float(delta.item())

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

    def normalize_entropies(self, client_entropies):
        """Normaliza as entropias para que fiquem no intervalo [0, 1]"""
        # Obter as entropias
        entropies = np.array(list(client_entropies.values()))

        # Calcular o valor mínimo e máximo
        min_entropy = np.min(entropies)
        max_entropy = np.max(entropies)

        # Normalizar as entropias
        normalized_entropies = (entropies - min_entropy) / (max_entropy - min_entropy)

        # Atualizar o dicionário com as entropias normalizadas
        normalized_client_entropies = {client_id: normalized_entropy for client_id, normalized_entropy in zip(client_entropies.keys(), normalized_entropies)}

        # Exibir as entropias normalizadas
        for client_id, normalized_entropy in normalized_client_entropies.items():
            print(f"Normalized Shannon entropy for client {client_id}: {normalized_entropy:.4f}")

        return normalized_client_entropies
    def set_client_quarantine(self, client_id):
        self.client_quarantine_dict[client_id]['quarentena'] = self.client_quarantine_dict[client_id]['quarentena'] +1
        self.client_quarantine_dict[client_id]['roundsQuarent'] = self.client_quarantine_dict[client_id]['quarentena'] *2

    def decrease_quarentine(self, client_id):
        if self.client_quarantine_dict[client_id]['roundsQuarent'] ==0:
            self.client_quarantine_dict[client_id]['roundsQuarent'] = 0
        else:
            self.client_quarantine_dict[client_id]['roundsQuarent'] = self.client_quarantine_dict[client_id]['roundsQuarent'] -1
    def compute_fpr_frr(self):
        """
        Calcula False Positive Rate (FPR) e False Rejection Rate (FRR)
        usando self.client_quarantine_dict e self.index_malicious.
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
        """
        Calcula FPR e FRR com base nos clientes removidos do cluster.
        """
        FP = 0  # Falsos positivos: clientes não maliciosos removidos
        TP = 0  # Verdadeiros positivos: maliciosos removidos
        FN = 0  # Falsos negativos: maliciosos não removidos
        TN = 0  # Verdadeiros negativos: não maliciosos não removidos

        # Comparar os clientes removidos com a lista de maliciosos
        for client_id in removed_clients:
            is_malicious = client_id in self.index_malicious  # Verificar se é malicioso
            if is_malicious:
                TP += 1  # Cliente malicioso corretamente removido
            else:
                FP += 1  # Cliente não malicioso removido erroneamente

        # Verificar os clientes que não foram removidos (ainda estão no cluster)
        for client_id, cluster in cluster_tuples:
            if client_id not in removed_clients:
                is_malicious = client_id in self.index_malicious
                if is_malicious:
                    FN += 1  # Cliente malicioso não removido
                else:
                    TN += 1  # Cliente não malicioso não removido

        # Calcular FPR e FRR
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        FRR = FN / (FN + TP) if (FN + TP) > 0 else 0

        return FPR, FRR

    def train(self):
        
        for i in range(self.global_rounds+1):
            s_t = time.time()
            quarantined_at_round_start = {
                client_id for client_id, status in self.client_quarantine_dict.items()
                if status['roundsQuarent'] > 0
            }
            self.selected_clients = self.select_clients()
            self.send_models()
            self.removed_clients = []
            self.cluster_tuples = ()
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
            #for client in self.selected_clients:
            #    client.train()
            for client in self.selected_clients:
                client.current_round = self.current_round

            threads = [Thread(target=client.train)
                       for client in self.selected_clients]
            [t.start() for t in threads]
            [t.join() for t in threads]

            self.receive_models()

            # Dump state_dicts pra geracao de dataset (modo --dump_state_dicts)
            if self.dump_dir:
                clients_by_id = {c.id: c for c in self.clients}
                n_saved = fl_save.save_round_dump(
                    self.uploaded_models, self.uploaded_ids,
                    clients_by_id, self.index_malicious,
                    round_idx=i, out_dir=self.dump_dir,
                )
                print(f'[dump] round {i}: salvos {n_saved} state_dicts em {self.dump_dir}')

            if i > 0 and self.uploaded_models:
                #comparar com o modelo
                if self.cc==0:
                    global_model_params = list(self.global_model.parameters()) 
                # Calcular a similaridade de cosseno entre os modelos dos clientes e o modelo global
                    similarities = self.calculate_similarity_with_global_model(global_model_params)
                    for sim in similarities:
                        print(f"Cosine similarity between client {sim[0]} and the global model: {sim[1]:.4f}")
                #comparar com todos os modelos, esse não funciona no momento
                if self.cc==1:
                    similarity_scores = self.calculate_similarity_scores()
                    for client_id, score in similarity_scores.items():
                        print(f"Cosine similarity for client {client_id}: {score:.4f}")
                    normalized_client_entropies = self.normalize_entropies(similarity_scores)
                #comparar com todos os modelos e fazer cluster
                if self.cc==2:
                    oi = time.time()
                    if len(self.uploaded_models) < 2:
                        print("cc=2: menos de 2 uploads validos; pulando clustering neste round.")
                    else:
                        similarity_matrix, a = self.calculate_similarity_scores()

                        # Realizar a clusterização
                        num_clusters = min(2, len(self.uploaded_models))  # Defina o número de clusters conforme necessário
                        clusters = self.perform_clustering(similarity_matrix, num_clusters)
                        #for idx, cluster in enumerate(clusters):
                            #print(f"Client {self.ids[idx]} is in cluster {cluster}")

                        self.cluster_tuples = [(self.ids[idx], cluster) for idx, cluster in enumerate(clusters)]
                        for idx, cluster in enumerate(clusters):
                            print(f"Client {self.ids[idx]} is in cluster {cluster}")
                        cluster_counts = Counter([cluster for _, cluster in self.cluster_tuples])
                        min_cluster = min(cluster_counts, key=cluster_counts.get)

                        for idx in range(len(self.cluster_tuples) - 1, -1, -1):
                            client_id, cluster = self.cluster_tuples[idx]
                            #print(self.ids)
                            if cluster == min_cluster:
                                print(f"Removing client {client_id} from cluster {cluster}")
                                self.removed_clients.append(client_id)
                                # Remover o cliente das listas associadas
                                del self.uploaded_models[idx]
                                del self.ids[idx]
                                del self.uploaded_ids[idx]
                                del self.uploaded_weights[idx]
                                #print(self.ids)
                        if self.uploaded_weights:
                            s = sum(self.uploaded_weights)
                            if s > 0:
                                self.uploaded_weights = [weight / s for weight in self.uploaded_weights]
                    bye = time.time()
                    vish = bye- oi  # Calcula o tempo decorrido
                    print(f"Tempo de execução: {vish:.4f} segundos")
                #metodo do cosseno mas com score
                if self.cc==3:
                    oi = time.time()
                    if len(self.uploaded_models) < 2:
                        print("cc=3: menos de 2 uploads validos; pulando score neste round.")
                    else:
                        similarity_matrix, client_scores  = self.calculate_similarity_scores()
                        # Converte os scores para array e calcula a média
                        scores_array = np.array(list(client_scores.values()))
                        mean_score = np.mean(scores_array)
                        std_score = np.std(scores_array)
                        print(f"Average score: {mean_score:.4f}")
                        mean_score = mean_score - std_score
                        print(f"Average score: {mean_score:.4f}")
                        # Cria uma lista de tuplas para manter a posição dos clientes
                        client_tuples = [(self.ids[idx], client_scores[self.ids[idx]]) for idx in range(len(self.ids))]
                        total = len(self.index_malicious)
                        a = 0
                        # Itera de trás para frente para remover clientes abaixo da média
                        if std_score<0.001:
                            print("nenhum malicioso")
                        else:
                            for idx in range(len(client_tuples) - 1, -1, -1):
                                client_id, score = client_tuples[idx]
                                print(f"Esse  {client_id} with score {score:.4f} ")
                                if score < mean_score:
                                    if client_id in self.index_malicious:
                                        a = a+1
                                    print(f"Removing client {client_id} with score {score:.4f} (below average)")
                                    self.set_client_quarantine(client_id)
                                    # Remover o cliente das listas associadas
                                    del self.uploaded_models[idx]
                                    del self.ids[idx]
                                    del self.uploaded_ids[idx]
                                    del self.uploaded_weights[idx]
                        a = (a/total) *100 if total > 0 else 0.0
                        print("porcentagem de clientes maliciosos de verdade achados: "+ str(a) + "%")
                        if self.uploaded_weights:
                            s = sum(self.uploaded_weights)
                            if s > 0:
                                self.uploaded_weights = [weight / s for weight in self.uploaded_weights]
                    bye = time.time()
                    vish = bye - oi  # Calcula o tempo decorrido
                    print(f"Tempo de execução: {vish:.4f} segundos")
                
                if self.cc ==4:
                    oi = time.time()
                    k = 3
                    client_entropies = self.calculate_client_entropies()
                    entropies = np.array(list(client_entropies.values()))
                    mean_entropy = np.mean(entropies)
                    std_entropy = np.std(entropies)
                    lower_bound = mean_entropy - std_entropy
                    upper_bound = mean_entropy + std_entropy-(std_entropy/2)
                    
                    print(f"Mean entropy: {mean_entropy:.4f}, Std: {std_entropy:.4f}")
                    print(f"Keeping clients with entropy in [{lower_bound:.4f}, {upper_bound:.4f}]")
                    
                    # 3. Lista de tuplas para manter índice
                    client_tuples = [(self.ids[idx], client_entropies[self.ids[idx]]) for idx in range(len(self.ids))]

                    # 4. Remover outliers (de trás para frente)
                    for idx in range(len(client_tuples) - 1, -1, -1):
                        client_id, entropy = client_tuples[idx]
                        if entropy < lower_bound or entropy > upper_bound:
                            print(f"Removing client {client_id} with entropy {entropy:.4f} (outlier)")

                            # Remover das listas associadas
                            del self.uploaded_models[idx]
                            del self.ids[idx]
                            del self.uploaded_ids[idx]
                            del self.uploaded_weights[idx]
                    #normalized_client_entropies = self.normalize_entropies(client_entropies)
                    if self.uploaded_weights:
                        s = sum(self.uploaded_weights)
                        if s > 0:
                            self.uploaded_weights = [w / s for w in self.uploaded_weights]
                    bye = time.time()
                    vish = bye - oi  # Calcula o tempo decorrido
                    print(f"Tempo de execução: {vish:.4f} segundos")
                if self.cc==5:
                    print("vai rolar nada")
                if self.cc in (6, 7, 8, 9, 10):
                    oi = time.time()
                    detector_name = (
                        'DistilBERT' if self.cc == 6 else (
                            'MLP' if self.cc == 7 else (
                                'MLP+VAL' if self.cc == 8 else (
                                    'DistilBERT+MLP+LF' if self.cc == 9 else (
                                        'DistilBERT+MLP+TARGET-LF' if self.cc == 10
                                        else 'DistilBERT+MLP+UNKNOWN'
                                    )
                                )
                            )
                        )
                    )
                    val_scores = {}
                    lf_scores = {}
                    behavior_scores = {}
                    target_lf_scores = {}
                    clients_by_id = {c.id: c for c in self.clients}
                    if self.cc == 8:
                        val_scores = self.public_val_check.score_round(
                            self.global_model, self.uploaded_models, self.ids
                        )
                    if self.cc == 9:
                        lf_scores = self.label_flip_check.score_round(
                            self.global_model, self.uploaded_models, self.ids
                        )
                    if self.cc == 10:
                        target_lf_scores = self.targeted_label_flip_check.score_round(
                            self.global_model, self.uploaded_models, self.ids
                        )
                    true_positive_uploads = 0
                    malicious_uploads = 0
                    detail_rows = []
                    for idx in range(len(self.uploaded_models) - 1, -1, -1):
                        client_id = self.ids[idx]
                        client = clients_by_id.get(client_id)
                        attack_type = getattr(client, 'last_attack_type', 'unknown')
                        is_malicious_round = bool(getattr(client, 'is_malicious', False))
                        if is_malicious_round:
                            malicious_uploads += 1
                        sd = self.uploaded_models[idx].state_dict()
                        mlp_result = {'is_malicious': False, 'score': 0.0}
                        bert_result = {'is_malicious': False, 'score': 0.0}
                        if self.cc in (9, 10):
                            mlp_result = self.mlp_client_check.classify(sd)
                            bert_result = self.bert_client_check.classify(sd)
                            mlp_hit = bool(mlp_result['is_malicious'])
                            bert_hit = bool(bert_result['is_malicious'])
                        elif self.cc == 6:
                            bert_result = self.client_check.classify(sd)
                            bert_hit = bool(bert_result['is_malicious'])
                            mlp_hit = False
                        else:
                            mlp_result = self.client_check.classify(sd)
                            mlp_hit = bool(mlp_result['is_malicious'])
                            bert_hit = False
                        val_hit = bool(val_scores.get(client_id, {}).get('reject', False))
                        lf_hit = bool(lf_scores.get(client_id, {}).get('reject', False))
                        behavior_hit = bool(behavior_scores.get(client_id, {}).get('reject', False))
                        target_lf_hit = bool(target_lf_scores.get(client_id, {}).get('reject', False))
                        cc9_hit = bool(
                            mlp_hit or (bert_hit and lf_hit) or (self.cc9_lf_standalone and lf_hit)
                        ) if self.cc == 9 else False
                        cc10_hit = bool(bert_hit or mlp_hit or target_lf_hit) if self.cc == 10 else False
                        removed = bool(
                            (self.cc == 6 and bert_hit)
                            or (self.cc == 7 and mlp_hit)
                            or (self.cc == 8 and (mlp_hit or val_hit))
                            or cc9_hit
                            or cc10_hit
                        )
                        reason_parts = []
                        if mlp_hit:
                            reason_parts.append('mlp')
                        if bert_hit:
                            reason_parts.append('bert')
                        if val_hit:
                            reason_parts.append('val')
                        if lf_hit:
                            reason_parts.append('lf')
                        if behavior_hit:
                            reason_parts.append('behavior')
                        if target_lf_hit:
                            reason_parts.append('target_lf')
                        reason = '+'.join(reason_parts) if reason_parts else 'none'
                        v_val = val_scores.get(client_id, {})
                        v_lf = lf_scores.get(client_id, {})
                        v_behavior = behavior_scores.get(client_id, {})
                        v_target_lf = target_lf_scores.get(client_id, {})
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
                            'bert_hit': bert_hit,
                            'val_hit': val_hit,
                            'lf_hit': lf_hit,
                            'behavior_hit': behavior_hit,
                            'target_lf_hit': target_lf_hit,
                            'mlp_score': float(mlp_result.get('score', 0.0)),
                            'bert_score': float(bert_result.get('score', 0.0)),
                            'val_score': float(v_val.get('score', 0.0)),
                            'lf_cos': float(v_lf.get('final_cos', 0.0)),
                            'lf_worst_class_delta': float(v_lf.get('worst_class_delta', 0.0)),
                            'behavior_margin_delta': float(v_behavior.get('margin_delta', 0.0)),
                            'behavior_loss_delta': float(v_behavior.get('loss_delta', 0.0)),
                            'target_lf_score': float(v_target_lf.get('score', 0.0)),
                            'target_lf_suspect_class': int(v_target_lf.get('suspect_class', -1)),
                            'target_lf_target_class': int(v_target_lf.get('target_class', -1)),
                            'target_lf_margin_delta': float(v_target_lf.get('margin_delta', 0.0)),
                            'target_lf_loss_delta': float(v_target_lf.get('loss_delta', 0.0)),
                            'target_lf_target_prob_delta': float(v_target_lf.get('target_prob_delta', 0.0)),
                            'target_lf_head_score': float(v_target_lf.get('head_score', 0.0)),
                            'target_lf_final_delta': self._final_layer_delta(self.uploaded_models[idx]),
                            'target_lf_score_outlier': bool(v_target_lf.get('score_outlier', False)),
                            'target_lf_behavior_allowed': bool(v_target_lf.get('behavior_allowed', False)),
                            'target_lf_capped': bool(v_target_lf.get('capped_by_max_reject_fraction', False)),
                        })
                        if removed:
                            if is_malicious_round:
                                true_positive_uploads += 1
                            if self.cc == 8:
                                print(
                                    f"cc=8: removing client {client_id} ({detector_name}) "
                                    f"mlp={mlp_hit} val={val_hit} score={v_val.get('score', 0.0):.4f}"
                                )
                            elif self.cc == 9:
                                print(
                                    f"cc=9: removing client {client_id} ({detector_name}) "
                                    f"bert={bert_hit} mlp={mlp_hit} lf={lf_hit} "
                                    f"cos={v_lf.get('final_cos', 0.0):.4f} "
                                    f"class={v_lf.get('worst_class', -1)} "
                                    f"delta={v_lf.get('worst_class_delta', 0.0):.4f}"
                                )
                            elif self.cc == 10:
                                print(
                                    f"cc=10: removing client {client_id} ({detector_name}) "
                                    f"distilbert={bert_hit} mlp={mlp_hit} target_lf={target_lf_hit} "
                                    f"score={v_target_lf.get('score', 0.0):.4f} "
                                    f"class={v_target_lf.get('suspect_class', -1)}"
                                    f"->{v_target_lf.get('target_class', -1)} "
                                    f"margin_delta={v_target_lf.get('margin_delta', 0.0):.4f} "
                                    f"loss_delta={v_target_lf.get('loss_delta', 0.0):.4f} "
                                    f"head={v_target_lf.get('head_score', 0.0):.4f}"
                                )
                            else:
                                print(f'cc={self.cc}: removing client {client_id} ({detector_name} detector)')
                            self.set_client_quarantine(client_id)
                            del self.uploaded_models[idx]
                            del self.ids[idx]
                            del self.uploaded_ids[idx]
                            del self.uploaded_weights[idx]
                    self.save_cc_detail_to_csv(detail_rows)
                    self.save_cc_type_to_csv(i, detail_rows)
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
            FPR=0
            FRR = 0
            if self.cc ==2:
                FPR, FRR = self.compute_fpr_frr_cluster(self.removed_clients, self.cluster_tuples)
            if self.cc ==3:
                FPR, FRR = self.compute_fpr_frr()
            if self.cc ==6:
                FPR, FRR = self.compute_fpr_frr()
            if self.cc ==7:
                FPR, FRR = self.compute_fpr_frr()
            if self.cc ==8:
                FPR, FRR = self.compute_fpr_frr()
            if self.cc ==9:
                FPR, FRR = self.compute_fpr_frr()
            if self.cc ==10:
                FPR, FRR = self.compute_fpr_frr()
            print(f"Round {i}: False Positive Rate = {FPR:.4f}, False Rejection Rate = {FRR:.4f}")
            self.save_fpr_frr_to_csv(i, FPR, FRR)
            for client_id in quarantined_at_round_start:
                self.decrease_quarentine(client_id)
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

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
