from tdc import Oracle

from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit import RDLogger
from rdkit.Chem import QED
from rdkit.Chem import Mol as RDMol
RDLogger.DisableLog('rdApp.*')
import multiprocessing

import torch
import torch_geometric.data as gd
from gflownet.models import bengio2021flow
from gflownet.utils.misc import get_worker_device
from gflownet.tasks.seh_frag_moo import aux_tasks


def get_scores(smiles, mode="QED", n_process=1, models=None, pref_cond=None, vina=None, hist= {}):
    smiles_groups = []
    group_size = len(smiles) / n_process
    for i in range(n_process):
        smiles_groups += [smiles[int(i * group_size):int((i + 1) * group_size)]]

    temp_data = []
    if models is None and mode.startswith("docking"):
        pool = multiprocessing.Pool(processes = n_process)
        for index in range(n_process):
            temp_data.append(pool.apply_async(get_scores_subproc, args=(smiles_groups[index], mode, vina)))
        pool.close()
        pool.join()
        scores = []
        for index in range(n_process):
            scores += temp_data[index].get()
    else:
        return get_scores_subproc(smiles, mode, models, pref_cond=pref_cond, vina=vina, hist=hist)

    return scores


def safe(f, x, default):
    try:
        return f(x)
    except Exception:
        return default


def calc_seh_reward(graphs: list[gd.Data]):
    seh_model = bengio2021flow.load_original_model()
    device = get_worker_device()
    seh_model.to(device)
    seh_model.eval()

    batch = gd.Batch.from_data_list([i for i in graphs if i is not None]).to(device)
    preds = seh_model(batch).reshape((-1,)).data.cpu() / 8
    preds[preds.isnan()] = 0
    return preds.clip(1e-4, 100).reshape((-1,))


def mol2seh(mols: list[RDMol], default=0):
    """Calculate SEH scores for a list of molecules using the pretrained model."""
    # Load SEH model if not provided
    model = bengio2021flow.load_original_model()
    device = get_worker_device()
    model.to(device)
    model.eval()
    
    scores = []
    for mol in mols:
        if mol is None:
            scores.append(default)
            continue
            
        # Convert molecule to graph
        try:
            graph = bengio2021flow.mol2graph(mol)
            # Compute SEH score
            device = model.device if hasattr(model, "device") else get_worker_device()
            batch = gd.Batch.from_data_list([graph]).to(device)
            
            with torch.no_grad():
                pred = model(batch).reshape((-1,)).data.cpu() / 8
                pred[pred.isnan()] = 0
                pred = pred.clip(1e-4, 100).item()
        
        except Exception as e:
            scores.append(default)
            continue
        
        scores.append(pred)
    
    return scores


def get_scores_subproc(smiles, mode, models=None, default=0.0, pref_cond=None, vina=None, hist= {}):
    scores = []
    mols = [MolFromSmiles(s) for s in smiles]
    oracle_QED = Oracle(name='QED')
    oracle_SA = Oracle(name='SA')

    if mode == "QED":
        for i in range(len(smiles)):
            if mols[i]:
                scores.append([QED.qed(mols[i])])
            else:
                scores.append([default])

    elif mode == "SEH":
        scores = mol2seh(mols, default=default)
    elif mode.startswith("SEH+"):

        graphs = [bengio2021flow.mol2graph(i) for i in mols]
        assert len(graphs) == len(mols)
        is_valid = [i is not None for i in graphs]
        is_valid_t = torch.tensor(is_valid, dtype=torch.bool)

        # Compute objectives
        flat_r = []
        for obj in mode.split('+'):
            if obj == "SEH":
                flat_r.append(calc_seh_reward(graphs))
            else:
                flat_r.append(aux_tasks[obj.lower()](mols, is_valid))
        flat_r = torch.stack(flat_r, dim=1)   # (N, n_objectives)
        # qed = torch.tensor([safe(QED.qed, i, 0.0) for i in mols], dtype=torch.float32)
        # flat = torch.stack([seh, qed], dim=1)

        # Determine preferences
        prefs = pref_cond.sample(len(smiles))["preferences"].float()
        moo_scores = (flat_r * prefs).sum(1)
        scores = torch.zeros((len(smiles), len(mode.split('+')) + 1))
        scores[is_valid_t, 0] = moo_scores
        scores[is_valid_t, 1:] = flat_r
        scores = scores.tolist()  # (N, n_objectives + 1)
        # scores = [[c.item(), s.item(), q.item()] for c, s, q in zip(moo_scores, seh, qed)]

    elif mode == "DRD2":
        oracle = Oracle(name='DRD2')
        for i in range(len(smiles)):
            if mols[i] != None:
                try:
                    scores += safe(oracle, [smiles[i]], 0.0)
                except Exception as e:
                    scores += [0.0]
            else:
                scores += [0.0]

    elif mode == "GSK3B":
        oracle = Oracle(name='GSK3B')
        for i in range(len(smiles)):
            if mols[i] != None:
                try:
                    scores += safe(oracle, [smiles[i]], 0.0)
                except Exception as e:
                    scores += [0.0]
            else:
                scores += [0.0]

    elif mode == "JNK3":
        oracle = Oracle(name='JNK3')
        for i in range(len(smiles)):
            if mols[i] != None:
                try:
                    scores += safe(oracle, [smiles[i]], 0.0)
                except Exception as e:
                    scores += [0.0]
            else:
                scores += [0.0]
    elif mode == "vina":
        assert vina is not None
        
        # vina_scores = vina.run_smiles(smiles, save_path=None)  # parallel
        # if the element is smaller than -20, set as 0
        vina_scores = []  # [v if v >= -20 else 0 for v in vina_scores]
        qed_scores = []
        unseen_idx = []
        for i in range(len(smiles)):
            if mols[i]:
                canonical_s = MolToSmiles(mols[i], isomericSmiles=False)
                if MolToSmiles(mols[i]) in hist.keys():
                    vina_scores.append(hist[canonical_s]['vina'])
                    qed_scores.append(hist[canonical_s]['qed'])
                else:
                    unseen_idx.append(i)
                    vina_scores.append(0)
                    qed_scores.append(QED.qed(mols[i]))
            else:
                vina_scores.append(0)
                qed_scores.append(default)
        unseen_vina_scores = vina.run_smiles([smiles[i] for i in unseen_idx], save_path=None)

        for i, v in enumerate(unseen_vina_scores):
            vina_scores[unseen_idx[i]] = v
        vina_scores = [v if v >= -20 else 0 for v in vina_scores]
        r = [0.5 * (-0.1 * min(vina_score, 0)) + 0.5 * qed_score for vina_score, qed_score in zip(vina_scores, qed_scores)]
        scores = list(zip(r, vina_scores, qed_scores))  # (r, vina_score, qed_score)

    else:
        raise Exception("Scoring function undefined!")


    return scores
