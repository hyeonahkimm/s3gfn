from __future__ import annotations

from pathlib import Path

from rdkit import Chem

from gflownet.utils import sascore
from rxnflow.envs.action import Protocol, RxnActionType
from rxnflow.envs.reaction import BiReaction, Reaction, UniReaction
from rxnflow.envs.retrosynthesis import MultiRetroSyntheticAnalyzer, RetroSynthesisTree


class SynthesizabilityEvaluator:
    def __init__(
        self,
        num_workers: int = 4,
        invalid: float = 0.0,
        max_size: int = 50_000,
        use_retrosynthesis: bool = False,
        sa_threshold: float = 3.0,
        env: str = "stock",
        max_steps: int = 2,
    ):
        if use_retrosynthesis:
            repo_root = Path(__file__).resolve().parents[2]
            env_dir = repo_root / "data" / "envs" / env
            reaction_template_path = env_dir / "template.txt"
            building_block_path = env_dir / "building_block.smi"

            protocols: list[Protocol] = [
                Protocol("stop", RxnActionType.Stop),
                Protocol("firstblock", RxnActionType.FirstBlock),
            ]
            with reaction_template_path.open() as file:
                reaction_templates = [line.strip() for line in file]
            for i, template in enumerate(reaction_templates):
                reaction = Reaction(template)
                if reaction.num_reactants == 1:
                    protocols.append(Protocol(f"unirxn{i}", RxnActionType.UniRxn, reaction))
                elif reaction.num_reactants == 2:
                    for block_is_first in [True, False]:
                        birxn = BiReaction(template, block_is_first)
                        protocols.append(Protocol(f"birxn{i}_{block_is_first}", RxnActionType.BiRxn, birxn))

            with building_block_path.open() as file:
                blocks = [line.split()[0] for line in file]
            self.retrosynthesis_analyzer = MultiRetroSyntheticAnalyzer.create(protocols, blocks, num_workers=num_workers)
        else:
            self.retrosynthesis_analyzer = None

        self.sa_threshold = sa_threshold
        self.invalid = invalid
        self._seen: dict[str, float | RetroSynthesisTree | None] = {}
        self._max = max_size
        self._max_steps = max_steps

    def get_synthesis(self, smiles: str) -> RetroSynthesisTree | None:
        if not self.retrosynthesis_analyzer:
            return None

        canonical_smiles = self._canonicalize(smiles)
        if canonical_smiles is None:
            return None
        if canonical_smiles in self._seen:
            return self._seen[canonical_smiles]  # type: ignore[return-value]

        self.retrosynthesis_analyzer.submit(0, smiles, self._max_steps, [])
        _, retro_tree = self.retrosynthesis_analyzer.result()[0]
        self._cache(canonical_smiles, retro_tree)
        return retro_tree

    def score(self, smiles: str) -> float:
        canonical_smiles = self._canonicalize(smiles)
        if canonical_smiles is None:
            return 0.0

        if canonical_smiles in self._seen:
            cached = self._seen[canonical_smiles]
            if self.retrosynthesis_analyzer:
                return float(bool(cached))
            return float(cached < self.sa_threshold)  # type: ignore[operator]

        if self.retrosynthesis_analyzer:
            self.retrosynthesis_analyzer.submit(0, canonical_smiles, self._max_steps, [])
            _, result = self.retrosynthesis_analyzer.result()[0]
            score = float(bool(result))
        else:
            mol = Chem.MolFromSmiles(canonical_smiles)
            try:
                result = sascore.calculateScore(mol)
            except Exception:
                result = 10.0
            score = float(result < self.sa_threshold)

        self._cache(canonical_smiles, result)
        return score

    def score_batch(self, smiles_list: list[str]) -> list[float]:
        return [self.score(smiles) for smiles in smiles_list]

    @staticmethod
    def _canonicalize(smiles: str) -> str | None:
        try:
            mol = Chem.MolFromSmiles(smiles)
            return Chem.MolToSmiles(mol, isomericSmiles=False)
        except Exception:
            return None

    def _cache(self, smiles: str, value: float | RetroSynthesisTree | None):
        if len(self._seen) < self._max:
            self._seen[smiles] = value
