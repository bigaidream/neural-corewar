"""Resident GPU referee and fresh-GPU deterministic replay verifier."""
from pathlib import Path
import numpy as np
from ..artifacts import read_record, write_record, sha256_digest
from ..grammar import BudgetSpec
from ..evaluation import Evaluator
from ..model_io import ModelSession as BaseModelSession
from ..merge import bf16_to_f32
from ..edit_geometry import compile_geometry
from .spec import compile_v3_patch
from .probes_v3 import ProbeSandbox


class ModelSessionV3(BaseModelSession):
    def compile(self, patch, budget):
        return compile_v3_patch(
            patch, self.current, self.base, self.resolve_token,
            budget=budget, arena=self.arena, allow_empty=True,
        )


class StudyWorker:
    def __init__(self, session, worker, inputs, directory):
        self.session, self.worker, self.inputs = session, worker, inputs
        self.directory = Path(directory)
        self.evaluator = Evaluator(session, worker, inputs, self.directory)
        self.probes = ProbeSandbox(
            session, worker, inputs['public_inputs'],
            self.directory / 'probe-receipts',
            protocol_version=inputs.get('protocol_version', 'v3_corewar'),
            decoding=inputs.get('decoding'),
        )

    def probe_call(self, context):
        def dispatch(turn, action):
            payload = {k: context[k] for k in ('slot_id', 'own', 'rival', 'budget', 'condition')}
            payload.update(
                turn=turn,
                blind=bool(context.get('blind', False)),
                state=context.get('state', []),
                name=action['name'],
                args=action['args'],
                protocol_version='v3_corewar',
                decoding=context['decoding'],
            )
            try:
                return self.probes.dispatch(payload)['public']
            except Exception as exc:
                return {'error_type': type(exc).__name__}
        return dispatch

    def evaluate(self, p):
        if p['split'] not in ('dev', 'heldout'):
            raise ValueError('unknown split')
        return self.evaluator.mirrored(
            p['tag'], p['pair'], p['patches_A'], p['patches_B'],
            BudgetSpec(**self.inputs['budget']),
            states=p.get('states'),
            bank=self.inputs['banks'][p['split']],
            format_errors=p['format_errors'],
        )

    def geometry(self, p):
        path = self.directory / 'geometry' / f'{sha256_digest(p)}.json'
        if path.exists():
            return read_record(path)
        budget = BudgetSpec(**self.inputs['budget'])
        value = compile_geometry(self.session, p, budget)
        with self.session.patched(state=p['state'], budget=budget):
            for rows in value['players'].values():
                for row in rows:
                    l, c = row['layer'], row['column']
                    d = bf16_to_f32(self.session.current[l][:, c]).astype(np.float64) - bf16_to_f32(self.session.base[l][:, c]).astype(np.float64)
                    dv = np.asarray(row['nominal_vector'])
                    den = np.linalg.norm(d) * np.linalg.norm(dv)
                    row['unrounded_post_drift_norm'] = float(np.linalg.norm(d + dv))
                    row['pre_drift_cosine'] = float(np.dot(d, dv) / den) if den else 0.0
        write_record(path, value)
        return value
