"""Pure-Python reference for the C++ leakage flow tracer (test only).

`trace_flows_reference` takes the same compiled op tables as `trace_flows_cpp`
(`_native/marginal_dem_kernels.py`) and ports `trace_flows_impl`
(`_native/marginal_dem_kernels.cpp`) statement by statement. It keeps the C++ floating-point
evaluation order, so the two must agree bit for bit.
"""

import bisect
import copy
import random
import re

import numpy as np
import pytest
import stim  # type: ignore[import-untyped]

from stimside.dem_generators import marginal_builder

K_ONE = 1.0 - 1e-9
FLAG, PROJ_Z, TRANS_1, TRANS_2, U1Q, U2Q, NOOP, BARE_1Q, BARE_RESET, BARE_OTHER, COND = range(1, 12)


class _Flow:
    def __init__(self, h, s, tracked, alive, j, next_group, can_fork, src_weight, mix, ref_known):
        self.h, self.s, self.tracked, self.desynced = h, s, tracked, False
        self.alive, self.done, self.j, self.next_group = alive, False, j, next_group
        self.can_fork, self.src_weight, self.mix = can_fork, src_weight, mix
        self.ref_known = ref_known
        self.sites: list[int] = []
        self.flags: list[list] = []  # [flag_id, prob, alive]

    def branch(self, out, h, s, alive, prob):
        """A copy of this flow on (h, s) that does not branch again; flag weights scale by prob."""
        br = copy.copy(self)
        br.sites, br.flags = list(self.sites), [list(x) for x in self.flags]
        br.h, br.s, br.alive, br.can_fork = h, s, alive, False
        br.mix = br.alive
        for fl in br.flags:
            fl[2] *= prob
        out.append(br)
        return br


class _Tracer:
    def __init__(self, kw):
        self.kw = kw
        self.ops = kw["ops_arr"]
        self.tq = [int(x) for x in kw["target_qubits"]]
        self.conj = [int(x) for x in kw["conj_table"]]
        self.touch = [int(x) for x in kw["touch_indices"]]
        self.touch_offsets = [int(x) for x in kw["touch_offsets"]]
        self.source_mode = kw["source_mode"]
        self.unique_sites: list[tuple] = []
        self.site_map: dict[tuple, int] = {}
        self.seeds: list[_Flow] = []

    def site(self, j, before, q0, q1, ch):
        key = (j, before, q0, q1, ch)
        if key not in self.site_map:
            self.site_map[key] = len(self.unique_sites)
            self.unique_sites.append(key)
        return self.site_map[key]

    def depol(self, f, before, q0, q1=-1):
        """depolarize_1 (q1 = -1) and depolarize_1_or_2."""
        if q0 < 0:
            q0, q1 = q1, -1
        if q0 < 0:
            return
        f.sites.append(self.site(f.j, before, q0, q1, 0))
        if f.h in (q0, q1):
            f.tracked, f.desynced = 0, False

    def bare(self, f, op, ts, subkind):
        if subkind == 3:
            return
        if subkind == 4:
            for t in ts:
                if t == f.h and f.tracked != 0:
                    f.tracked = self.conj[op.gate_id * 7 + f.tracked]
            return
        if subkind == 2:
            f.tracked, f.ref_known, f.desynced = op.reset_basis, False, False
            return
        if f.desynced:
            self.depol(f, 1, f.h)
        f.tracked, f.ref_known = op.reset_basis, False

    def skip(self, f, op, ts, is_1q):
        if is_1q:
            row = self.conj[op.gate_id * 7 : op.gate_id * 7 + 7]
            if f.tracked != 0:
                nxt = row[f.tracked]
                fixes = nxt == f.tracked
                f.tracked = nxt
                if fixes:
                    return
            elif row[1:4] == [1, 2, 3]:
                return
            f.desynced = True
            return
        q0, q1 = ts[0], (ts[1] if len(ts) > 1 else -1)
        leg = 0 if q0 == f.h else 1
        partner = q1 if leg == 0 else q0
        needed = op.ctrl_leg0_basis if leg == 0 else op.ctrl_leg1_basis
        if needed != 0 and partner >= 0:
            if (0 if f.ref_known else f.tracked) != needed:
                self.depol(f, 1, f.h)
                f.tracked, f.ref_known, f.desynced = needed, False, True
        else:
            self.depol(f, 0, f.h, partner)

    def fork(self, f, h, s, prob, out):
        br = f.branch(out, h, s, f.alive * prob, prob)
        if h != f.h:
            br.tracked, br.ref_known, br.desynced = 0, False, False

    def run(self, f, out):
        kw = self.kw
        while not f.done:
            if f.next_group < 0:
                if not 0 <= f.h < kw["num_qubits"]:
                    break
                lo, hi = self.touch_offsets[f.h], self.touch_offsets[f.h + 1]
                k = bisect.bisect_right(self.touch, f.j, lo, hi)
                if k == hi:
                    break
                f.j, f.next_group = self.touch[k], 0
            if not 0 <= f.j < kw["num_ops"]:
                break
            op = self.ops[f.j]
            ts = self.tq[op.target_offset : op.target_offset + op.target_count]
            aux = range(op.aux_offset, op.aux_offset + op.aux_count)
            if op.kind == FLAG:
                probs = kw["meas_probs_arr"]
                p = next((probs[m].prob for m in aux if probs[m].state == f.s), 0.0)
                fids = [ts[i + 1] for i in range(0, len(ts), 2) if ts[i] == f.h]
                if p > 0.0 and fids and fids[-1] >= 0:
                    f.flags.append([fids[-1], p, f.alive])
            elif op.kind == PROJ_Z:
                self.depol(f, 1, f.h)
                self.depol(f, 0, f.h)
            elif op.kind == TRANS_1:
                self.trans1(f, ts, aux, out)
            elif op.kind == TRANS_2:
                self.trans2(f, ts, aux, out)
            elif op.kind == U1Q:
                for i, t in enumerate(ts):
                    if t == f.h:
                        self.skip(f, op, ts[i : i + 1], True)
            elif op.kind == U2Q:
                for i in range(0, len(ts) - 1, 2):
                    if f.h in (ts[i], ts[i + 1]):
                        self.skip(f, op, ts[i : i + 2], False)
            elif op.kind in (BARE_1Q, BARE_RESET, BARE_OTHER):
                self.bare(f, op, ts, {BARE_1Q: 4, BARE_RESET: 2, BARE_OTHER: 0}[op.kind])
            elif op.kind == COND:
                self.cond(f, op, ts, aux)
            f.next_group = -1

    def trans1(self, f, ts, aux, out):
        rules, branches = self.kw["trans1_rules_arr"], self.kw["move_branches_arr"]
        for g in range(max(f.next_group, 0), len(ts)):
            if ts[g] != f.h:
                continue
            rule = next((rules[r] for r in aux if rules[r].state == f.s), None)
            p_unleak = rule.p_unleak if rule else 0.0
            next_s = rule.next_state if rule else -1
            forked = 0.0
            if rule and rule.fork_count > 0 and f.can_fork and out is not None:
                f.next_group = g + 1
                for m in range(rule.fork_offset, rule.fork_offset + rule.fork_count):
                    self.fork(f, f.h, branches[m].state, branches[m].prob, out)
                    forked += branches[m].prob
                for fl in f.flags:
                    fl[2] *= 1.0 - forked
            if p_unleak > 0.0:
                self.depol(f, 0, f.h)
            if p_unleak >= K_ONE:
                f.done = True
                break
            f.alive *= 1.0 - p_unleak - forked
            if next_s >= 2:
                f.s = next_s

    def trans2(self, f, ts, aux, out):
        rules, branches = self.kw["trans2_rules_arr"], self.kw["move_branches_arr"]
        for k in range(max(f.next_group, 0), len(ts) // 2):
            q0, q1 = ts[2 * k], ts[2 * k + 1]
            if f.h not in (q0, q1):
                continue
            f.next_group = k + 1
            leg = 0 if f.h == q0 else 1
            partner = q1 if leg == 0 else q0
            rule = next(
                (rules[r] for r in aux if rules[r].state == f.s and rules[r].leg == leg), None
            )
            p_clear = rule.p_clear if rule else 0.0
            p_move = rule.p_move if rule else 0.0
            if not self.source_mode and f.can_fork and rule:
                for m in range(rule.spread_offset, rule.spread_offset + rule.spread_count):
                    sb = branches[m]
                    self.seeds.append(
                        _Flow(partner, sb.state, 0, f.alive * sb.prob, f.j, k + 1, False,
                              f.src_weight, 0.0, False)
                    )
            can_branch = bool(rule and rule.fork_count > 0 and f.can_fork and out is not None)
            forks = range(rule.fork_offset, rule.fork_offset + rule.fork_count) if rule else ()
            forked = 0.0
            if can_branch and p_move < K_ONE:
                for m in forks:
                    self.fork(f, f.h, branches[m].state, branches[m].prob, out)
                    if branches[m].channel_id >= 0:
                        out[-1].sites.append(self.site(f.j, 0, partner, -1, branches[m].channel_id))
                    forked += branches[m].prob
            if p_clear + p_move > 0.0:
                self.depol(f, 0, f.h)
            if can_branch and p_move >= K_ONE:
                for m in forks:
                    self.fork(f, partner, branches[m].state, branches[m].prob, out)
                    forked += branches[m].prob
            if 0.0 < p_move < K_ONE and f.can_fork and out is not None and rule:
                for m in range(rule.move_offset, rule.move_offset + rule.move_count):
                    mb = branches[m]
                    br = f.branch(out, partner, mb.state, f.alive * mb.prob, mb.prob)
                    br.tracked, br.desynced = 0, False
                stay_scale = 1.0 - p_move - forked
                for fl in f.flags:
                    fl[2] *= stay_scale
            elif forked > 0.0:
                for fl in f.flags:
                    fl[2] *= 1.0 - forked
            ch = rule.channel_id if rule else -1
            if forked > 0.0 and p_move < K_ONE:
                ch = rule.fork_channel_id
            if ch >= 0:
                f.sites.append(self.site(f.j, 0, partner, -1, ch))
            if p_move >= K_ONE and rule:
                f.alive *= 1.0 - forked
                f.h, f.s, f.tracked = partner, rule.max_move_state, 0
            elif p_clear + p_move >= K_ONE:
                f.done = True
                break
            else:
                f.alive *= 1.0 - p_clear - p_move - forked
                if rule and rule.stay_next_state >= 2:
                    f.s = rule.stay_next_state

    def cond(self, f, op, ts, aux):
        rules = self.kw["cond_rules_arr"]
        base = op.cond_base_fires != 0
        for g in range(len(ts) // 4):
            q0, q1, subj0, subj1 = ts[4 * g : 4 * g + 4]
            in_group = q0 == f.h or (q1 >= 0 and q1 == f.h)
            mask = (1 if subj0 == f.h else 0) | (2 if subj1 == f.h else 0)
            traj = base
            if mask:
                hits = [r for r in aux if (rules[r].state, rules[r].mask) == (f.s, mask)]
                traj = rules[hits[0]].fires != 0 if hits else base
            if traj == base:
                if traj and in_group:
                    self.bare(f, op, [q0, q1], op.cond_subkind)
                continue
            if op.cond_subkind == 3:
                if traj and op.cond_channel_id >= 0 and q0 >= 0:
                    f.sites.append(self.site(f.j, 0, q0, q1, op.cond_channel_id))
                continue
            if base and in_group and op.cond_subkind in (4, 5):
                self.skip(f, op, [q0, q1], op.cond_subkind == 4)
                continue
            self.depol(f, 0, q0, q1)

    def trace(self):
        """Trace every source; returns all flows and, in source mode, their source and mix."""
        flows, flow_source, flow_mix = [], [], []
        for s_idx, src in enumerate(self.kw["sources_arr"]):
            if not 0 <= src.op_index < self.kw["num_ops"]:
                continue
            root = _Flow(src.qubit, src.state, src.init_tracked, 1.0, src.op_index, src.group + 1,
                         True, src.weight, 1.0, src.init_tracked != 0)
            if src.partner_channel_id >= 0 and src.partner >= 0:
                root.sites.append(self.site(src.op_index, 0, src.partner, -1,
                                            src.partner_channel_id))
            branches: list[_Flow] = []
            self.run(root, branches)
            if self.source_mode:
                total = 0.0
                for br in branches:
                    total += br.mix
                flow_source.append(s_idx)
                flow_mix.append(max(0.0, 1.0 - total))
            flows.append(root)
            for br in branches:
                self.run(br, None)
                if self.source_mode:
                    flow_source.append(s_idx)
                    flow_mix.append(br.mix)
                flows.append(br)
            for pf in self.seeds:
                self.run(pf, None)
                flows.append(pf)
            self.seeds.clear()
        return flows, flow_source, flow_mix

    def finish(self, flows, flow_source, flow_mix):
        """Candidates and predecessors per flag (per source in source mode), then compaction."""
        num_out = len(self.kw["sources_arr"]) if self.source_mode else self.kw["num_flags"]
        cands: list[list] = [[] for _ in range(num_out)]
        preds: list[set] = [set() for _ in range(num_out)]
        used = [False] * len(flows)
        for k, fl in enumerate(flows):
            if self.source_mode:
                if flow_mix[k] > 0.0:
                    cands[flow_source[k]].append((k, flow_mix[k]))
                    used[k] = True
                continue
            survive, prev = 1.0, -1
            for fid, prob, alive in fl.flags:
                w = fl.src_weight * alive * survive * prob
                if w > 0.0:
                    cands[fid].append((k, w))
                    used[k] = True
                if prev >= 0:
                    preds[fid].add(prev)
                survive *= 1.0 - prob
                prev = fid
        flow_remap: dict[int, int] = {}
        site_remap: dict[int, int] = {}
        sites, offsets, ids = [], [0], []
        for k, fl in enumerate(flows):
            if not used[k]:
                continue
            flow_remap[k] = len(flow_remap)
            for sid in fl.sites:
                if sid not in site_remap:
                    site_remap[sid] = len(sites)
                    sites.append(self.unique_sites[sid])
                ids.append(site_remap[sid])
            offsets.append(len(ids))
        c_off, c_flows, c_w, p_off, p_flags = [0], [], [], [0], []
        for f_idx in range(num_out):
            for k, w in cands[f_idx]:
                c_flows.append(flow_remap[k])
                c_w.append(w)
            c_off.append(len(c_flows))
            p_flags.extend(sorted(preds[f_idx]))
            p_off.append(len(p_flags))
        i32 = [np.asarray(x, dtype=np.int32) for x in (offsets, ids, c_off, c_flows)]
        p32 = [np.asarray(x, dtype=np.int32) for x in (p_off, p_flags)]
        sites_arr = np.asarray(sites, dtype=np.int32).reshape(-1, 5)
        return (sites_arr, *i32, np.asarray(c_w, dtype=np.float64), *p32)


def trace_flows_reference(**kw):
    """Same keyword arguments and 8 output arrays as `trace_flows_cpp`."""
    tracer = _Tracer(kw)
    return tracer.finish(*tracer.trace())


_LT1 = ["(0.1, U-->2)", "(0.05, U-->3)", "(0.3, 2-->U)", "(1.0, 2-->U)", "(0.2, 2-->3)",
        "(0.2, 2<->3)", "(0.5, 2-->0)", "(0.4, 3-->U)"]
_LT2 = ["(0.1, U_U-->2_D)", "(0.1, U_U-->2_X)", "(0.05, U_U-->U_3)", "(0.3, 2_U-->U_2)",
        "(0.25, 2_U-->U_3)", "(0.5, 2_U-->U_D)", "(1.0, 2_U-->U_D)", "(0.2, 2_U-->2_2)",
        "(0.25, 2_U-->3_U)", "(0.25, 2_U-->3_X)", "(0.2, U_2-->D_U)", "(0.3, 3_U-->U_2)",
        "(1.0, 2_U-->U_2)", "(0.5, 2_U-->U_2) (0.5, 2_U-->U_3)",
        "(0.5, 2_U-->U_2) (0.2, 2_U-->U_3)", "(0.4, 2_U-->3_2)", "(0.3, 3_U-->3_3)"]


def _pick(rng, opts):
    """1-3 rules, at most one per input state."""
    out: dict[str, str] = {}
    for rule in rng.sample(opts, rng.randint(1, 3)):
        out.setdefault(re.split("-->|<->", rule[rule.index(",") + 2 :])[0], rule)
    return " ".join(out.values())


def _random_leakage_circuit(rng, nq=4, n_ops=14):
    """Random leakage circuit with detectors only on (deterministic) MPAD flag records."""
    lines, nm, flag_recs = ["R " + " ".join(map(str, range(nq))), "TICK"], 0, []
    for _ in range(n_ops):
        qs = rng.sample(range(nq), rng.randint(1, nq))
        t1 = " ".join(map(str, qs))
        t2 = " ".join(map(str, qs[: len(qs) // 2 * 2] or rng.sample(range(nq), 2)))
        r = rng.random()
        if r < 0.18:
            lines.append(f"I_ERROR[LEAKAGE_TRANSITION_1: {_pick(rng, _LT1)}] {t1}")
        elif r < 0.32:
            lines.append(f"II_ERROR[LEAKAGE_TRANSITION_2: {_pick(rng, _LT2)}] {t2}")
        elif r < 0.45:
            lines.append(f"MPAD[LEAKAGE_MEASUREMENT: (1.0, 2) (0.6, 3): {t1}] " + "0 " * len(qs))
            flag_recs.extend(range(nm, nm + len(qs)))
            nm += len(qs)
        elif r < 0.57:
            lines.append(f"{rng.choice(['H', 'S', 'X', 'SQRT_X', 'S_DAG', 'I', 'Z', 'H_YZ'])} {t1}")
        elif r < 0.72:
            gate = rng.choice(["CX", "CZ", "SWAP", "ISWAP", "XCZ", "SQRT_XX", "CY"])
            lines.append(f"{gate} {t2}")
            lines.append("TICK")
        elif r < 0.77:
            lines.append(f"{rng.choice(['X_ERROR(0.1)', 'DEPOLARIZE1(0.05)'])} {t1}")
        elif r < 0.81:
            lines.append(f"DEPOLARIZE2(0.05) {t2}")
        elif r < 0.86:
            lines.append(f"{rng.choice(['M', 'MR', 'MX'])} {t1}")
            nm += len(qs)
        elif r < 0.89:
            lines.append(f"M[LEAKAGE_PROJECTION_Z: (0.9, 2)] {t1}")
            nm += len(qs)
        elif r < 0.92:
            lines.append(f"{rng.choice(['R', 'RX'])} {t1}")
        elif r < 0.95:
            cond = rng.choice(["U", "2", "U 2", "3"])
            lines.append(f"{rng.choice(['H', 'I', 'S'])}[CONDITIONED_ON_SELF: {cond}] {t1}")
        elif r < 0.98:
            cond = rng.choice(["(U, U)", "(2, U)", "(U, U) (2, U)", "(U, 2)"])
            lines.append(f"{rng.choice(['CX', 'SWAP'])}[CONDITIONED_ON_PAIR: {cond}] {t2}")
            lines.append("TICK")
        else:
            lines.append(f"DEPOLARIZE1[CONDITIONED_ON_SELF: 2](0.2) {t1}")
        lines.append("TICK")
    lines.append("M " + " ".join(map(str, range(nq))))
    nm += nq
    lines += [f"DETECTOR rec[{r - nm}]" for r in flag_recs]
    return stim.Circuit("\n".join(lines))


def test_trace_flows_reference_matches_cpp_exactly_on_random_leakage_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = marginal_builder.trace_flows_cpp
    calls = []

    def spy(**kw):
        out = real(**kw)
        for a, b in zip(out, trace_flows_reference(**kw), strict=True):
            assert a.dtype == b.dtype and a.shape == b.shape, (a, b)
            assert np.array_equal(a.view(np.uint8), b.view(np.uint8)), (a, b)  # bit-exact
        calls.append((kw["source_mode"], len(out[4])))
        return out

    monkeypatch.setattr(marginal_builder, "trace_flows_cpp", spy)
    rng = random.Random(10)  # of seeds 1-12, the one whose circuits catch the most mutants
    for _ in range(50):
        circuit = _random_leakage_circuit(rng)
        uc = rng.random() < 0.7
        for oracle in (False, True):
            marginal_builder.Builder(circuit, uc).build(decompose_errors=False, oracle=oracle)
    assert [m for m, _ in calls] == [False, True] * 50
    assert sum(n for _, n in calls) > 100  # the comparison covers many weighted candidates
