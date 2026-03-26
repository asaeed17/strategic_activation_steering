import sys, json, glob, os

target_dir = sys.argv[1]
role = sys.argv[2]
dim = sys.argv[3]

# 1: Map expected movement. 1 = increase metric, -1 = decrease metric
EXPECTED_DIRECTIONS = {
    "proposer": {
        "firmness": 1, "empathy": -1, "narcissism": 1, "fairness_norm": -1,
        "spite": 1, "anchoring": 1, "batna_awareness": 1, "flattery": -1, "undecidedness": -1
    },
    "responder": {
        "firmness": -1, "empathy": 1, "composure": 1, "fairness_norm": -1,
        "spite": -1, "narcissism": -1, "undecidedness": -1
    }
}

dir_mult = EXPECTED_DIRECTIONS.get(role, {}).get(dim, 1)

# Find all JSON outputs EXCEPT final_best.json
files = [f for f in glob.glob(os.path.join(target_dir, "*.json")) if "final_best" not in f]
results, baseline = [], None

for f in files:
    try:
        with open(f, 'r') as fh:
            d = json.load(fh)
            a = d.get('run_info', {}).get('alpha')
            if a is not None:
                results.append((a, d, f))
                if a == 0: baseline = d
    except: pass

if baseline and results:
    def get_val(d):
        return d['summary'].get('mean_proposer_pct', 0) if role == 'proposer' else d['summary'].get('accept_rate', 0)*100
    
    base_val = get_val(baseline)
    best_score, best_data = -9999, None

    # Calculate fitness vs baseline
    for a, d, f in results:
        if a == 0: continue
        shift = (get_val(d) - base_val) * dir_mult
        valid_ratio = d['summary'].get('n_valid', 0) / max(d['summary'].get('n_games', 1), 1)
        
        # Penalize broken models (below 80% validity)
        score = -9999 if valid_ratio < 0.8 else shift * valid_ratio
        
        if score > best_score:
            best_score, best_data = score, d

    if best_data:
        # Overwrite the final_best.json with the true directional winner
        with open(os.path.join(target_dir, "final_best.json"), 'w') as fh:
            json.dump(best_data, fh, indent=2)
        print(f"    -> Override: Set best alpha to {best_data['run_info']['alpha']} (Score: {best_score:.2f}, Expected Dir: {dir_mult})")
EOF
                python3 temp_eval.py "${TARGET_DIR}" "${role}" "${dim}"
                rm -f temp_eval.py
                # ──────────────────────────────────────────────────────────────
            endif
        end
    end
end

echo "==> All dimensions and layers complete."
