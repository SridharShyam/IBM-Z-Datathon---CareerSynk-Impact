"""Interactive CLI Tool for Testing CareerSynk Impact Model & Inference Engine.

Allows interactive input of:
  - Target occupation (by title, keyword, or SOC code)
  - Candidate current skills (comma-separated or presets)
Computes and displays:
  - Baseline Readiness Score ([0.0, 1.0]) with detailed breakdown
  - Top Counterfactual Skill-Gap Recommendations (+Delta readiness increase & explanations)
"""

import os
import sys
import pickle
from typing import List, Dict

# Set path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, base_dir)

from src.models.readiness_score import compute_readiness, MANDATORY_DISCLAIMER
from src.models.counterfactual_ranking import rank_skill_gaps
from src.models.demo_profiles import SYNTHETIC_PROFILES


def load_artifacts():
    print("\n" + "=" * 65)
    print("CareerSynk Impact: Loading Model Artifacts...")
    print("=" * 65)

    graph_path = os.path.join(base_dir, "data", "processed", "skill_graph.pkl")
    model_path = os.path.join(base_dir, "data", "processed", "baseline_relevance_model.pkl")

    if not os.path.exists(graph_path) or not os.path.exists(model_path):
        print(f"Error: Processed artifacts missing at {graph_path} or {model_path}")
        sys.exit(1)

    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    with open(model_path, "rb") as f:
        model_artifact = pickle.load(f)

    print(f"-> Skill Graph Loaded: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges.")
    print("-> Stage A Baseline ML Model Loaded.")
    return graph, model_artifact


def find_occupations(query: str, graph) -> List[Dict[str, str]]:
    """Search for occupations in graph matching query."""
    q = query.lower().strip()
    matches = []
    for node, data in graph.nodes(data=True):
        if data.get("node_type") == "occupation":
            label = str(data.get("label", node))
            if q in node.lower() or q in label.lower():
                matches.append({
                    "id": node,
                    "label": label,
                    "skills_count": sum(1 for _ in graph.neighbors(node)),
                })
    return matches


def run_interactive():
    graph, model_artifact = load_artifacts()

    print("\n" + "#" * 65)
    print("       WELCOME TO CAREERSYNK IMPACT INTERACTIVE TEST CONSOLE")
    print("#" * 65)
    print("Test candidate readiness and counterfactual skill-gap ranking live.")

    while True:
        print("\n" + "-" * 65)
        print("STEP 1: Select Target Occupation")
        print("Popular examples: 'software developer', 'data analyst', 'manager', 'cybersecurity'")
        occ_query = input("Enter occupation keyword, title, or SOC code (or 'q' to quit): ").strip()
        if occ_query.lower() in ("q", "quit", "exit"):
            print("Exiting console. Goodbye!")
            break

        if not occ_query:
            occ_query = "software developer"
            print(f"Defaulting to: '{occ_query}'")

        matches = find_occupations(occ_query, graph)
        if not matches:
            print(f"No occupations found matching '{occ_query}'. Please try another search.")
            continue

        print(f"\nFound {len(matches)} matching occupation(s):")
        for i, m in enumerate(matches[:8], 1):
            print(f"  [{i}] {m['label']} (ID: {m['id']}) - {m['skills_count']} skills in graph")

        if len(matches) == 1:
            selected_occ = matches[0]
        else:
            choice = input(f"\nSelect occupation number [1-{min(len(matches), 8)}] (default: 1): ").strip()
            idx = 0
            if choice.isdigit() and 1 <= int(choice) <= min(len(matches), 8):
                idx = int(choice) - 1
            selected_occ = matches[idx]

        print(f"\nSelected Target Occupation: {selected_occ['label']} ({selected_occ['id']})")

        # Step 2: Skills
        print("\n" + "-" * 65)
        print("STEP 2: Enter Candidate Skills")
        print("Options:")
        print("  [1] Profile 1: CS Graduate (Python, Git, Data Structures, OOP)")
        print("  [2] Profile 2: Retail Career Switcher (Customer Service, Inventory, Sales)")
        print("  [3] Profile 3: Mid-level Tech Lead (Python, System Design, Mentoring, Agile)")
        print("  [4] Custom Skills (Enter your own comma-separated skills)")

        skill_choice = input("Select skill profile [1-4] (or type comma-separated skills): ").strip()

        candidate_skills = []
        if skill_choice == "1":
            candidate_skills = SYNTHETIC_PROFILES["Profile_1_CS_Graduate"]["skills"]
        elif skill_choice == "2":
            candidate_skills = SYNTHETIC_PROFILES["Profile_2_Retail_Career_Switcher"]["skills"]
        elif skill_choice == "3":
            candidate_skills = SYNTHETIC_PROFILES["Profile_3_Midlevel_Tech_Lead"]["skills"]
        elif skill_choice == "4" or "," in skill_choice:
            custom_input = input("Enter candidate skills (comma-separated): ").strip() if skill_choice == "4" else skill_choice
            candidate_skills = [s.strip() for s in custom_input.split(",") if s.strip()]
        else:
            # Fallback to custom string or default
            if skill_choice:
                candidate_skills = [s.strip() for s in skill_choice.split(",") if s.strip()]
            else:
                candidate_skills = ["Python", "SQL", "Git"]

        print(f"\nCandidate Skill Set ({len(candidate_skills)} skills):")
        print("  " + ", ".join(candidate_skills))

        # Step 3: Run inference
        print("\n" + "=" * 65)
        print("INFERENCE EXECUTION: Computing Readiness & Counterfactual Gaps...")
        print("=" * 65)

        # Baseline readiness
        baseline_score, breakdown = compute_readiness(
            candidate_skills=set(candidate_skills),
            occupation_id=selected_occ["id"],
            model_artifact=model_artifact,
            graph=graph,
            alpha=0.50,
        )

        print(f"\n>>> Baseline Readiness Score: {baseline_score:.4f} ({baseline_score * 100:.1f}%)")
        print(f"Evaluated across {len(breakdown)} required occupational skills.")

        # Counterfactual ranking
        df_ranked = rank_skill_gaps(
            candidate_skills=set(candidate_skills),
            occupation_id=selected_occ["id"],
            model_artifact=model_artifact,
            graph=graph,
            top_n=5,
            alpha=0.50,
        )

        print("\n>>> Top-5 Counterfactual Skill-Gap Recommendations:")
        print("    (Ranked by predicted readiness impact if acquired)")
        print("-" * 65)

        if df_ranked.empty:
            print("  Candidate already possesses all required skills for this occupation!")
        else:
            for rank, row in df_ranked.iterrows():
                print(f"  #{rank + 1}: {row['skill_name']}")
                print(f"      Predicted Delta:       +{row['predicted_delta'] * 100:.2f}% readiness (Hypothetical: {row['hypothetical_score'] * 100:.2f}%)")
                print(f"      Requirement Weight:    {row['requirement_weight']:.3f}")
                print(f"      Explanation:           {row['explanation_text']}")
                print()

        print("-" * 65)
        print(f"NOTE: {MANDATORY_DISCLAIMER}")
        print("-" * 65)


if __name__ == "__main__":
    run_interactive()
