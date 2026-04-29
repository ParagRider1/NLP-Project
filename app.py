import streamlit as st
from src.sbert_model import SBERTModel
from src.repetition_detector import RepetitionDetector
from datasets import load_dataset
import random
import numpy as np
import pandas as pd

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Paraphrase Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 Paraphrase Detection & Semantic Similarity")
st.markdown(
    "Analyze semantic similarity and detect paraphrases between text pairs using "
    "state-of-the-art transformer models. Compare predictions across different architectures."
)

# ============================================================================
# SIDEBAR – Model Selection
# ============================================================================
st.sidebar.header("⚙️ Model Configuration")

model_descriptions = {
    "MiniLM": "Lightweight & Fast - Best for real-time applications with lower latency requirements",
    "MPNet":  "Balanced Quality - High accuracy with moderate computational cost",
    "Elite":  "Maximum Precision - DeBERTa Cross-Encoder for deep semantic reasoning",
}

model_choice = st.sidebar.radio(
    "**Choose the NLP Model**",
    options=["MiniLM", "MPNet", "Elite"],
    format_func=lambda x: f"{x}\n└─ {model_descriptions[x]}",
    help="Select a model based on your accuracy/speed requirements",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Selected Model Details")
with st.sidebar.container():
    if model_choice == "MiniLM":
        st.sidebar.info(
            "**Model:** all-MiniLM-L6-v2\n\n"
            "• 6 layers, 22M parameters\n"
            "• ~5-10ms per inference\n"
            "• Best for: Speed-critical applications"
        )
    elif model_choice == "MPNet":
        st.sidebar.info(
            "**Model:** all-mpnet-base-v2\n\n"
            "• 12 layers, 109M parameters\n"
            "• ~20-30ms per inference\n"
            "• Best for: Balanced applications"
        )
    else:
        st.sidebar.info(
            "**Model:** cross-encoder/stsb-roberta-large\n\n"
            "• Cross-Encoder architecture, 355M parameters\n"
            "• ~50-100ms per inference\n"
            "• Best for: Highest accuracy requirements"
        )

# ============================================================================
# LOAD MODEL (cached)
# ============================================================================
@st.cache_resource
def load_model(model_name):
    return SBERTModel(model_name)

model = load_model(model_choice)

# ============================================================================
# TABS
# ============================================================================
tab1, tab2 = st.tabs(["🔍 Sentence / Paragraph Similarity", "📄 Document Repetition Detector"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – Original Sentence/Paragraph Similarity
# ─────────────────────────────────────────────────────────────────────────────
with tab1:

    st.markdown("---")
    st.subheader("📚 Sample Sentence Pairs")
    st.markdown("Click on any sample below to auto-populate the input fields")

    @st.cache_data
    def load_sample_pairs():
        samples = []
        try:
            mrpc = load_dataset("glue", "mrpc", split="validation")
            mrpc_samples = random.sample(range(len(mrpc)), min(3, len(mrpc)))
            for idx in mrpc_samples:
                sample = mrpc[int(idx)]
                samples.append({
                    "sentence1": sample["sentence1"],
                    "sentence2": sample["sentence2"],
                    "label":     sample["label"],
                    "score":     1.0 if sample["label"] == 1 else 0.0,
                    "source":    "MRPC (Paraphrase Detection)",
                })
            stsb = load_dataset("glue", "stsb", split="validation")
            stsb_samples = random.sample(range(len(stsb)), min(2, len(stsb)))
            for idx in stsb_samples:
                sample = stsb[int(idx)]
                samples.append({
                    "sentence1": sample["sentence1"],
                    "sentence2": sample["sentence2"],
                    "label":     1 if sample["label"] >= 3.5 else 0,
                    "score":     sample["label"] / 5.0,
                    "source":    "STSb (Semantic Similarity)",
                })
        except Exception as e:
            st.warning(f"Could not load datasets: {e}. Using example samples instead.")
            samples = [
                {"sentence1": "A plane is taking off with a full load of people.",
                 "sentence2": "An air plane is taking off with a full load of people.",
                 "label": 1, "score": 0.95, "source": "Example (High Similarity)"},
                {"sentence1": "The cat sat on the mat.",
                 "sentence2": "The feline rested on the rug.",
                 "label": 1, "score": 0.78, "source": "Example (Paraphrase)"},
                {"sentence1": "A woman is playing the violin.",
                 "sentence2": "A man is driving a motorcycle.",
                 "label": 0, "score": 0.15, "source": "Example (Low Similarity)"},
                {"sentence1": "The price of gold went up.",
                 "sentence2": "The cost of gold increased.",
                 "label": 1, "score": 0.88, "source": "Example (Synonym Paraphrase)"},
                {"sentence1": "She loves cooking Italian food.",
                 "sentence2": "He dislikes eating spicy dishes.",
                 "label": 0, "score": 0.25, "source": "Example (Opposite Sentiment)"},
            ]
        return samples

    samples = load_sample_pairs()

    for idx, sample in enumerate(samples):
        with st.container(border=True):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"**Sample {idx + 1}** · *{sample['source']}*")
                st.text(f"S1: {sample['sentence1'][:80]}...")
                st.text(f"S2: {sample['sentence2'][:80]}...")
            with col2:
                st.markdown(f"**Score:** {sample['score']:.2f}")
            with col3:
                if st.button("Use Sample", key=f"sample_{idx}", use_container_width=True):
                    st.session_state.sentence1_input = sample["sentence1"]
                    st.session_state.sentence2_input = sample["sentence2"]
                    st.success("✅ Sample loaded!")
                    st.rerun()

    st.markdown("---")
    st.subheader("🔍 Check Similarity")
    st.markdown("Enter or paste your text pairs below:")

    if "sentence1_input" not in st.session_state:
        st.session_state.sentence1_input = ""
    if "sentence2_input" not in st.session_state:
        st.session_state.sentence2_input = ""

    col1, col2 = st.columns(2)
    with col1:
        sentence1 = st.text_area(
            "Paragraph 1", value=st.session_state.sentence1_input,
            height=150, placeholder="Enter the first text...")
        st.session_state.sentence1_input = sentence1
    with col2:
        sentence2 = st.text_area(
            "Paragraph 2", value=st.session_state.sentence2_input,
            height=150, placeholder="Enter the second text...")
        st.session_state.sentence2_input = sentence2

    bcol1, bcol2, _ = st.columns([2, 1, 1])
    with bcol1:
        if st.button("✨ Check Similarity", use_container_width=True, type="primary"):
            if sentence1.strip() and sentence2.strip():
                st.session_state.last_result = {"sentence1": sentence1, "sentence2": sentence2, "run": True}
            else:
                st.warning("⚠️ Please enter both paragraphs before checking!")
    with bcol2:
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state.sentence1_input = ""
            st.session_state.sentence2_input = ""
            st.rerun()

    if "last_result" in st.session_state and st.session_state.last_result.get("run"):
        r = st.session_state.last_result
        with st.spinner(f"🔄 Analyzing with {model_choice} model..."):
            score = model.similarity(r["sentence1"], r["sentence2"])
        is_paraphrase = score > 0.75

        st.markdown("---")
        st.subheader("📊 Results")
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.metric("Similarity Score", f"{score:.4f}")
        with rc2:
            st.metric("Paraphrase?", "✅ YES" if is_paraphrase else "❌ NO")
        with rc3:
            st.metric("Model Used", model_choice, delta={"MiniLM": "Fast", "MPNet": "Balanced", "Elite": "Precise"}[model_choice])

        st.markdown("---")
        st.subheader("💡 Interpretation")
        with st.container(border=True):
            if score >= 0.9:
                st.markdown("**🎯 Nearly Identical:** The texts are nearly identical or very close paraphrases.")
            elif score >= 0.75:
                st.markdown("**✅ Strong Paraphrase:** The texts convey essentially the same meaning despite different wording.")
            elif score >= 0.5:
                st.markdown("**⚠️ Partial Similarity:** The texts share some concepts but differ in meaning or focus.")
            elif score >= 0.25:
                st.markdown("**📍 Low Similarity:** The texts have minimal semantic overlap.")
            else:
                st.markdown("**❌ Dissimilar:** The texts are semantically unrelated.")

        st.markdown("---")
        st.subheader("📝 Input Summary")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("**Paragraph 1**")
            st.text(r["sentence1"])
        with c2:
            st.caption("**Paragraph 2**")
            st.text(r["sentence2"])
        st.session_state.last_result["run"] = False


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – Document Repetition Detector
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("📄 Document Repetition Detector")
    st.markdown(
        "Paste a **multi-paragraph document** below. "
        "The tool embeds every paragraph and flags pairs **and groups** that are "
        "semantically repetitive — paragraphs that say the same thing in different words. "
        "It also highlights the **specific sentences** driving the overlap."
    )

    rep_threshold = st.slider(
        "Similarity threshold for 'repetitive'",
        min_value=0.50, max_value=0.99, value=0.75, step=0.01,
        help="Pairs with cosine similarity ≥ this value are flagged as repetitive.",
    )

    SAMPLE_DOC = (
        "Climate change refers to long-term shifts in global temperatures and weather patterns. "
        "While some of these changes are natural, since the 19th century human activities have "
        "been the main driver of climate change, primarily due to the burning of fossil fuels.\n\n"
        "Renewable energy sources such as solar and wind power are becoming increasingly cost-competitive "
        "with traditional fossil fuels. Investment in clean energy technology has grown substantially "
        "over the past decade, driven by both policy incentives and falling technology costs.\n\n"
        "Global warming is a phenomenon characterised by the steady rise in average temperatures "
        "across the Earth over extended periods. Human-induced greenhouse gas emissions are considered "
        "the primary cause of this temperature increase since the industrial era began.\n\n"
        "Artificial intelligence is transforming industries from healthcare to finance. "
        "Machine learning algorithms can now diagnose diseases, detect fraud, and generate "
        "creative content at a level previously thought possible only for humans.\n\n"
        "Solar and wind energy have become significantly more affordable than fossil fuels in many markets. "
        "Substantial investment in green energy infrastructure has accelerated over recent years, "
        "spurred by government policies and declining equipment costs."
    )

    if st.button("📋 Load Sample Document", key="load_sample_doc"):
        st.session_state.doc_input = SAMPLE_DOC

    document_input = st.text_area(
        "Paste your document here",
        value=st.session_state.get("doc_input", ""),
        height=300,
        placeholder="Paste a multi-paragraph document...\n\nSeparate paragraphs with a blank line.",
        key="doc_textarea",
    )
    st.session_state.doc_input = document_input

    dcol1, dcol2 = st.columns([2, 1])
    with dcol1:
        analyze_btn = st.button("🔎 Detect Repetitive Paragraphs", type="primary", key="analyze_doc")
    with dcol2:
        if st.button("🗑️ Clear Document", key="clear_doc"):
            st.session_state.doc_input = ""
            st.rerun()

    if analyze_btn:
        if not document_input.strip():
            st.warning("⚠️ Please paste a document first.")
        else:
            detector = RepetitionDetector(model)
            with st.spinner("Embedding paragraphs and computing similarities…"):
                results = detector.analyze(document_input, threshold=rep_threshold)

            paragraphs        = results["paragraphs"]
            sim_matrix        = results["sim_matrix"]
            rep_pairs         = results["repetitive_pairs"]
            groups            = results["groups"]
            sentence_overlaps = results["sentence_overlaps"]
            n                 = len(paragraphs)

            st.markdown("---")

            # ── Summary banner ──────────────────────────────────────────────
            if rep_pairs:
                st.error(
                    f"⚠️  **{len(rep_pairs)} repetitive pair(s)** detected among "
                    f"{n} paragraphs across **{len(groups)} group(s)** "
                    f"(threshold: {rep_threshold:.2f})."
                )
            else:
                st.success(
                    f"✅ No repetitive paragraphs found among {n} paragraphs "
                    f"(threshold: {rep_threshold:.2f})."
                )

            # ── Repetition groups summary ───────────────────────────────────
            if groups:
                st.subheader("🗂️ Repetition Groups")
                st.caption(
                    "Paragraphs in the same group all express the same core idea. "
                    "Consider merging or removing all but one."
                )
                # Assign a colour per group for cross-referencing
                GROUP_COLORS = ["#FF6B6B", "#FFA94D", "#FFD43B", "#69DB7C",
                                "#4DABF7", "#CC5DE8", "#F06595", "#63E6BE"]
                group_color_map = {}   # para_index -> colour
                sorted_groups = sorted(groups, key=lambda s: min(s))
                for gi, g in enumerate(sorted_groups):
                    color = GROUP_COLORS[gi % len(GROUP_COLORS)]
                    sorted_g = sorted(g)
                    for idx in sorted_g:
                        group_color_map[idx] = color
                    para_labels = ", ".join(f"Paragraph {x}" for x in sorted_g)
                    st.markdown(
                        f'<div style="border-left:5px solid {color};padding:8px 14px;'
                        f'margin-bottom:8px;border-radius:4px;background:rgba(255,255,255,0.04);">'
                        f'<strong>Group {gi+1}:</strong> {para_labels} — all express the same idea.'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # ── Paragraph list with colour-coded flags ──────────────────────
            st.markdown("---")
            st.subheader(f"📝 Document — {n} Paragraph(s)")
            flagged_indices = set()
            for a, b, _ in rep_pairs:
                flagged_indices.add(a)
                flagged_indices.add(b)

            for p in paragraphs:
                idx = p["index"]
                is_flagged   = idx in flagged_indices
                border_color = group_color_map.get(idx, "#444") if rep_pairs else "#444"
                if is_flagged:
                    flag_label = f'&nbsp;<span style="color:{border_color};font-weight:bold;">● Repetitive</span>'
                else:
                    flag_label = ""
                st.markdown(
                    f'<div style="border-left:4px solid {border_color};padding:8px 14px;'
                    f'margin-bottom:10px;border-radius:4px;">'
                    f'<strong>Paragraph {idx}</strong>{flag_label}<br>'
                    f'<span style="font-size:0.88em;color:#aaa;">{p["preview"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # ── Repetitive pairs detail with sentence overlap ───────────────
            if rep_pairs:
                st.markdown("---")
                st.subheader("🔁 Repetitive Pairs — Detail")
                for rank, (a, b, score) in enumerate(rep_pairs, 1):
                    pa = next(p for p in paragraphs if p["index"] == a)
                    pb = next(p for p in paragraphs if p["index"] == b)

                    if score >= 0.90:
                        severity_icon  = "🔴"
                        severity_label = "Near-duplicate"
                    elif score >= 0.80:
                        severity_icon  = "🟠"
                        severity_label = "High overlap"
                    else:
                        severity_icon  = "🟡"
                        severity_label = "Moderate overlap"

                    with st.expander(
                        f"#{rank}  Paragraph {a} ↔ Paragraph {b}  —  "
                        f"similarity {score:.3f}  {severity_icon} {severity_label}",
                        expanded=(rank == 1),
                    ):
                        ca, cb = st.columns(2)
                        with ca:
                            st.markdown(f"**Paragraph {a}**")
                            st.info(pa["text"])
                        with cb:
                            st.markdown(f"**Paragraph {b}**")
                            st.info(pb["text"])

                        if score >= 0.90:
                            st.error("🔴 **Near-duplicate** — strongly consider removing or merging one paragraph.")
                        elif score >= 0.80:
                            st.warning("🟠 **High overlap** — review whether both paragraphs are necessary.")
                        else:
                            st.warning("🟡 **Moderate overlap** — these paragraphs share significant content.")

                        # Sentence-level overlap
                        overlaps = sentence_overlaps.get((a, b), [])
                        if overlaps:
                            st.markdown("**📌 Overlapping sentences driving the repetition:**")
                            for si, (sa, sb, ss) in enumerate(overlaps[:5], 1):   # top-5
                                pct = int(ss * 100)
                                bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
                                st.markdown(
                                    f'<div style="background:rgba(255,255,255,0.05);'
                                    f'border-radius:6px;padding:10px 14px;margin-bottom:8px;">'
                                    f'<div style="font-size:0.8em;color:#aaa;margin-bottom:4px;">'
                                    f'Sentence pair #{si} — {pct}% similar &nbsp; <code>{bar}</code></div>'
                                    f'<div style="color:#f8f9fa;margin-bottom:4px;">🅐 {sa}</div>'
                                    f'<div style="color:#f8f9fa;">🅑 {sb}</div>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.caption("No dominant sentence overlap detected (the whole paragraphs are generally similar).")

            # ── Similarity heatmap ──────────────────────────────────────────
            st.markdown("---")
            st.subheader("🗺️ Paragraph Similarity Heatmap")
            st.caption(
                "Each cell shows cosine similarity between two paragraphs. "
                "Red shades = above repetition threshold. Diagonal = 1.0 (self)."
            )

            labels = [f"P{p['index']}" for p in paragraphs]
            df_sim = pd.DataFrame(sim_matrix, index=labels, columns=labels).round(3)

            def colour_cell(val):
                if val >= 0.99:
                    return "background-color:#ccc;color:#000;"
                elif val >= rep_threshold:
                    intensity = int(180 * (val - rep_threshold) / (1.0 - rep_threshold))
                    return f"background-color:rgb(255,{180-intensity},{180-intensity});color:#000;"
                else:
                    g = int(160 + 95 * val)
                    return f"background-color:rgb(220,{g},220);color:#000;"

            st.dataframe(df_sim.style.map(colour_cell), use_container_width=True)

            # ── Download report ─────────────────────────────────────────────
            st.markdown("---")
            report_lines = [
                "=" * 60,
                "DOCUMENT REPETITION REPORT",
                "=" * 60,
                "",
                results["summary"],
                "",
            ]

            if groups:
                report_lines.append("--- Repetition Groups ---")
                sorted_groups = sorted(groups, key=lambda s: min(s))
                for gi, g in enumerate(sorted_groups, 1):
                    sorted_g = sorted(g)
                    report_lines.append(
                        f"Group {gi}: Paragraphs {', '.join(str(x) for x in sorted_g)}"
                        f" all express the same idea."
                    )
                report_lines.append("")

            if sentence_overlaps:
                report_lines.append("--- Sentence-Level Overlaps ---")
                for (a, b), overlaps in sentence_overlaps.items():
                    report_lines.append(f"\nParagraph {a} <-> Paragraph {b}:")
                    for sa, sb, ss in overlaps[:3]:
                        report_lines.append(f"  [{ss:.2f}] A: {sa}")
                        report_lines.append(f"         B: {sb}")
                report_lines.append("")

            report_lines.append("--- Full Paragraph Texts ---")
            for p in paragraphs:
                flag = " [REPETITIVE]" if p["index"] in flagged_indices else ""
                report_lines.append(f"\nParagraph {p['index']}{flag}:\n{p['text']}\n")

            st.download_button(
                label="⬇️ Download Repetition Report (.txt)",
                data="\n".join(report_lines),
                file_name="repetition_report.txt",
                mime="text/plain",
            )