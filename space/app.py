"""Gradio demo — the app is a report, not a label (§14.1).

Two tabs:

* **Batch** (primary) — upload a CSV of feedback, get a bias-corrected
  satisfaction rate with a confidence interval plus an aspect breakdown.
* **Single comment** — one text box, for inspection and for the sceptical
  reader.

Everything routes through ``src.predict.SatisfactionModel`` so the served
pipeline is byte-identical to the evaluated one (§3.1).

Deploy notes:
* Model loads and quantises at **module scope**, never per request (§3.4).
* Do **not** call ``torch.set_num_threads(1)`` here. That belongs in the
  benchmark (§13.2); carrying it into serving halves throughput.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.aspects import attribute_sentences, load_lexicons  # noqa: E402
from src.estimate import acc_prevalence  # noqa: E402
from src.normalize import is_empty  # noqa: E402
from src.predict import POSITIVE, SatisfactionModel, load_config  # noqa: E402

MAX_ROWS = 5_000  # cap so a free CPU Space cannot be wedged

# --------------------------------------------------------------------------
# Load once, at import (§3.4 -- never per request)
# --------------------------------------------------------------------------
LOAD_ERROR: str | None = None
MODEL: SatisfactionModel | None = None
CFG = None
# Corpus-derived lexicons from notebook 07, so the demo's aspect table matches
# the published one. Falls back to the generic starter set if absent.
LEXICONS = load_lexicons()
try:
    CFG = load_config(REPO / "inference_config.json")
    MODEL = SatisfactionModel(CFG, quantize=True)
except Exception as exc:  # noqa: BLE001 - surface any startup failure in the UI
    LOAD_ERROR = str(exc)


def _looks_french(text: str) -> bool:
    """Cheap heuristic language guard (§3.5).

    A reader *will* type English into this box. CamemBERT on English emits
    confident nonsense, and "92% satisfait" about an English sentence
    undercuts the whole project. We warn rather than block: blocking is worse
    UX and would trust a tiny classifier to gate the bigger one.

    Swap in py3langid for a real check; this keyword heuristic avoids the
    dependency and catches the common case.
    """
    fr = {"le", "la", "les", "de", "des", "un", "une", "et", "est", "pas",
          "je", "ce", "qui", "que", "pour", "mais", "très", "avec", "plus"}
    en = {"the", "is", "are", "and", "was", "this", "that", "with", "for",
          "not", "but", "very", "have", "it", "of", "to"}
    words = {w.strip(".,!?;:").lower() for w in text.split()}
    if len(words) < 4:
        return True  # too short to judge; don't nag
    return len(words & fr) >= len(words & en)


# --------------------------------------------------------------------------
# Single comment
# --------------------------------------------------------------------------
def analyse_one(text: str):
    if LOAD_ERROR:
        return {"erreur": 1.0}, f"⚠️ Modèle non chargé : {LOAD_ERROR}"
    if is_empty(text):
        return {}, "Saisissez un commentaire."

    pred = MODEL.predict([text])[0]
    notes = []
    if not _looks_french(text):
        notes.append(
            "⚠️ Ce texte ne semble pas être en français ; "
            "les résultats ne sont pas fiables."
        )
    if pred.truncated:
        notes.append(
            f"ℹ️ Texte tronqué à {CFG.max_length} tokens. L'avis final d'un "
            "commentaire se trouve souvent dans la dernière phrase."
        )
    if not pred.confident:
        notes.append(
            "ℹ️ Score dans la bande neutre : le modèle s'abstient. "
            "Les données d'entraînement ne contiennent pas d'avis neutres "
            "(voir les limites)."
        )

    scores = {"positif": pred.p_pos, "négatif": 1 - pred.p_pos}
    return scores, f"**{pred.label.upper()}**\n\n" + "\n\n".join(notes)


# --------------------------------------------------------------------------
# Batch
# --------------------------------------------------------------------------
def analyse_batch(file):
    if LOAD_ERROR:
        return f"⚠️ Modèle non chargé : {LOAD_ERROR}", None, None
    if file is None:
        return "Importez un fichier CSV.", None, None

    try:
        df = pd.read_csv(file.name)
    except Exception as exc:  # noqa: BLE001
        return f"Lecture du CSV impossible : {exc}", None, None

    col = next((c for c in ("text", "texte", "commentaire", "review", "avis")
                if c in df.columns), None)
    if col is None:
        return (
            "Colonne de texte introuvable. Attendu l'une de : "
            f"text, texte, commentaire, review, avis. Trouvé : {list(df.columns)}",
            None, None,
        )

    texts = [t for t in df[col].astype(str).tolist() if not is_empty(t)][:MAX_ROWS]
    if not texts:
        return "Aucun texte exploitable dans le fichier.", None, None

    preds = MODEL.predict(texts, batch_size=32)
    decided = [p for p in preds if p.confident]
    n_neutral = len(preds) - len(decided)

    if not decided:
        return "Tous les commentaires tombent dans la bande neutre.", None, None

    est = MODEL.estimate_satisfaction(texts)
    summary = (
        f"## {est.rate:.0%} de clients satisfaits\n\n"
        f"**Intervalle de confiance 95 % : {est.ci_low:.0%} – {est.ci_high:.0%}**\n\n"
        f"- Commentaires analysés : **{len(preds)}**\n"
        f"- Dont neutres / mixtes (exclus du taux) : **{n_neutral}** "
        f"({n_neutral/len(preds):.0%})\n"
        f"- Comptage brut non corrigé : {est.naive_rate:.0%} "
        f"— écart de {abs(est.rate - est.naive_rate)*100:.1f} points\n\n"
        f"> Le comptage brut des prédictions est biaisé dès que les erreurs du "
        f"classifieur sont asymétriques. Le taux ci-dessus applique la "
        f"correction *Adjusted Classify and Count* (TPR={CFG.tpr:.3f}, "
        f"FPR={CFG.fpr:.3f})."
    )

    dist = pd.DataFrame(
        [{"étiquette": lab,
          "n": sum(p.label == lab for p in preds),
          "part": f"{sum(p.label == lab for p in preds)/len(preds):.0%}"}
         for lab in ("positif", "neutre", "négatif")]
    )

    # Per-aspect sentiment, corrected the same way (§11 step 4).
    rows = []
    for aspect in LEXICONS:
        sents = [s for t in texts
                 for s, asp in attribute_sentences(t, LEXICONS) if aspect in asp]
        if len(sents) < 5:
            continue
        sp = [p for p in MODEL.predict(sents, batch_size=64) if p.confident]
        if not sp:
            continue
        raw = sum(p.label == POSITIVE for p in sp) / len(sp)
        rows.append({
            "aspect": aspect,
            "mentions": len(sents),
            "satisfaction corrigée": f"{acc_prevalence(raw, CFG.tpr, CFG.fpr):.0%}",
            "brut": f"{raw:.0%}",
        })
    rows.sort(key=lambda r: int(r["satisfaction corrigée"].rstrip("%")))
    aspects_df = pd.DataFrame(rows) if rows else pd.DataFrame(
        [{"aspect": "aucun aspect détecté", "mentions": 0,
          "satisfaction corrigée": "—", "brut": "—"}]
    )
    return summary, dist, aspects_df


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
with gr.Blocks(title="Analyse de satisfaction client (FR)") as demo:
    gr.Markdown(
        "# Analyse de satisfaction client — français\n"
        "Estime la **part de clients satisfaits** à partir de leurs retours, "
        "avec un intervalle de confiance, et identifie **les sujets** de "
        "mécontentement.\n\n"
        "*Premier chargement après une mise en veille : 20–40 s.*"
    )
    if LOAD_ERROR:
        gr.Markdown(f"> ⚠️ **Modèle non chargé.** `{LOAD_ERROR}`")

    with gr.Tab("Analyse groupée (CSV)"):
        gr.Markdown(
            "Importez un CSV contenant une colonne `text` "
            "(ou `texte`, `commentaire`, `review`, `avis`). "
            f"Maximum {MAX_ROWS:,} lignes."
        )
        f_in = gr.File(label="Fichier CSV", file_types=[".csv"])
        btn = gr.Button("Analyser", variant="primary")
        out_md = gr.Markdown()
        out_dist = gr.Dataframe(label="Répartition")
        out_asp = gr.Dataframe(label="Satisfaction par sujet")
        btn.click(analyse_batch, f_in, [out_md, out_dist, out_asp])

    with gr.Tab("Commentaire unique"):
        t_in = gr.Textbox(label="Commentaire", lines=4,
                          placeholder="Collez un avis client en français…")
        t_btn = gr.Button("Analyser", variant="primary")
        t_scores = gr.Label(label="Probabilités calibrées")
        t_notes = gr.Markdown()
        t_btn.click(analyse_one, t_in, [t_scores, t_notes])
        gr.Examples(
            [
                "Un très beau film, touchant, porté par un duo d'acteurs remarquable.",
                "Le jeu des acteurs est excellent, mais le scénario est creux.",
                "Je l'ai vu hier soir, je ne sais pas encore quoi en penser.",
                "Scénario incohérent, dialogues ridicules. Une perte de temps.",
            ],
            t_in,
        )

    gr.Markdown(
        "---\n"
        "**Limites.** Entraîné sur des critiques de films (Allociné) : le "
        "transfert vers d'autres domaines est mesuré et documenté dans la "
        "fiche du modèle. Les données d'entraînement ne contiennent **pas** "
        "d'avis neutres, d'où la bande d'abstention. L'estimation agrégée est "
        "fiable ; **les prédictions individuelles ne le sont pas** et ne "
        "doivent pas servir à décider du traitement d'un client."
    )

if __name__ == "__main__":
    demo.launch()
