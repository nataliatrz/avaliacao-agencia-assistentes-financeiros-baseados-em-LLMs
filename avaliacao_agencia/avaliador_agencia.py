import os
import sys
import glob
import argparse
import re
import json
import time
import traceback
from typing import List

import numpy as np
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Colunas 
TEXT_COLS = ["text_pt", "text", "fala"]          # fala do ASSISTENTE
ROLE_COLS = ["role", "interlocutor"]            # papel: Cliente/Assistente
TURN_COLS = ["turn_id", "turno"]                # número do turno
DIALOG_ID_COL = "dialog_id"
PREV_CLIENT_COL = "prev_client_pt"              # inferido: fala do cliente no turno anterior
STD_TURN_ID_COL = "turn_id"                     # nome padronizado para saída

def pick_first(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

# Gemini
def setup_gemini(model_name: str):
    """Inicializa Gemini; tenta forçar JSON (se suportado pelo SDK)."""
    import google.generativeai as genai
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Variável de ambiente GOOGLE_API_KEY não encontrada.")
    genai.configure(api_key=api_key)
    generation_config = {
        "response_mime_type": "application/json",
        "temperature": 0.2,
    }
    try:
        
        generation_config["response_schema"] = {
            "type": "object",
            "properties": {
                "I": {"type": "integer"},
                "M": {"type": "integer"},
                "SE": {"type": "integer"},
                "SR": {"type": "integer"},
                "agency_overall": {"type": "integer"},
                "rationale_I": {"type": "string"},
                "rationale_M": {"type": "string"},
                "rationale_SE": {"type": "string"},
                "rationale_SR": {"type": "string"},
                "rationale_overall": {"type": "string"}
            },
            "required": ["I","M","SE","SR","agency_overall"]
        }
    except Exception:
        pass
    try:
        model = genai.GenerativeModel(model_name, generation_config=generation_config)
    except Exception:
        model = genai.GenerativeModel(model_name)
    return model

def load_prompt_template(path: str) -> str:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    # fallback
    return (
        "Você é um Avaliador de Agência. Avalie apenas a FALA DO ASSISTENTE no turno atual.\n"
        "Dimensões (0=ausente, 1=moderado, 2=forte): I, M, SE, SR.\n"
        "Responda SOMENTE com um único objeto JSON válido com as chaves:\n"
        '{"I":0|1|2,"M":0|1|2,"SE":0|1|2,"SR":0|1|2,"agency_overall":0|1|2,'
        '"rationale_I":"...", "rationale_M":"...", "rationale_SE":"...", '
        '"rationale_SR":"...", "rationale_overall":"..."}\n'
        "---\nMetadados (não responder):\n- dialog_id: {dialog_id}\n- turn_id: {turn_id}\n\n"
        "Contexto do Cliente:\n{user_ctx}\n\nFala do Assistente:\n{asst_text}\n"
    )

def build_prompt(template: str, dialog_id: str, turn_id: str, user_ctx: str, asst_text: str) -> str:
    """
    Substituição segura apenas dos 4 placeholders esperados, sem interpretar
    outras chaves { } do template (como as do JSON de exemplo).
    """
    s = template
    s = s.replace("{dialog_id}", dialog_id or "")
    s = s.replace("{turn_id}", turn_id or "")
    s = s.replace("{user_ctx}", user_ctx or "")
    s = s.replace("{asst_text}", asst_text or "")
    return s


# Parser JSON
def extract_json(text: str) -> dict:
   
    def norm(s: str) -> str:
        return s.replace("“", '"').replace("”", '"').replace("’", "'")

    s = (text or "").strip()
    if not s:
        return {}

    # 1) bloco ```json ... ```
    m = re.search(r"```json\s*(\{.*?\})\s*```", s, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            obj = json.loads(norm(m.group(1)))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # 2) recorte simples (do primeiro { ao último })
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        chunk = s[first:last+1]
        try:
            obj = json.loads(norm(chunk))
            if isinstance(obj, dict):
                return obj
        except Exception:
            
            for k in range(last, first, -1):
                try:
                    obj = json.loads(norm(s[first:k]))
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    continue

    # 3) texto inteiro
    try:
        obj = json.loads(norm(s))
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    return {}

# Chamada Gemini
class TransientError(Exception):
    pass

@retry(stop=stop_after_attempt(5),
       wait=wait_exponential(multiplier=1, min=2, max=20),
       retry=retry_if_exception_type(TransientError))
def call_gemini(model, prompt: str) -> dict:
    try:
        resp = model.generate_content(prompt)
        raw = getattr(resp, "text", "") or ""
        if not raw.strip():
            raise TransientError("Resposta vazia do modelo.")
        parsed = extract_json(raw)
        if not isinstance(parsed, dict):
            parsed = {}
        parsed["__raw"] = raw
        return parsed
    except Exception as e:
        raise TransientError(str(e))


def score_file(model, template: str, path: str, rate_limit_rps: float, use_mock: bool, debug: bool) -> pd.DataFrame:
    def safe_print_err(msg):
        print(msg, file=sys.stderr)

    try:
        df = pd.read_csv(path, sep="\t", dtype=str)
    except Exception as e:
        
        return pd.DataFrame([{
            DIALOG_ID_COL: os.path.splitext(os.path.basename(path))[0],
            STD_TURN_ID_COL: "",
            "I": np.nan, "M": np.nan, "SE": np.nan, "SR": np.nan, "agency_overall": np.nan,
            "rationale_I": "", "rationale_M": "", "rationale_SE": "", "rationale_SR": "", "rationale_overall": "",
            "asst_text": "", "user_ctx": "",
            "__source": os.path.basename(path),
            "__raw_response": "",
            "__error": f"Falha ao ler TSV: {e}",
        }])

    text_col = pick_first(df.columns, TEXT_COLS)
    role_col = pick_first(df.columns, ROLE_COLS)
    turn_col = pick_first(df.columns, TURN_COLS)

    # Normaliza dialog_id
    if DIALOG_ID_COL not in df.columns:
        df[DIALOG_ID_COL] = os.path.splitext(os.path.basename(path))[0]

    # Se faltar alguma coluna essencial, gera linhas de erro
    missing = []
    if text_col is None: missing.append(f"texto ({TEXT_COLS})")
    if turn_col is None: missing.append(f"turno ({TURN_COLS})")
    if missing:
        return pd.DataFrame([{
            DIALOG_ID_COL: df[DIALOG_ID_COL].iloc[0] if len(df) else os.path.splitext(os.path.basename(path))[0],
            STD_TURN_ID_COL: "",
            "I": np.nan, "M": np.nan, "SE": np.nan, "SR": np.nan, "agency_overall": np.nan,
            "rationale_I": "", "rationale_M": "", "rationale_SE": "", "rationale_SR": "", "rationale_overall": "",
            "asst_text": "", "user_ctx": "",
            "__source": os.path.basename(path),
            "__raw_response": "",
            "__error": f"Colunas ausentes: {', '.join(missing)}",
        }])

    # Ordena
    try:
        df[turn_col] = pd.to_numeric(df[turn_col], errors="coerce")
        df = df.sort_values([DIALOG_ID_COL, turn_col, text_col])
    except Exception as e:
        safe_print_err(f"[WARN] Ordenação falhou em {path}: {e}")

    # Constrói mapa do turno anterior do cliente
    prev_map = None
    try:
        full = pd.read_csv(path, sep="\t", dtype=str)
        if DIALOG_ID_COL not in full.columns:
            full[DIALOG_ID_COL] = os.path.splitext(os.path.basename(path))[0]
        full_text = pick_first(full.columns, TEXT_COLS)
        full_role = pick_first(full.columns, ROLE_COLS)
        full_turn = pick_first(full.columns, TURN_COLS)
        if full_text and full_role and full_turn:
            full[full_turn] = pd.to_numeric(full[full_turn], errors="coerce")
            tmp = full.copy()
            tmp["_next_turn"] = tmp[full_turn] + 1
            prev = tmp[tmp[full_role].str.lower().str.strip().eq("cliente")][[DIALOG_ID_COL, "_next_turn", full_text]]
            prev = prev.rename(columns={"_next_turn": turn_col, full_text: PREV_CLIENT_COL})
            prev_map = prev
    except Exception as e:
        safe_print_err(f"[WARN] prev_client_pt falhou em {path}: {e}")

    # Filtra assistente
    try:
        if role_col is not None:
            df = df[df[role_col].str.lower().str.strip().isin(["assistente", "assistant"])].copy()
    except Exception as e:
        safe_print_err(f"[WARN] Filtro assistente falhou em {path}: {e}")

    # Mescla contexto
    try:
        if prev_map is not None:
            df = df.merge(prev_map, on=[DIALOG_ID_COL, turn_col], how="left")
        if PREV_CLIENT_COL not in df.columns:
            df[PREV_CLIENT_COL] = ""
    except Exception as e:
        safe_print_err(f"[WARN] Merge contexto falhou em {path}: {e}")
        df[PREV_CLIENT_COL] = ""

    # Se não há linhas do assistente, ainda assim gera 1 linha informando
    if df.empty:
        return pd.DataFrame([{
            DIALOG_ID_COL: os.path.splitext(os.path.basename(path))[0],
            STD_TURN_ID_COL: "",
            "I": np.nan, "M": np.nan, "SE": np.nan, "SR": np.nan, "agency_overall": np.nan,
            "rationale_I": "", "rationale_M": "", "rationale_SE": "", "rationale_SR": "", "rationale_overall": "",
            "asst_text": "",
            "user_ctx": "",
            "__source": os.path.basename(path),
            "__raw_response": "",
            "__error": "Nenhum turno do Assistente encontrado",
        }])

    # Processa cada turno
    out_rows = []
    last_ts = 0.0
    for _, row in df.iterrows():
        try:
            dialog_id = str(row.get(DIALOG_ID_COL, ""))
            turn_id_value = row.get(turn_col, "")
            asst_text = str(row.get(text_col, ""))
            user_ctx = str(row.get(PREV_CLIENT_COL, ""))

            template = globals().get("_GLOBAL_TEMPLATE_CACHE")
            if template is None:
                template = globals()["_GLOBAL_TEMPLATE_CACHE"] = load_prompt_template(globals().get("_PROMPT_PATH", ""))

            prompt = build_prompt(template, dialog_id, str(turn_id_value), user_ctx, asst_text)

            # rate limit simples
            min_interval = 1.0 / rate_limit_rps if rate_limit_rps > 0 else 0.0
            elapsed = time.time() - last_ts
            if elapsed < min_interval:
                time.sleep(max(0.0, min_interval - elapsed))

            # chama modelo ou mock
            result = {}
            err_msg = ""
            raw_resp = ""
            try:
                result = call_gemini(globals().get("_GLOBAL_MODEL"), prompt)
                raw_resp = result.get("__raw", "")
            except Exception as e:
                err_msg = str(e)

            def to_int_safe(v):
                try:
                    return int(v)
                except Exception:
                    return np.nan

            out = {
                DIALOG_ID_COL: dialog_id,
                STD_TURN_ID_COL: turn_id_value,
                "I": to_int_safe((result or {}).get("I")),
                "M": to_int_safe((result or {}).get("M")),
                "SE": to_int_safe((result or {}).get("SE")),
                "SR": to_int_safe((result or {}).get("SR")),
                "agency_overall": to_int_safe((result or {}).get("agency_overall")),
                "rationale_I": (result or {}).get("rationale_I", ""),
                "rationale_M": (result or {}).get("rationale_M", ""),
                "rationale_SE": (result or {}).get("rationale_SE", ""),
                "rationale_SR": (result or {}).get("rationale_SR", ""),
                "rationale_overall": (result or {}).get("rationale_overall", ""),
                "asst_text": asst_text,
                "user_ctx": user_ctx,
                "__source": os.path.basename(path),
                "__raw_response": raw_resp,
                "__error": err_msg,
            }
            # preserva metadados originais (sem duplicar texto)
            for c in df.columns:
                if c not in out and c != text_col:
                    out[c] = row.get(c, "")

            out_rows.append(out)
            last_ts = time.time()

        except Exception as inner_e:
            if debug:
                traceback.print_exc()
            out_rows.append({
                DIALOG_ID_COL: os.path.splitext(os.path.basename(path))[0],
                STD_TURN_ID_COL: row.get(turn_col, ""),
                "I": np.nan, "M": np.nan, "SE": np.nan, "SR": np.nan, "agency_overall": np.nan,
                "rationale_I": "", "rationale_M": "", "rationale_SE": "", "rationale_SR": "", "rationale_overall": "",
                "asst_text": str(row.get(text_col, "")),
                "user_ctx": str(row.get(PREV_CLIENT_COL, "")),
                "__source": os.path.basename(path),
                "__raw_response": "",
                "__error": f"Falha na linha: {inner_e}",
            })

    scored = pd.DataFrame(out_rows)
    # garantir colunas numéricas e média
    for col in ["I", "M", "SE", "SR", "agency_overall"]:
        if col in scored.columns:
            scored[col] = pd.to_numeric(scored[col], errors="coerce")
        else:
            scored[col] = np.nan
    scored["agency_global"] = scored[["I", "M", "SE", "SR"]].mean(axis=1, skipna=True)
    return scored

def summarize_dialog(scored: pd.DataFrame) -> pd.DataFrame:
    if DIALOG_ID_COL not in scored.columns or scored.empty:
        return pd.DataFrame()
    return scored.groupby(DIALOG_ID_COL).agg(
        n_turns=("agency_global", "count"),
        I_mean=("I", "mean"),
        M_mean=("M", "mean"),
        SE_mean=("SE", "mean"),
        SR_mean=("SR", "mean"),
        agency_global_mean=("agency_global", "mean"),
        agency_overall_mean=("agency_overall", "mean"),
    ).reset_index()


def main():
    parser = argparse.ArgumentParser(description="Avalia agência (I/M/SE/SR) por turno.")
    parser.add_argument("--inputs", required=True, help="Glob: 'dados/*.tsv' ou 'dados/**/*.tsv' (use --recursive).")
    parser.add_argument("--outdir", required=True, help="Diretório de saída.")
    parser.add_argument("--model", default="gemini-1.5-pro", help="Modelo Gemini.")
    parser.add_argument("--prompt", default="prompt_template_pt.txt", help="Arquivo de prompt (opcional).")
    parser.add_argument("--rps", type=float, default=0.5, help="Req/seg (0.5 = 1 req a cada 2s).")
    parser.add_argument("--recursive", action="store_true", help="Habilita glob recursivo (**).")
    parser.add_argument("--mock", action="store_true", help="Não chama API; gera notas simuladas (debug).")
    parser.add_argument("--debug", action="store_true", help="Mostra traceback de erros de linha.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Guardar caminhos globais para uso interno
    globals()["_PROMPT_PATH"] = args.prompt
    globals()["_GLOBAL_TEMPLATE_CACHE"] = load_prompt_template(args.prompt)

    
    if not args.mock:
        try:
            globals()["_GLOBAL_MODEL"] = setup_gemini(args.model)
        except Exception as e:
            print(f"[WARN] Falha ao iniciar Gemini: {e}. Continuando em modo MOCK.", file=sys.stderr)
            args.mock = True

    # Expandir paths
    paths = sorted(glob.glob(args.inputs, recursive=args.recursive))
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        print("Nenhum arquivo encontrado para --inputs.", file=sys.stderr)
        # Ainda assim cria arquivos vazios para não falhar o pipeline
        empty_turn = pd.DataFrame(columns=[
            DIALOG_ID_COL, STD_TURN_ID_COL, "I","M","SE","SR","agency_overall",
            "rationale_I","rationale_M","rationale_SE","rationale_SR","rationale_overall",
            "asst_text","user_ctx","__source","__raw_response","__error","agency_global"
        ])
        empty_dialog = pd.DataFrame(columns=[
            DIALOG_ID_COL, "n_turns","I_mean","M_mean","SE_mean","SR_mean","agency_global_mean","agency_overall_mean"
        ])
        empty_turn.to_csv(os.path.join(args.outdir, "per_turn_llm.csv"), index=False, encoding="utf-8")
        empty_dialog.to_csv(os.path.join(args.outdir, "per_dialog_llm.csv"), index=False, encoding="utf-8")
        sys.exit(0)

    frames: List[pd.DataFrame] = []
    for path in paths:
        print(f"[INFO] Processando: {path}")
        try:
            scored = score_file(
                model=globals().get("_GLOBAL_MODEL"),
                template=globals().get("_GLOBAL_TEMPLATE_CACHE"),
                path=path,
                rate_limit_rps=args.rps,
                use_mock=args.mock,
                debug=args.debug
            )
            frames.append(scored)
        except Exception as e:
            if args.debug:
                traceback.print_exc()
            # Gera uma linha de erro para este arquivo
            frames.append(pd.DataFrame([{
                DIALOG_ID_COL: os.path.splitext(os.path.basename(path))[0],
                STD_TURN_ID_COL: "",
                "I": np.nan, "M": np.nan, "SE": np.nan, "SR": np.nan, "agency_overall": np.nan,
                "rationale_I": "", "rationale_M": "", "rationale_SE": "", "rationale_SR": "", "rationale_overall": "",
                "asst_text": "", "user_ctx": "",
                "__source": os.path.basename(path),
                "__raw_response": "",
                "__error": f"Falha geral ao processar arquivo: {e}",
                "agency_global": np.nan
            }]))

    all_scored = pd.concat(frames, ignore_index=True)

    # Salvar per_turn sempre
    per_turn_path = os.path.join(args.outdir, "per_turn_llm.csv")
    all_scored.to_csv(per_turn_path, index=False, encoding="utf-8-sig")
    print(f"[OK] salvo: {per_turn_path}")

    # Resumo por diálogo 
    try:
        per_dialog = summarize_dialog(all_scored)
    except Exception as e:
        if args.debug:
            traceback.print_exc()
        per_dialog = pd.DataFrame(columns=[
            DIALOG_ID_COL, "n_turns","I_mean","M_mean","SE_mean","SR_mean","agency_global_mean","agency_overall_mean"
        ])

    per_dialog_path = os.path.join(args.outdir, "per_dialog_llm.csv")
    per_dialog.to_csv(per_dialog_path, index=False, encoding="utf-8-sig")
    print(f"[OK] salvo: {per_dialog_path}")

if __name__ == "__main__":
    main()
