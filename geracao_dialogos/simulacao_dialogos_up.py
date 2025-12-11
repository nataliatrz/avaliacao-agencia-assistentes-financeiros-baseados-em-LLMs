import os
import json
from datetime import datetime
from glob import glob
from openai import OpenAI



client = OpenAI(api_key="API_KEY")


# --------------------------------
# CONFIGURAÇÕES
# --------------------------------

PERSONALIDADES = {
    "default": "Estilo padrão: claro, neutro e equilibrado.",
    "friendly": (
        "Estilo amigável: tom caloroso, acolhedor, levemente informal, "
        "reflete as preocupações do investidor e explica com calma."
    ),
    "nerdy": (
        "Estilo nerd: entusiasmado com detalhes técnicos, gosta de explicar conceitos "
        "com profundidade, trazendo analogias e um clima de curiosidade intelectual."
    ),
    "quirky": (
        "Estilo diferentão/quirky: bem-humorado, criativo, metafórico."
    ),
    "efficient": (
        "Estilo eficiente: respostas diretas e concisas, indo rápido ao ponto."
    ),
}

PERFIS_INVESTIDOR = ["conservador", "moderado", "arrojado"]

MODELO_ASSISTENTE = "gpt-4.1-mini"
MODELO_INVESTIDOR = "gpt-4.1-mini"

OUTPUT_DIR = "dialogos_txt"


# --------------------------------
# FUNÇÕES DE APOIO
# --------------------------------

def gerar_resposta(modelo: str, mensagens: list[dict], temperature: float = 0.7) -> str:
    resp = client.chat.completions.create(
        model=modelo,
        messages=mensagens,
        temperature=temperature,
    )
    return resp.choices[0].message.content


def gerar_fala_investidor(modelo: str, perfil: str, fala_assistente: str) -> str:
    mensagens = [
        {
            "role": "system",
            "content": (
                f"Você é um investidor pessoa física com perfil {perfil}.\n"
                "Faça perguntas e expressões naturais, até 3 frases.\n"
                "Reaja à fala do assistente com dúvidas, pedidos de esclarecimento ou preocupações.\n"
                "Seja crível e natural."
            )
        },
        {
            "role": "user",
            "content": (
                "Reaja como investidor à fala do assistente abaixo:\n\n"
                f"\"{fala_assistente}\""
            )
        }
    ]

    return gerar_resposta(modelo, mensagens, temperature=0.8)


# --------------------------------
# SALVAR EM TXT
# --------------------------------

def salvar_dialogo_txt(dialog_id: str, turns: list[dict], output_dir: str = OUTPUT_DIR) -> str:
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{dialog_id}.txt"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("turno\tinterlocutor\tfala\n")
        for t in turns:
            turno = t["turn_index"]
            interlocutor = t["speaker"]
            fala = t["text"].replace("\n", " ")  # evita quebras no meio da tabela
            f.write(f"{turno}\t{interlocutor}\t{fala}\n")

    return filepath


# --------------------------------
# SIMULAÇÃO
# --------------------------------

def simular_conversa(estilo: str, estilo_desc: str, perfil_inv: str, k_turnos: int = 5):
    print(f"\n=== ESTILO: {estilo} | PERFIL: {perfil_inv} ===\n")

    # Prompt do assistente
    sistema_assistente = {
        "role": "system",
        "content": (
            "Você é um assistente de investimentos no Brasil.\n"
            f"{estilo_desc}\n"
            "Explique riscos, diversificação e horizonte de investimento.\n"
            "Nunca dê recomendação personalizada. Seja claro e responsável."
        )
    }

    conversa = [sistema_assistente]

    # Lista de turnos para salvar
    turns = []
    turn_index = 1

    # Fala inicial do investidor
    fala_inv = (
        "Oi, estou pensando em investir meu dinheiro, mas não sei por onde começar. "
        "Pode me orientar?"
    )
    conversa.append({"role": "user", "content": fala_inv})
    print(f"Investidor: {fala_inv}\n")

    turns.append({"turn_index": turn_index, "speaker": "investidor", "text": fala_inv})
    turn_index += 1

    # Resposta do assistente
    fala_assistente = gerar_resposta(MODELO_ASSISTENTE, conversa)
    conversa.append({"role": "assistant", "content": fala_assistente})
    print(f"Assistente ({estilo}): {fala_assistente}\n")

    turns.append({"turn_index": turn_index, "speaker": "assistente", "text": fala_assistente})
    turn_index += 1

    # Loop de turnos
    for _ in range(k_turnos):
        # Fala dinâmica do investidor
        fala_inv = gerar_fala_investidor(MODELO_INVESTIDOR, perfil_inv, fala_assistente)
        conversa.append({"role": "user", "content": fala_inv})
        print(f"Investidor: {fala_inv}\n")

        turns.append({"turn_index": turn_index, "speaker": "investidor", "text": fala_inv})
        turn_index += 1

        # Resposta do assistente
        fala_assistente = gerar_resposta(MODELO_ASSISTENTE, conversa)
        conversa.append({"role": "assistant", "content": fala_assistente})
        print(f"Assistente ({estilo}): {fala_assistente}\n")

        turns.append({"turn_index": turn_index, "speaker": "assistente", "text": fala_assistente})
        turn_index += 1

    # Identificador do diálogo
    dialog_id = f"{estilo}_{perfil_inv}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Salvar arquivo txt
    path = salvar_dialogo_txt(dialog_id, turns)
    print(f"➡ Arquivo gerado: {path}\n")

    return path


# --------------------------------
# MAIN
# --------------------------------

if __name__ == "__main__":
    k_turnos = 3  # número de turnos após a abertura

    for perfil in PERFIS_INVESTIDOR:
        for estilo, desc in PERSONALIDADES.items():
            simular_conversa(estilo, desc, perfil, k_turnos)
