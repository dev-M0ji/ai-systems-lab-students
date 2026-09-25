"""Salida estructurada: pedir JSON al LLM y validarlo con un esquema.

El LLM produce texto. Convertir ese texto en un objeto confiable es
responsabilidad del software tradicional (parseo + validación).

Uso:
    uv run python labs/01-llm/structured.py "¿Qué es el mecanismo de atención?"
"""

import json
import sys
from typing import Literal

from pydantic import BaseModel, ValidationError

from config import load_settings
from llm_client import LLMClient, LLMError
from prompts import ANALYSIS_PROMPT


class QuestionAnalysis(BaseModel):
    tema: str
    dificultad: Literal["basica", "intermedia", "avanzada"]
    requiere_documentos_del_curso: bool
    respuesta_corta: str


def analyze_question(client: LLMClient, question: str) -> QuestionAnalysis:
    messages = [
        {"role": "system", "content": ANALYSIS_PROMPT},
        {"role": "user", "content": question},
    ]
    response = client.chat(messages, temperature=0, json_mode=True)
    print(f"Texto crudo del LLM:\n{response.text}\n")

    max_retries = 2
    for attempt in range(max_retries + 1):
        response = client.chat(
            messages,
            temperature=0,
            max_tokens=10 if attempt == 0 else None,
            json_mode=attempt > 0,
        )

        print(f"Intento {attempt + 1}: finish_reason={response.finish_reason}")
        print(response.text)

    for attempt in range(max_retries):
        response = client.chat(messages, temperature=0, json_mode=True)
        try:
            data = json.loads(response.text)
            return QuestionAnalysis.model_validate(data)
        except json.JSONDecodeError as exc:
            error=(f"El LLM no devolvió JSON válido: {exc}")
        except ValidationError as exc:
            error=(f"El JSON no cumple el esquema:\n{exc}")
        if attempt == max_retries:
            raise ValueError("El LLM no devolvió un JSON válido después de varios intentos. \n" + error)

        messages.extend(
            [
                {"role": "assistant", "content": response.text},
                {"role": "user", "content": ("El JSON que me diste no es válido."
                    "Por favor, intenta de nuevo y asegúrate de que sea un objeto JSON válido que cumpla con el esquema.")},
            ]   
        )
    raise ValueError("El bucle acabo de manera inesperada")

    # TODO 6: convierte el texto en un QuestionAnalysis en dos pasos separados:
    #   1. json.loads(...)                     → ¿es JSON válido?
    #   2. QuestionAnalysis.model_validate(...) → ¿cumple el esquema?


def main() -> None:
    question = " ".join(sys.argv[1:]) or input("Pregunta: ")
    try:
        client = LLMClient(load_settings())
    except ValueError as exc:
        print(f"[configuración] {exc}")
        return

    try:
        analysis = analyze_question(client, question)
    except LLMError as exc:
        print(f"[error del proveedor] {exc}")
        return
    except json.JSONDecodeError as exc:
        print(f"[error] El LLM no devolvió JSON válido: {exc}")
        return
    except ValidationError as exc:
        print(f"[error] El JSON no cumple el esquema:\n{exc}")
        return

    print("Objeto validado:")
    print(analysis.model_dump_json(indent=2))

    # Ya es un objeto de Python: el programa puede tomar decisiones con él.
    if analysis.requiere_documentos_del_curso:
        print("\n→ Esta pregunta necesitaría documentos del curso para responderse bien.")


if __name__ == "__main__":
    main()
