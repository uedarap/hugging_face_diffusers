"""Exibe exemplos de comandos úteis para estudo.

Este arquivo não baixa modelos nem executa geração automaticamente.
Ele serve como referência rápida para lembrar combinações de comandos.
"""

EXAMPLES = [
    'python check_environment.py',
    'python generate_image.py --prompt "a futuristic city at sunrise" --profile cpu_safe',
    'python generate_image.py --prompt "a cinematic robot in the rain" --device cuda --profile gtx1080',
    'python generate_video.py --prompt "a small boat crossing a misty lake" --profile cpu_safe',
    'python generate_video.py --prompt "clouds moving over mountains at sunset" --profile gtx1080 --device cuda --frames 12 --steps 15',
]


def main() -> None:
    print("Exemplos de uso da POC:\n")
    for command in EXAMPLES:
        print(f"- {command}")


if __name__ == "__main__":
    main()
