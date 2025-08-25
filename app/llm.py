import subprocess

def ask_ollama(prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3"],
            input=prompt.encode("utf-8"),
            capture_output=True
        )
        return result.stdout.decode("utf-8").strip()
    except Exception as e:
        return "Ошибка LLM: " + str(e)
