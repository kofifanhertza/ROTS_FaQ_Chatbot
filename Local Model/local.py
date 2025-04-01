import subprocess

while True:
    prompt = input()

    if prompt.lower() in ["exit", "quit"]:
        break


    subprocess.run(["ollama", "run", "deepseek-r1:1.5b", f"{prompt}"]) 
    

