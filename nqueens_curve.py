import subprocess
import time
import matplotlib.pyplot as plt

def run_simplexe_nqueens(n_start, n_end):
    n_values = []
    execution_times = []

    print("Début des mesures de temps d'exécution...")

    for n in range(n_start, n_end + 1):
        file_name = f"instances/{n}queens.lp"
        program_path = "./target/release/simplexe"
        command = [program_path, file_name]
        
        print(f"  -> Exécution de '{' '.join(command)}'...", end="", flush=True)

        try:
            start_time = time.time()
            timout = False
            try:
                subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
            except subprocess.TimeoutExpired:
                print(" DÉLAI D'ATTENTE DÉPASSÉ (60 secondes).")
                timout = True
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            n_values.append(n)
            if timout:
                execution_times.append(60.0)
            else:
                execution_times.append(elapsed_time)
            
            print(f" terminé en {elapsed_time:.4f} secondes.")

        except FileNotFoundError:
            print(f" ERREUR : Le programme ou le fichier {command[0]} ou {command[1]} n'a pas été trouvé.")
            return None, None
        except subprocess.CalledProcessError as e:
            print(f" ERREUR : L'exécution a échoué avec le code de retour {e.returncode}.")
            print(f"   Sortie standard : {e.stdout}")
            print(f"   Sortie d'erreur : {e.stderr}")
            return None, None
        
    print("Mesures terminées.")
    return n_values, execution_times

def plot_performance_curve(n_values, times, nend):
    plt.style.use('ggplot') 
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, times, marker='o', linestyle='-', color='b')
    plt.title(f"Temps d'exécution du branch and bound pour le problème des n-reines (8 ≤ n ≤ {nend})")
    plt.xlabel("Nombre de Reines (n)")
    plt.ylabel("Temps d'exécution (secondes)")
    
    for i, txt in enumerate(times):
        plt.annotate(f"{txt:.2f}s", (n_values[i], times[i]), textcoords="offset points", xytext=(0,10), ha='center')
        
    plt.grid(True)
    plt.xticks(n_values)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    n_start = 8
    n_end = 30
    
    n_values, times = run_simplexe_nqueens(n_start, n_end)

    if n_values and times:
        plot_performance_curve(n_values, times, n_end)