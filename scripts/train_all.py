import subprocess
import os

def run_script(script_name):
    try:
        print(f"Running {script_name}...")
        result = subprocess.run(['python', script_name], check=True, text=True, capture_output=True)
        print(f"{script_name} finished with output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running {script_name}:\n{e.stderr}")

if __name__ == "__main__":
    # 定义要运行的脚本列表
    scripts = ['train_tgn.py', 'train_tgat.py', 'train_tgn_sim.py', 'train_tgat_sim.py']
    
    # 确保当前工作目录是脚本所在的目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 遍历并运行每个脚本
    for script in scripts:
        run_script(script)
