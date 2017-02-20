import subprocess
import sys

for ens in range(int(sys.argv[1])):
    subprocess.check_call([sys.executable, "-m", "mlp.train_one", str(ens)])
