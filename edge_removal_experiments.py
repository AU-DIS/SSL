import subprocess
import sys

dataset = sys.argv[1]
percentage_lower_bound = int(sys.argv[2])
percentage_upper_bound = int(sys.argv[3])
per = float(sys.argv[4])


subprocess_commands = []
for i in range(9, 10):
    subprocess_commands.append(['python3', 'experiments_ssl.py', dataset, str(percentage_lower_bound), str(percentage_upper_bound), str(per), '0.' + str(i)])

print(subprocess_commands)

for command in subprocess_commands:
    subprocess.run(command)
