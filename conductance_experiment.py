import subprocess

subprocess_commands = [
    ['python3', 'experiments_ssl.py', 'ant', '3', '4', '0.3', '0.05'], 
    ['python3', 'experiments_ssl.py', 'ant', '3', '4', '0.3', '0.10'], 
    ['python3', 'experiments_ssl.py', 'ant', '3', '4', '0.3', '0.15'],
    ['python3', 'experiments_ssl.py', 'ant', '3', '4', '0.3', '0.20'],
    ['python3', 'experiments_ssl.py', 'ant', '3', '4', '0.3', '0.25'],
    ['python3', 'experiments_ssl.py', 'ant', '3', '4', '0.3', '0.30'],
    ['python3', 'experiments_ssl.py', 'ant', '3', '4', '0.3', '0.35'],
    ['python3', 'experiments_ssl.py', 'ant', '3', '4', '0.3', '0.40'],
    ['python3', 'experiments_ssl.py', 'ant', '3', '4', '0.3', '0.45'],
    ['python3', 'experiments_ssl.py', 'ant', '3', '4', '0.3', '0.50'],
    ['python3', 'experiments_ssl.py', 'ant', '3', '4', '0.3', '0.55'],
    ['python3', 'experiments_ssl.py', 'ant', '3', '4', '0.3', '0.60'],
    ['python3', 'experiments_ssl.py', 'ant', '3', '4', '0.3', '0.65'],
    ['python3', 'experiments_ssl.py', 'ant', '3', '4', '0.3', '0.70'],
    ['python3', 'experiments_ssl.py', 'ant', '3', '4', '0.3', '0.75'],
    ['python3', 'experiments_ssl.py', 'ant', '3', '4', '0.3', '0.80'],
    ['python3', 'experiments_ssl.py', 'ant', '3', '4', '0.3', '0.85'],
    ['python3', 'experiments_ssl.py', 'ant', '3', '4', '0.3', '0.90'],
    ['python3', 'experiments_ssl.py', 'ant', '3', '4', '0.3', '0.95']
]

for command in subprocess_commands:
    subprocess.run(command)
