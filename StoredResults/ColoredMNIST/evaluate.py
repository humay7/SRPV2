import time
import subprocess
import os
from datetime import datetime

# Start the timer to measure the script's total execution time
start_time = time.time()

# Define file names
Base_Code_FileName = "/content/SRPV2/StoredResults/ColoredMNIST/base_code.py"
Visualization_FileName = "/content/SRPV2/StoredResults/ColoredMNIST/Visualization.py"
Comparison_FileName = "/content/SRPV2/StoredResults/ColoredMNIST/Comparison.py"

# Run the base code script
try:
    subprocess.run(["python", Base_Code_FileName])
except Exception as e:
    print(f"Error running {Base_Code_FileName}: {str(e)}")

# Optionally run the Visualization.py and Comparison.py (remove if not needed)
try:
    subprocess.run(["python", Visualization_FileName])
except Exception as e:
    print(f"Error running {Visualization_FileName}: {str(e)}")

try:
    subprocess.run(["python", Comparison_FileName])
except Exception as e:
    print(f"Error running {Comparison_FileName}: {str(e)}")

# End the timer
end_time = time.time()

# Calculate the total time taken
total_time = end_time - start_time

# Log the start, end, and total time
current_datetime = time.strftime("%Y-%m-%d %H:%M:%S")
start_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
end_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))

# Write computation time to a file
if not os.path.exists("Computation_Time.txt"):
    with open("Computation_Time.txt", "w") as file:
        file.write("StartTime | EndTime | CurrentDate | Total Time (seconds)\n")

with open("Computation_Time.txt", "a") as file:
    file.write(f"{start_datetime} | {end_datetime} | {current_datetime} | {total_time}\n")

print("Computation time logged successfully.")
