import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


combined_data = pd.DataFrame()
folder_path = "Comparison_Results"

for filename in os.listdir(folder_path):
    if filename.startswith("Accuracy_") and filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        combined_data = pd.concat([combined_data, df], ignore_index=True)
        os.remove(file_path)


current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
dump_comparison_path = os.path.join(folder_path, current_datetime)
if not os.path.exists(dump_comparison_path): os.makedirs(dump_comparison_path)
output_file_name = "Combined_results.csv"
csv_file_path = os.path.join(dump_comparison_path, output_file_name)
combined_data.to_csv(csv_file_path, index=False)

# Load the CSV data
df = pd.read_csv(csv_file_path)

# Sort the DataFrame by MethodName for both Train and Test Accuracy
df_train = df.sort_values(by='Final Train Accuracy', ascending=False)
df_test = df.sort_values(by='Final Test Accuracy', ascending=False)

fig, axs = plt.subplots(1,2, figsize=(25,15))

# Plotting Train Accuracy
axs[0].bar(df_train['MethodName'], df_train['Final Train Accuracy'], color='blue')
axs[0].set_title('Train Accuracy')
axs[0].set_xlabel('Method Name')
axs[0].set_ylabel('Accuracy')
axs[0].set_xticklabels(df_train['MethodName'], rotation=90)
axs[0].grid(True, linestyle='--', alpha=0.9)

# Annotating bars with values
for i, v in enumerate(df_train['Final Train Accuracy']):
    axs[0].text(i, v, str(round(v, 2)), ha='center', va='bottom')

# Plotting Test Accuracy
axs[1].bar(df_test['MethodName'], df_test['Final Test Accuracy'], color='orange')
axs[1].set_title('Test Accuracy')
axs[1].set_xlabel('Method Name')
axs[1].set_ylabel('Accuracy')
axs[1].set_xticklabels(df_test['MethodName'], rotation=90)
axs[1].grid(True, linestyle='--', alpha=0.9)

# Annotating bars with values
for i, v in enumerate(df_test['Final Test Accuracy']):
    axs[1].text(i, v, str(round(v, 2)), ha='center', va='bottom')

# Adjust layout and show the plot
plt.xticks(fontsize=16)
plt.tight_layout()
plot_path = os.path.join(dump_comparison_path, 'Accuracy_Comparison.png')
plt.savefig(plot_path)
#plt.show()

"""
# Plotting Train Accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(df_train['MethodName'], df_train['Final Train Accuracy'], color='blue')
plt.title('Train Accuracy')
plt.xlabel('MethodName')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
for index, value in enumerate(df_train['Final Train Accuracy']):
    plt.text(index, value, str(round(value, 2)), ha='center', va='bottom')

# Plotting Test Accuracy
plt.subplot(1, 2, 2)
plt.bar(df_test['MethodName'], df_test['Final Test Accuracy'], color='orange')
plt.title('Test Accuracy')
plt.xlabel('MethodName')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
for index, value in enumerate(df_test['Final Test Accuracy']):
    plt.text(index, value, str(round(value, 2)), ha='center', va='bottom')

# Show the plots
plt.tight_layout()
plot_path = os.path.join(dump_comparison_path, 'Accuracy_Comparison.png')
plt.savefig(plot_path)
plt.show()
"""