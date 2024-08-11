
import numpy as np
import csv
import warnings
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pandas as pd
from shutil import move
import json
import sys
warnings.filterwarnings("ignore")


python_file_name = sys.argv[1]
print(python_file_name)
# Parameters initialization
with open('parameters.json', 'r') as f:
    params = json.load(f)

epochs = params['epochs']
batch_size = params['batch_size']

num_runs = params['num_runs']

df = pd.read_csv('resnet_results.csv')

# Sort the DataFrame by 'Run' and 'Epoch' columns
df_sorted = df.sort_values(by=['Run', 'Epoch'])

# Write the sorted DataFrame back to a CSV file
df_sorted.to_csv('resnet_results_sorted.csv', index=False)

def read_csv(filename):
    data = {'Run': [], 'Epoch': [], 'Train Loss': [], 'Train Accuracy': [], 'Test Accuracy': []}
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for key in data:
                data[key].append(float(row[key]))
    return data

folder_name = "Augumentation_Results"
if not os.path.exists(folder_name): os.makedirs(folder_name)
today_date = datetime.now().strftime("%Y-%m-%d")
folder_path = os.path.join(folder_name, today_date)
if not os.path.exists(folder_path): os.makedirs(folder_path)
timestamp = datetime.now().strftime("%H-%M-%S")
timestamp_folder = os.path.join(folder_path, f"{today_date}_{timestamp}")
#current_file_name = python_file_name
file_name_without_extension = python_file_name
csv_file_name = file_name_without_extension + '.csv'
script_folder = os.path.join(timestamp_folder, csv_file_name)
if not os.path.exists(script_folder):os.makedirs(script_folder)
sorted_csv_path = os.path.join(script_folder, 'resnet_results_sorted.csv')



# Read data from CSV
filename = 'resnet_results_sorted.csv'
data = read_csv(filename)



# Plot settings
plt.figure(figsize=(15, 10))

def plot_line_graph(x, y, label, ylabel, title):
    plt.plot(x, y, label=label)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.xticks(range(1, epochs + 1))



# Plot training loss
plt.subplot(1, 3, 1)
for run in range(1, num_runs+1):
    x = list(range(1, epochs + 1))
    y = data['Train Loss'][run * epochs - epochs:run * epochs]
    plot_line_graph(x, y, f'Run {run}', 'Training Loss', 'Training Loss')

# Plot training accuracy
plt.subplot(1,3, 2)
for run in range(1, num_runs+1):
    x = list(range(1, epochs + 1))
    y = data['Train Accuracy'][run * epochs - epochs:run * epochs]
    plot_line_graph(x, y, f'Run {run}', 'Training Accuracy', 'Training Accuracy')

# Plot test accuracy
plt.subplot(1,3,3)
for run in range(1, num_runs+1):
    x = list(range(1, epochs + 1))
    y = data['Test Accuracy'][run * epochs - epochs:run * epochs]
    plot_line_graph(x, y, f'Run {run}', 'Test Accuracy', 'Test Accuracy')


plt.tight_layout()
plot_path = os.path.join(script_folder, 'Accuracy_plot.png')
plt.savefig(plot_path)

df = pd.read_csv('resnet_results_sorted.csv')
df.drop(columns=['Run'])
average_df = df.groupby('Epoch').agg({'Train Loss':'mean','Train Accuracy': 'mean', 'Test Accuracy': 'mean'}).reset_index()
average_df.columns = ['Epoch','Train Loss Average', 'Train Accuracy Average', 'Test Accuracy Average']
average_df.to_csv('average_result.csv', index=False)
average_csv_path = os.path.join(script_folder, 'average_result.csv')
plt.figure(figsize=(10, 6))

# Plot lines
plt.plot(average_df['Epoch'], average_df['Train Accuracy Average'], label='Train Accuracy Average', marker='o')
plt.plot(average_df['Epoch'], average_df['Test Accuracy Average'], label='Test Accuracy Average', marker='o')

# Add dots and annotations for each epoch
epochs_list = list(np.arange(1, epochs+1))
for epoch in epochs_list:
    train_accuracy = average_df.loc[average_df['Epoch'] == epoch, 'Train Accuracy Average'].values[0]
    test_accuracy = average_df.loc[average_df['Epoch'] == epoch, 'Test Accuracy Average'].values[0]
    plt.scatter(epoch, train_accuracy, color='blue')
    plt.scatter(epoch, test_accuracy, color='orange')
    plt.text(epoch, train_accuracy, f'{train_accuracy:.2f}', ha='right', va='bottom', fontsize=8)
    plt.text(epoch, test_accuracy, f'{test_accuracy:.2f}', ha='right', va='bottom', fontsize=8)

# Add labels and legend
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy over Epochs')
plt.legend()
plt.grid(True)

# Show plot
plt.tight_layout()
plot_avg_path = os.path.join(script_folder, 'Average_plot.png')
plt.savefig(plot_avg_path)

comparison_folder_name = "Comparison_Results"
if not os.path.exists(comparison_folder_name): os.makedirs(comparison_folder_name)
comparison_file_name = 'Accuracy_'+str(file_name_without_extension)+ '.csv'
final_epoch_index = average_df['Epoch'].idxmax()
final_train_accuracy = average_df.loc[final_epoch_index, 'Train Accuracy Average']
final_test_accuracy = average_df.loc[final_epoch_index, 'Test Accuracy Average']
#pd.DataFrame({'Final Epoch Accuracy': [final_epoch_accuracy]}).to_csv(comparison_file_name, index=False)
final_accuracy_df = pd.DataFrame({'Final Epoch': [epochs],  
                                  'MethodName': [file_name_without_extension],
                                  'Final Train Accuracy': [final_train_accuracy],
                                  'Final Test Accuracy': [final_test_accuracy]})
dump_comparison_csv_path = os.path.join(comparison_folder_name, comparison_file_name)
final_accuracy_df.to_csv(dump_comparison_csv_path, index=False)

move('resnet_results_sorted.csv', sorted_csv_path)
move('average_result.csv', average_csv_path)
if os.path.exists('resnet_results_sorted.csv'):os.remove('resnet_results_sorted.csv')
if os.path.exists('resnet_results.csv'):os.remove('resnet_results.csv')
if os.path.exists('average_result.csv'):os.remove('average_result.csv')