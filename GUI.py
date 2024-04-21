import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import clustering

def browse_file():
    filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if filename:
        entry_file_path.delete(0, tk.END)
        entry_file_path.insert(tk.END, filename)

def run_clustering():
    file_path = entry_file_path.get()
    percentage = float(entry_percentage.get())
    k = int(entry_clusters.get())
    threshold = float(3)

    try:
        dataset = pd.read_csv(file_path)
        X = dataset[['IMDB Rating']].values
        num_records = int(len(X) * (percentage / 100))
        X = X[:num_records]
        clusters, centroids, outliers, outlier_indices = clustering.kmeans(X, k, threshold)  # Adjust function call

        text_output.delete(1.0, tk.END)  # Clear previous output

        text_output.insert(tk.END, "Outlier records:\n")
        for idx in outlier_indices:
            text_output.insert(tk.END, f"{dataset.iloc[idx]['Movie Name']} {dataset.iloc[idx]['IMDB Rating']}\n")
        text_output.insert(tk.END, "\n\n")

        for cluster_id in range(k):
            text_output.insert(tk.END, f"Cluster {cluster_id + 1}:\n")
            cluster_indices = np.where(np.array(clusters) == cluster_id)[0]
            for idx in cluster_indices:
                text_output.insert(tk.END, f"{dataset.iloc[idx]['Movie Name']} {dataset.iloc[idx]['IMDB Rating']}\n")
            text_output.insert(tk.END, "\n\n")
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
root = tk.Tk()
root.title("K-Means Clustering with Z-Score Outlier Detection")

label_file_path = tk.Label(root, text="CSV File Path:")
label_file_path.grid(row=0, column=0)
entry_file_path = tk.Entry(root, width=20)
entry_file_path.grid(row=0, column=1)
button_browse = tk.Button(root, text="Browse", command=browse_file)
button_browse.grid(row=0, column=2)

label_percentage = tk.Label(root, text="Percentage of Records to Analyze:")
label_percentage.grid(row=1, column=0)
entry_percentage = tk.Entry(root)
entry_percentage.grid(row=1, column=1)

label_clusters = tk.Label(root, text="Number of Clusters (k):")
label_clusters.grid(row=2, column=0)
entry_clusters = tk.Entry(root)
entry_clusters.grid(row=2, column=1)


button_run = tk.Button(root, text="...Run...", command=run_clustering)
button_run.grid(row=4, column=0, columnspan=3)

text_output = tk.Text(root, height=20, width=60)
text_output.grid(row=5, column=0, columnspan=3)

root.mainloop()