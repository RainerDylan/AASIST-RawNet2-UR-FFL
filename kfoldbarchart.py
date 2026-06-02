import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
def plot_kfold_eer(dataset_name, eer_data, filename, bar_color):
    # Setup labels for the 6 bars
    labels = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Mean']
    
    # Calculate Mean and Standard Deviation
    mean_eer = np.mean(eer_data)
    std_eer = np.std(eer_data)
    
    # Append the mean to the data array for the 6th bar
    data_to_plot = eer_data + [mean_eer]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create the bar chart
    bars = ax.bar(labels, data_to_plot, color=bar_color, alpha=0.8, edgecolor='black')
    
    # Visually separate the 'Mean' bar with a gray color
    bars[-1].set_color('gray')
    bars[-1].set_edgecolor('black')
    
    # Add the horizontal dotted line for the mean EER
    ax.axhline(y=mean_eer, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_eer:.2f}%')
    
    # Add the text box for the Standard Deviation
    ax.text(0.05, 0.95, f'Std Dev: ±{std_eer:.2f}%', transform=ax.transAxes, 
            fontsize=12, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Display the exact EER values on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    # Format the chart labels and title
    ax.set_title(f'5-Fold Cross Validation EER Results - {dataset_name} Dataset', fontsize=14)
    ax.set_ylabel('Equal Error Rate (EER %)', fontsize=12)
    
    # Increase Y-axis limit slightly to fit the text labels
    ax.set_ylim(0, max(data_to_plot) * 1.2)  
    ax.legend(loc='upper right')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Successfully saved {filename}")

# Extracted EER data
eer_la = [8.2571, 10.0, 8.8571, 9.4571, 6.0286]
eer_df = [29.0009, 27.8409, 29.8769, 28.4801, 26.2074]

# Generate both plots
plot_kfold_eer('ASVspoof 2019 LA', eer_la, 'LA_KFold_EER.png', 'skyblue')
plot_kfold_eer('ASVspoof 2021 DF', eer_df, 'DF_KFold_EER.png', 'lightcoral')