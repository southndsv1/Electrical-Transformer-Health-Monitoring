#!/usr/bin/env python3
"""
Visualize Oil Temperature vs Date for ETT datasets.
Generates publication-quality plots showing temperature patterns over time.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11


def load_and_plot_oil_temperature(dataset_name='ETTh2', save_path=None):
    """
    Load ETT dataset and create Oil Temperature vs Date visualization.

    Args:
        dataset_name: 'ETTh1' or 'ETTh2'
        save_path: Path to save the figure (optional)
    """
    # Load data
    data_file = f'{dataset_name}.csv'
    print(f"Loading {data_file}...")

    df = pd.read_csv(data_file)
    df['date'] = pd.to_datetime(df['date'])

    print(f"Data loaded: {len(df)} samples")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"OT range: [{df['OT'].min():.2f}, {df['OT'].max():.2f}] °C")

    # Create figure with multiple views
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Complete time series
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df['date'], df['OT'], linewidth=0.8, alpha=0.8, color='#2E86AB')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Oil Temperature (°C)', fontsize=12)
    ax1.set_title(f'{dataset_name}: Oil Temperature Time Series (Complete Dataset)',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add statistics text
    stats_text = f'Mean: {df["OT"].mean():.2f}°C | Std: {df["OT"].std():.2f}°C | Min: {df["OT"].min():.2f}°C | Max: {df["OT"].max():.2f}°C'
    ax1.text(0.5, 0.97, stats_text, transform=ax1.transAxes,
             ha='center', va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. First month detail
    first_month = df[df['date'] < df['date'].min() + pd.Timedelta(days=30)]
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(first_month['date'], first_month['OT'], linewidth=1.5,
             marker='o', markersize=2, alpha=0.8, color='#A23B72')
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Oil Temperature (°C)', fontsize=11)
    ax2.set_title('First Month Detail', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 3. Daily pattern (average hourly profile)
    df['hour'] = df['date'].dt.hour
    hourly_avg = df.groupby('hour')['OT'].agg(['mean', 'std', 'min', 'max'])

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(hourly_avg.index, hourly_avg['mean'], linewidth=2.5,
             marker='o', markersize=6, color='#F18F01', label='Mean')
    ax3.fill_between(hourly_avg.index,
                     hourly_avg['mean'] - hourly_avg['std'],
                     hourly_avg['mean'] + hourly_avg['std'],
                     alpha=0.3, color='#F18F01', label='±1 Std Dev')
    ax3.set_xlabel('Hour of Day', fontsize=11)
    ax3.set_ylabel('Oil Temperature (°C)', fontsize=11)
    ax3.set_title('Daily Temperature Pattern (Hourly Average)', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(0, 24, 3))
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best', fontsize=9)

    # 4. Monthly pattern
    df['month'] = df['date'].dt.month
    monthly_avg = df.groupby('month')['OT'].agg(['mean', 'std', 'min', 'max'])

    ax4 = fig.add_subplot(gs[2, 0])
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    x_pos = range(1, 13)
    bars = ax4.bar(x_pos, monthly_avg['mean'], alpha=0.7, color='#06A77D',
                   edgecolor='black', linewidth=1)
    ax4.errorbar(x_pos, monthly_avg['mean'], yerr=monthly_avg['std'],
                fmt='none', color='black', capsize=5, linewidth=1.5, alpha=0.7)
    ax4.set_xlabel('Month', fontsize=11)
    ax4.set_ylabel('Oil Temperature (°C)', fontsize=11)
    ax4.set_title('Monthly Temperature Pattern', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(month_names, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')

    # Color bars by temperature
    colors = plt.cm.RdYlBu_r(monthly_avg['mean'] / monthly_avg['mean'].max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # 5. Temperature distribution
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.hist(df['OT'], bins=50, alpha=0.7, color='#C73E1D',
             edgecolor='black', linewidth=0.5, density=True)
    ax5.axvline(df['OT'].mean(), color='blue', linestyle='--', linewidth=2,
               label=f'Mean: {df["OT"].mean():.2f}°C')
    ax5.axvline(df['OT'].median(), color='green', linestyle='--', linewidth=2,
               label=f'Median: {df["OT"].median():.2f}°C')

    # Add critical temperature line (95°C for transformers)
    if df['OT'].max() > 70:  # Only if we have high temps
        ax5.axvline(95, color='red', linestyle='--', linewidth=2, alpha=0.7,
                   label='Critical: 95°C')

    ax5.set_xlabel('Oil Temperature (°C)', fontsize=11)
    ax5.set_ylabel('Density', fontsize=11)
    ax5.set_title('Temperature Distribution', fontsize=12, fontweight='bold')
    ax5.legend(loc='best', fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')

    # Overall title
    fig.suptitle(f'{dataset_name} Dataset: Electrical Transformer Oil Temperature Analysis',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save figure
    if save_path is None:
        save_path = f'{dataset_name}_Oil_Temperature_Analysis.png'

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")

    # Also create a simple version for quick view
    fig_simple, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df['date'], df['OT'], linewidth=1, alpha=0.8, color='#2E86AB')
    ax.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax.set_ylabel('Oil Temperature (°C)', fontsize=13, fontweight='bold')
    ax.set_title(f'{dataset_name}: Oil Temperature vs Date',
                fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add statistics box
    stats_box = f'Statistics:\nMean: {df["OT"].mean():.2f}°C\nStd: {df["OT"].std():.2f}°C\nMin: {df["OT"].min():.2f}°C\nMax: {df["OT"].max():.2f}°C'
    ax.text(0.02, 0.98, stats_box, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()

    simple_path = f'{dataset_name}_Oil_Temperature_vs_Date.png'
    plt.savefig(simple_path, dpi=300, bbox_inches='tight')
    print(f"Simple visualization saved to: {simple_path}")

    plt.close('all')

    return df


def main():
    """Main function to generate all visualizations."""
    print("="*80)
    print("Oil Temperature Visualization")
    print("="*80)

    # Generate for both datasets
    for dataset in ['ETTh1', 'ETTh2']:
        print(f"\n{'='*80}")
        print(f"Processing {dataset}")
        print(f"{'='*80}")

        try:
            df = load_and_plot_oil_temperature(dataset)

            # Print summary statistics
            print(f"\n{dataset} Summary Statistics:")
            print(f"  Total samples: {len(df)}")
            print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"  Duration: {(df['date'].max() - df['date'].min()).days} days")
            print(f"  Temperature range: [{df['OT'].min():.2f}, {df['OT'].max():.2f}] °C")
            print(f"  Mean temperature: {df['OT'].mean():.2f} °C")
            print(f"  Std deviation: {df['OT'].std():.2f} °C")

            # Check for extreme temperatures
            high_temp = df[df['OT'] > 40]
            if len(high_temp) > 0:
                print(f"  High temperature events (>40°C): {len(high_temp)} ({len(high_temp)/len(df)*100:.2f}%)")

            critical_temp = df[df['OT'] > 95]
            if len(critical_temp) > 0:
                print(f"  ⚠️  CRITICAL temperature events (>95°C): {len(critical_temp)}")
            else:
                print(f"  ✓ No critical temperature events (all <95°C)")

        except Exception as e:
            print(f"Error processing {dataset}: {e}")

    print("\n" + "="*80)
    print("✅ All visualizations generated successfully!")
    print("="*80)
    print("\nGenerated files:")
    print("  - ETTh1_Oil_Temperature_Analysis.png (comprehensive)")
    print("  - ETTh1_Oil_Temperature_vs_Date.png (simple)")
    print("  - ETTh2_Oil_Temperature_Analysis.png (comprehensive)")
    print("  - ETTh2_Oil_Temperature_vs_Date.png (simple)")
    print("="*80)


if __name__ == '__main__':
    main()
