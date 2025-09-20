"""
Visualization Utilities for CORD-19 Analysis
Helper functions for creating charts and plots
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from collections import Counter

class VisualizationHelper:
    """Utility class for creating visualizations"""
    
    def __init__(self, style='default'):
        """Initialize with plotting style"""
        plt.style.use(style)
        sns.set_palette("husl")
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def create_year_distribution(self, df, title="Publications by Year"):
        """Create year distribution plot"""
        if 'year' not in df.columns:
            print("Year column not found")
            return None
        
        year_counts = df['year'].value_counts().sort_index()
        
        # Matplotlib version
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(year_counts.index, year_counts.values, 
                     color=self.colors[0], alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Publications', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + max(year_counts.values) * 0.01,
                       f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    def create_journal_distribution(self, df, top_n=10, title="Top Publishing Journals"):
        """Create journal distribution plot"""
        if 'journal' not in df.columns:
            print("Journal column not found")
            return None
        
        journal_counts = df['journal'].value_counts().head(top_n)
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(range(len(journal_counts)), journal_counts.values, 
                      color=self.colors[1], alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Truncate long journal names
        labels = [j[:50] + '...' if len(j) > 50 else j for j in journal_counts.index]
        ax.set_yticks(range(len(journal_counts)))
        ax.set_yticklabels(labels)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Number of Papers', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + max(journal_counts.values) * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{int(width)}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def create_word_frequency_plot(self, word_freq, title="Most Frequent Words"):
        """Create word frequency plot"""
        if not word_freq:
            print("No word frequency data provided")
            return None
        
        words, counts = zip(*word_freq)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(range(len(words)), counts, 
                      color=self.colors[2], alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Frequency', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + max(counts) * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{int(width)}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def create_abstract_length_distribution(self, df, title="Abstract Length Distribution"):
        """Create abstract length distribution"""
        if 'abstract_word_count' not in df.columns:
            print("Abstract word count column not found")
            return None
        
        word_counts = df['abstract_word_count']
        word_counts = word_counts[word_counts > 0]  # Remove zero counts
        
        if len(word_counts) == 0:
            print("No abstract data available")
            return None
        
        # Remove extreme outliers (top 5%)
        word_counts = word_counts[word_counts <= word_counts.quantile(0.95)]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        n, bins, patches = ax.hist(word_counts, bins=50, color=self.colors[3], 
                                  alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Word Count', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_words = word_counts.mean()
        median_words = word_counts.median()
        ax.axvline(mean_words, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_words:.1f}')
        ax.axvline(median_words, color='orange', linestyle='--', alpha=0.8, label=f'Median: {median_words:.1f}')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def create_source_distribution(self, df, title="Papers by Source"):
        """Create source distribution pie chart"""
        if 'source_x' not in df.columns:
            print("Source column not found")
            return None
        
        source_counts = df['source_x'].value_counts().head(8)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        wedges, texts, autotexts = ax.pie(source_counts.values, labels=source_counts.index, 
                                         autopct='%1.1f%%', startangle=90, colors=self.colors)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Improve text readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        return fig
    
    def create_comprehensive_dashboard(self, df, word_freq=None):
        """Create a comprehensive dashboard with multiple plots"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('CORD-19 Dataset Analysis Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Publications by year
        if 'year' in df.columns:
            year_counts = df['year'].value_counts().sort_index()
            year_counts = year_counts[year_counts.index >= 2000]  # Filter recent years
            
            axes[0, 0].bar(year_counts.index, year_counts.values, 
                          color=self.colors[0], alpha=0.7)
            axes[0, 0].set_title('Publications by Year', fontweight='bold')
            axes[0, 0].set_xlabel('Year')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Top journals
        if 'journal' in df.columns:
            journal_counts = df['journal'].value_counts().head(8)
            axes[0, 1].barh(range(len(journal_counts)), journal_counts.values, 
                           color=self.colors[1], alpha=0.7)
            axes[0, 1].set_yticks(range(len(journal_counts)))
            axes[0, 1].set_yticklabels([j[:25] + '...' if len(j) > 25 else j 
                                       for j in journal_counts.index])
            axes[0, 1].set_title('Top Journals', fontweight='bold')
            axes[0, 1].set_xlabel('Count')
        
        # 3. Word frequency
        if word_freq:
            words, counts = zip(*word_freq[:10])
            axes[0, 2].barh(range(len(words)), counts, 
                           color=self.colors[2], alpha=0.7)
            axes[0, 2].set_yticks(range(len(words)))
            axes[0, 2].set_yticklabels(words)
            axes[0, 2].set_title('Top Words in Titles', fontweight='bold')
            axes[0, 2].set_xlabel('Frequency')
        
        # 4. Abstract length distribution
        if 'abstract_word_count' in df.columns:
            word_counts = df['abstract_word_count']
            word_counts = word_counts[(word_counts > 0) & (word_counts <= word_counts.quantile(0.95))]
            
            if len(word_counts) > 0:
                axes[1, 0].hist(word_counts, bins=30, color=self.colors[3], alpha=0.7)
                axes[1, 0].set_title('Abstract Length Distribution', fontweight='bold')
                axes[1, 0].set_xlabel('Word Count')
                axes[1, 0].set_ylabel('Frequency')
        
        # 5. Papers by source
        if 'source_x' in df.columns:
            source_counts = df['source_x'].value_counts().head(6)
            axes[1, 1].pie(source_counts.values, labels=source_counts.index, 
                          autopct='%1.1f%%', colors=self.colors)
            axes[1, 1].set_title('Papers by Source', fontweight='bold')
        
        # 6. Monthly distribution (if year data available)
        if 'month' in df.columns and 'year' in df.columns:
            recent_data = df[df['year'] >= 2020]  # Focus on recent years
            if len(recent_data) > 0:
                month_counts = recent_data['month'].value_counts().sort_index()
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                axes[1, 2].bar(month_counts.index, month_counts.values, 
                              color=self.colors[4], alpha=0.7)
                axes[1, 2].set_title('Publications by Month (2020+)', fontweight='bold')
                axes[1, 2].set_xlabel('Month')
                axes[1, 2].set_ylabel('Count')
                axes[1, 2].set_xticks(range(1, 13))
                axes[1, 2].set_xticklabels(month_names, rotation=45)
        
        plt.tight_layout()
        return fig
    
    def create_plotly_interactive_dashboard(self, df, word_freq=None):
        """Create interactive dashboard using Plotly"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Publications by Year', 'Top Journals', 'Word Frequency',
                           'Abstract Length', 'Papers by Source', 'Timeline'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "pie"}, {"secondary_y": False}]]
        )
        
        # 1. Publications by year
        if 'year' in df.columns:
            year_counts = df['year'].value_counts().sort_index()
            year_counts = year_counts[year_counts.index >= 2000]
            
            fig.add_trace(
                go.Bar(x=year_counts.index, y=year_counts.values, 
                      name="Publications", marker_color='lightblue'),
                row=1, col=1
            )
        
        # 2. Top journals
        if 'journal' in df.columns:
            journal_counts = df['journal'].value_counts().head(10)
            
            fig.add_trace(
                go.Bar(x=journal_counts.values, y=journal_counts.index, 
                      orientation='h', name="Journals", marker_color='lightcoral'),
                row=1, col=2
            )
        
        # 3. Word frequency
        if word_freq:
            words, counts = zip(*word_freq[:15])
            
            fig.add_trace(
                go.Bar(x=list(counts), y=list(words), 
                      orientation='h', name="Words", marker_color='lightgreen'),
                row=1, col=3
            )
        
        # 4. Abstract length distribution
        if 'abstract_word_count' in df.columns:
            word_counts = df['abstract_word_count']
            word_counts = word_counts[(word_counts > 0) & (word_counts <= word_counts.quantile(0.95))]
            
            fig.add_trace(
                go.Histogram(x=word_counts, nbinsx=30, name="Length", 
                           marker_color='lightyellow'),
                row=2, col=1
            )
        
        # 5. Papers by source (pie chart)
        if 'source_x' in df.columns:
            source_counts = df['source_x'].value_counts().head(8)
            
            fig.add_trace(
                go.Pie(labels=source_counts.index, values=source_counts.values, 
                      name="Sources"),
                row=2, col=2
            )
        
        # 6. Timeline (cumulative publications)
        if 'year' in df.columns:
            year_counts = df['year'].value_counts().sort_index()
            cumulative = year_counts.cumsum()
            
            fig.add_trace(
                go.Scatter(x=cumulative.index, y=cumulative.values, 
                          mode='lines+markers', name="Cumulative", 
                          line=dict(color='purple')),
                row=2, col=3
            )
        
        # Update layout
        fig.update_layout(
            title_text="CORD-19 Interactive Dashboard",
            title_x=0.5,
            height=800,
            showlegend=False
        )
        
        return fig
    
    def save_all_plots(self, df, word_freq=None, output_dir='plots'):
        """Save all plots to files"""
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        plots_created = []
        
        # Year distribution
        fig = self.create_year_distribution(df)
        if fig:
            fig.savefig(f'{output_dir}/year_distribution.png', dpi=300, bbox_inches='tight')
            plots_created.append('year_distribution.png')
            plt.close(fig)
        
        # Journal distribution
        fig = self.create_journal_distribution(df)
        if fig:
            fig.savefig(f'{output_dir}/journal_distribution.png', dpi=300, bbox_inches='tight')
            plots_created.append('journal_distribution.png')
            plt.close(fig)
        
        # Word frequency
        if word_freq:
            fig = self.create_word_frequency_plot(word_freq)
            if fig:
                fig.savefig(f'{output_dir}/word_frequency.png', dpi=300, bbox_inches='tight')
                plots_created.append('word_frequency.png')
                plt.close(fig)
        
        # Abstract length
        fig = self.create_abstract_length_distribution(df)
        if fig:
            fig.savefig(f'{output_dir}/abstract_length.png', dpi=300, bbox_inches='tight')
            plots_created.append('abstract_length.png')
            plt.close(fig)
        
        # Source distribution
        fig = self.create_source_distribution(df)
        if fig:
            fig.savefig(f'{output_dir}/source_distribution.png', dpi=300, bbox_inches='tight')
            plots_created.append('source_distribution.png')
            plt.close(fig)
        
        # Comprehensive dashboard
        fig = self.create_comprehensive_dashboard(df, word_freq)
        if fig:
            fig.savefig(f'{output_dir}/comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
            plots_created.append('comprehensive_dashboard.png')
            plt.close(fig)
        
        print(f"Created {len(plots_created)} plots in '{output_dir}' directory:")
        for plot in plots_created:
            print(f"  - {plot}")
        
        return plots_created

def main():
    """Test visualization utilities"""
    print("Testing Visualization Utilities")
    print("=" * 40)
    
    # Create sample data for testing
    np.random.seed(42)
    sample_data = {
        'year': np.random.choice(range(2015, 2024), 1000),
        'journal': np.random.choice(['Nature', 'Science', 'Cell', 'NEJM', 'Lancet'], 1000),
        'abstract_word_count': np.random.normal(200, 50, 1000),
        'source_x': np.random.choice(['PubMed', 'ArXiv', 'bioRxiv', 'medRxiv'], 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df['abstract_word_count'] = df['abstract_word_count'].clip(lower=50)  # Ensure positive values
    
    # Sample word frequency data
    word_freq = [('covid', 150), ('virus', 120), ('treatment', 100), ('study', 90), ('analysis', 80)]
    
    # Test visualizations
    viz = VisualizationHelper()
    
    # Test individual plots
    print("Creating test visualizations...")
    
    fig1 = viz.create_year_distribution(df)
    if fig1:
        plt.show()
        plt.close(fig1)
    
    fig2 = viz.create_journal_distribution(df)
    if fig2:
        plt.show()
        plt.close(fig2)
    
    # Test comprehensive dashboard
    fig3 = viz.create_comprehensive_dashboard(df, word_freq)
    if fig3:
        plt.show()
        plt.close(fig3)
    
    print("Visualization utilities test completed!")

if __name__ == "__main__":
    main()
